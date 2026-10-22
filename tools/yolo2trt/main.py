import os
# Ultralytics shows a requirement  for ['tensorrt-cu13>=10.15,<11,!=10.2.0'] not found, 
# but tensorrt is/should be accessible from the site system packages
# This prevents ultralytics from trying to install it
os.environ["YOLO_AUTOINSTALL"] = "False"

from pathlib import Path
from ultralytics.models.yolo.model import YOLO



def yolo2trt(model_filepath:str):

    model = YOLO(model_filepath)
    model.export(
        # imgsz=640,
        format="engine", # TensorRT engine
        dynamic=False,  # Not required, TRT inference works fine when input tensor batch is lower than batch size
                        # dynamic also allows variable image sizes, but this is not required either
        batch=2,        # to run inference on 2 images at once
        workspace=2,
        quantize=16,      # FP16 precision for faster inference 
        simplify=True,  # onnx graph simplification for easier export
        device='cuda',
        nms=False, # required to have the model do nms and output the final detections. See: https://docs.ultralytics.com/guides/end2end-detection
    )

def remove_metadata(model_filepath:str):
    
    out_filepath = Path(model_filepath)
    out_filepath = out_filepath.with_stem(out_filepath.stem + "-out")
    with open(model_filepath, "rb") as in_model_file, open(out_filepath, "wb") as out_file:
        try:
            # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/tensorrt.py
            # YOLO export adds a metadata block to the engine file. 
            # To load the model it has to be skipped.
            meta_len = int.from_bytes(in_model_file.read(4), byteorder="little")
            # if metadata is decoded, then this engine file was exported with YOLO
            in_model_file.read(meta_len).decode("utf-8") 
        except UnicodeDecodeError:
            in_model_file.seek(0)
        
        while True:
            chunk = in_model_file.read(65536)
            if not chunk:
                break 
            out_file.write(chunk)
            
    return out_filepath

def pt2onnx(pt_model_filepath:str):

    
    # 1. Load your YOLO26 PyTorch model weights
    # Ensure your model class structure/definitions are loaded if using a custom build
    ckpt = torch.load(pt_model_filepath, map_location="cuda", weights_only=False)
    model = ckpt['model'] if isinstance(ckpt, dict) else ckpt.model
    model.eval()
    
    # 2. Create dummy input matching YOLO input dimensions (Batch, Channels, Height, Width)
    # Standard YOLO uses a default size of 640x640
    dummy_input = torch.randn(1, 3, 640, 640, device="cuda")

    onnx_filepath = Path(pt_model_filepath).with_suffix(".onnx")
    # 3. Export to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_filepath, 
        verbose=False, 
        opset_version=17,  # Use opset 12 or higher for YOLOv8/YOLO26 node compatibility
        input_names=["images"], 
        output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}} # Optional: enables dynamic batching
    )

    return onnx_filepath

def onnx2engine(onnx_model_filepath:str|Path):
    import tensorrt as trt
    
    # 1. Initialize core TensorRT builders and network definitions
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # EXPLICIT_BATCH is standard for parsing ONNX graphs
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    
    # 2. Configure the build optimization profile
    config = builder.create_builder_config()
    
    # TensorRT 10+ defaults to strong typing; allocation memory is handled internally
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # Optional: Set 2GB workspace limit
    
    # 3. Read and Parse the ONNX file
    with open(onnx_model_filepath, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
    
    # 4. Configure Dynamic Shapes using an Optimization Profile
    profile = builder.create_optimization_profile()
    
    # Identify your input layer name (e.g., 'images' from your PyTorch export)
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    
    # Define dimensions: (Batch, Channels, Height, Width)
    # You must provide the Minimum, Optimal (most frequent), and Maximum sizes.
    min_shape = (1, 3, 640, 640)
    opt_shape = (2, 3, 640, 640)
    max_shape = (2, 3, 640, 640)  # Allows batch sizes from 1 up to 4
    
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    # 5. Enable FP16 Mode (Highly Recommended)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 precision enabled.")
    
    # 6. Build and Serialize the Engine
    print("Building TensorRT engine... This may take a few minutes.")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Engine building failed.")
        return
    
    # 7. Save the engine to a file
    engine_output_path = Path(onnx_model_filepath).with_suffix(".engine")
    with open(engine_output_path, "wb") as engine_file:
        engine_file.write(serialized_engine)
    
    print(f"Success! Saved TensorRT engine to: {engine_output_path}")

if __name__=="__main__":
    pt_model = "/home/charles/repos/mv_app/_models/yolo26s.pt"
    yolo2trt(pt_model)
    # pt_model = remove_metadata(pt_model)
    # onnx_model = pt2onnx(pt_model)
    # engine_model = onnx2engine(onnx_model)


    
    # trtexec --onnx=yolo26n.onnx \
    #         --saveEngine=yolo26n.engine \
    #         --fp16 \
    #         --minShapes=images:1x3x640x640 \
    #         --optShapes=images:1x3x640x640 \
    #         --maxShapes=images:4x3x640x640
