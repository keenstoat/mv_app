
import tensorrt as trt
import cvcuda
import torch
import numpy as np
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager
from collections.abc import Iterator, Callable
import supervision as sv


from .utils import ValidationError, MsgType
from .frame import Frame

trt.init_libnvinfer_plugins(None, "") # pyright: ignore

class Detections:
    
    def __init__(self, 
        xyxy:np.ndarray, 
        conf:np.ndarray|None=None, 
        class_id:np.ndarray|None=None, 
        tracker_id:np.ndarray|None=None ):
            
        self.xyxy:np.ndarray = xyxy
        self.conf:np.ndarray = conf # pyright: ignore
        self.class_id:np.ndarray = class_id # pyright: ignore
        self.tracker_id:np.ndarray = tracker_id # pyright: ignore

        self.xyz:np.ndarray = None # pyright: ignore
        
    @property
    def is_empty(self):
        return self.xyxy.size == 0

    def to_sv_detections(self) -> sv.Detections:

        return sv.Detections(
            xyxy=self.xyxy,
            confidence=self.conf,
            class_id=self.class_id,
            tracker_id=self.tracker_id,
        )

    def update_with_sv_detections(self, detections:sv.Detections):

        filter = detections.tracker_id != -1
        self.xyxy = detections.xyxy[filter]
        self.conf = detections.confidence[filter] # pyright: ignore
        self.class_id = detections.class_id[filter] # pyright: ignore
        self.tracker_id = detections.tracker_id[filter] # pyright: ignore

class DetectionsBatch:

    def __init__(self, detections:list[Detections]):
        self.detections_list = detections

    def order_detctions_by_tracker_id(self):
        
        if len(self.detections_list) < 2:
            raise RuntimeError("Must have 2 or more detections in the list for ordering")

        if self.detections_list[0].tracker_id is None or self.detections_list[1].tracker_id is None:
            raise RuntimeError("Detections must have tracker ids assigned")

        # get only the tracker ids that exist in both detections, and get the index where they are
        # on each 
        tracker_id, idx_0, idx_1 = np.intersect1d(
            self.detections_list[0].tracker_id,
            self.detections_list[1].tracker_id,
            assume_unique=True, 
            return_indices=True
        ) # pyright: ignore

        # use the index of each to filter and order the arrays, and save in place
        # use the same tracker id array for both
        for detections, idx in zip(self.detections_list, [idx_0, idx_1]):
            detections.tracker_id = tracker_id
            detections.xyxy = detections.xyxy[idx]
            detections.conf = detections.conf[idx]
            detections.class_id = detections.class_id[idx]

class TensorRT:

    def __init__(self, engine_filepath:str|Path):

        self._device = torch.device("cuda")
        
        self.engine_filepath = engine_filepath
        with open(engine_filepath, "rb") as engine_file:
            try:
                # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/tensorrt.py
                # YOLO export adds a metadata block to the engine file. 
                # To load the model it has to be skipped.
                # The first 4 bytes is an integer value denoting the length of the metadata block
                meta_len = int.from_bytes(engine_file.read(4), byteorder="little")
                # if metadata is read and decoded, then this engine file was exported with YOLO
                # the read leaves the seek pointer at the start of the useful data
                engine_file.read(meta_len).decode("utf-8") 

            except UnicodeDecodeError:
                engine_file.seek(0)
            
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) # pyright: ignore
            self._engine = runtime.deserialize_cuda_engine(engine_file.read())

        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_filepath}")

        self._context = self._engine.create_execution_context()
        self._stream:torch.cuda.Stream = None

        # Configure the Input tensor ----------------------------------------
        # Expected Input tensor (0) shape format is NCHW
        self._input_tensor_name = self._engine.get_tensor_name(0)
        self._input_tensor_NCHW = tuple(self._engine.get_tensor_shape(self._input_tensor_name))

        # Dynamic models have an input tensor shape as (-1, 3, -1, -1), because the batch, height and width
        # of the image has a range. This range is defined by a profile.
        # Static models have a clear input tensor shape, ex: (2, 3, 640, 640)
        self._is_dynamic = -1 in self._input_tensor_NCHW
        
        if self._is_dynamic:
            # TRT models have 3 profiles for accepting input tensors based on the shape. Example:
            # [(1, 3, 32, 32), (2, 3, 640, 640), (2, 3, 1280, 1280)], each tuple is a NCHW format where
            # 0=min shape, 1=optimal shape, 2=max shape
            # if model is static, all profiles are equal. 
            # if model is dynamic, optimal profile (sub 1) is a good option
            profiles = self._engine.get_tensor_profile_shape(self._input_tensor_name, 0)
            self._input_tensor_NCHW = tuple(profiles[1])
    
            # because its dynamic, this may change at inference time to match the actual input image tensor
            self._context.set_input_shape(self._input_tensor_name, self._input_tensor_NCHW)

        # Configure the Output tensor ----------------------------------------
        # yolo26 end-to-end output tensor shape is (N, 300, 6)
        # See: https://docs.ultralytics.com/guides/end2end-detection
        # If the model is dynamic then N=-1, else it has a positive value.
        output_tensor_name = self._engine.get_tensor_name(1) 
        output_tensor_shape = tuple(self._engine.get_tensor_shape(output_tensor_name)) 

        # Reject models not implementing end-to-end detection
        if output_tensor_shape[1] != 300:
            raise RuntimeError(
                f"Model {engine_filepath} does not implement end-to-end detection. "
                f"Output tensor shape is {output_tensor_shape}"
            )

         # If model is dynamic, then set the expected batch size (N) from the input tensor 
        if self._is_dynamic:
            output_tensor_shape = (self._input_tensor_NCHW[0], *output_tensor_shape[1:])

        fp16_precision = self._engine.get_tensor_dtype(output_tensor_name) == trt.DataType.HALF
        self._dtype = torch.float16 if fp16_precision else torch.float32

        # Define an output tensor where inference will always put its result.
        # This is not done for the input tensor, because the input tensor is always created from a new image
        # Also, the optimized inference works on two input tensors in parallel.
        self._output_tensor = torch.empty(output_tensor_shape, dtype=self._dtype, device=self._device)
        self._context.set_tensor_address(output_tensor_name, self._output_tensor.data_ptr())

    def _get_new_cuda_stream(self):
        return torch.cuda.Stream(device=self._device)

    @contextmanager
    def _stream_n_sync(self, stream:torch.cuda.Stream):

        if not stream:
            raise RuntimeError("Must provide a stream to use")

        with torch.cuda.stream(stream):
            yield 
        stream.synchronize()
    
    def _preprocess(self, tensor_list:list[torch.Tensor]) -> torch.Tensor:
        """
        Applies transformation to the given image tensors to prepare it for inference. 
        Uses cvcuda for image transformations.
        Input image tensors are expected in HWC format and uint8 dtype (same as np arrays).
        Outputs a new tensor with NCHW format (stacked), as expected by TensorRT
        """

        if len(tensor_list) > self._input_tensor_NCHW[0]:
            raise RuntimeError(
                f"Max amount of tensor images allowed by model is {self._input_tensor_NCHW[0]}. "
                f"{len(tensor_list)} were sent for inference."
            )

        # apply a letterboxed resize to the image tensors to result in square images
        # keeping the top-left corner aligned. 
        cvcuda_tensor_list:list[cvcuda.Tensor] = []

        for tensor in tensor_list:
            h, w = tensor.shape[:2]
            max_side = max(h, w)
            cvcuda_tensor = cvcuda.copymakeborder(
                src=cvcuda.as_tensor(tensor, "HWC"),
                top=0,
                bottom=max_side - h,
                left=0,
                right=max_side - w,
                border_mode=cvcuda.Border.CONSTANT,
                border_value=[114, 114, 114], # padding color used by yolo
            )
            cvcuda_tensor_list.append(cvcuda_tensor)
        
        # if model is static and expects more input tensors than images provided,
        # add blank images as padding with the same shape as the first image
        missing_images_n = self._input_tensor_NCHW[0] - len(cvcuda_tensor_list)

        if not self._is_dynamic and missing_images_n > 0:
            for _ in range(missing_images_n):
                pad = torch.full((max_side, max_side, 3), 114, dtype=torch.uint8, device=self._device)
                cvcuda_tensor_list.append(cvcuda.as_tensor(pad, "HWC"))

        # stack the list to create a tensor with layout NHWC
        cvcuda_tensor = cvcuda.stack(cvcuda_tensor_list)
        
        h, w = self._input_tensor_NCHW[2:] # h and w of the expected tensor. default yolo is 640x640

        # do all the transforms in a single go
        cvcuda_tensor = cvcuda.resize_crop_convert_reformat(
            cvcuda_tensor,
            (h, w),
            cvcuda.Interp.LINEAR,
            cvcuda.RectI(0, 0, w, h),   # no cropping requried
            layout="NCHW",              # reformat from NHWC to NCHW, as tensorRT expects
            scale=1 / 255,              # normlize
            data_type=cvcuda.Type.F32,
        ) # pyright: ignore

        # return a torch tensor for compatibility. ToDo: check if a cvcuda tensor can be returned
        return torch.as_tensor(cvcuda_tensor.cuda(), device=self._device).contiguous()
 
    def _inference(self, input_tensor:torch.Tensor, stream:torch.cuda.Stream) -> torch.Tensor:
        """
        Runs inference over the input tensor on the given stream.
        The stream must be syncronized by the caller before fetching the output tensor values.
        """
        in_N, in_C, in_H, in_W = input_tensor.shape # given tensor shape
        ref_N, ref_C, ref_H, ref_W = self._input_tensor_NCHW # expected tensor shape

        if in_N > ref_N or in_C != ref_C or in_H != ref_H or in_W != ref_W:
            raise RuntimeError(
                "Input tensor shape exceeds allowed model shape: "
                f"{tuple(input_tensor.shape)} > {self._input_tensor_NCHW}"
            )

        # if model is dynamic, adjust the shape of the input tensor shape, so cuda knows how much to read
        # Also adjust the batch size (N) of the output tensor, to match the input tensor batch size
        # the output tensor is of shape (N, 300, 6)
        if self._is_dynamic:

            self._context.set_input_shape(self._input_tensor_name, input_tensor.shape)

            # Resizing the output tensor is not required
            # output_tensor_shape = (in_N, *self._output_tensor.shape[1:])
            # self._output_tensor.resize_(output_tensor_shape)

        self._context.set_tensor_address(self._input_tensor_name, input_tensor.data_ptr())

        self._context.execute_async_v3(stream_handle=stream.cuda_stream)

        # output tensor requires stream sync to be released!
        # ref: https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/_static/python-api/infer/Core/ExecutionContext.html#tensorrt.IExecutionContext.execute_async_v3
        
        return self._output_tensor

    def _postprocess(self, 
        output_tensor:torch.Tensor, 
        tensor_list:list[torch.Tensor], 
        conf:float=0.25) -> list[dict[str, np.ndarray]]:
        """
        Transforms the output tensor from yolo end-to-end predictions to a dictionary
        Args:
            output_tensor: the tensor returned by the inference
            tensor_list: the tensor list passed into preprocessing. Used to rescale bounding boxes.
            conf: minimum confidence score of included detections
        """
        # yolo end2end model output_tensors.shape is (N, 300, 6) where:
        #   N is the batch size
        #   300 is the number of detections (a row per detection)
        #   6 holds the bbox=0,1,2,3; conf=4, class id=5
        # the class id are floats, so gotta turn them to ints
    
        results = []
        for batch_idx in range(len(tensor_list)):

            # filter out predictions whose conf score is lower than the requested confidence
            predictions = output_tensor[batch_idx]
            scores = predictions[:, 4]
            predictions = predictions[scores >= conf]

            if predictions.shape[0] == 0:
                results.append({
                    "xyxy": np.empty((0, 4)), 
                    "conf": np.empty((0,)), 
                    "class_id": np.empty((0,))
                })
                continue
            
            predictions = predictions.cpu().numpy()
            
            tensor_image_HW = self._input_tensor_NCHW[2:]
            original_image_HW = tensor_list[batch_idx].shape[:2] # image tensors have HWC layout (as np array images)
            original_image_max = max(original_image_HW) # input images were letterboxed using the max side
            
            # rescale the bboxes in place ---------
            # calculate scale factor array
            target_shape_wh = np.array([original_image_max] * 2)
            input_shape_wh = np.flip(tensor_image_HW)
            scale_xy = target_shape_wh / input_shape_wh

            # scale back to original size
            predictions[:, 0:4] *= np.concatenate((scale_xy, scale_xy)) 

            # clip max and mins 
            max_clip = np.concatenate((target_shape_wh, target_shape_wh))
            min_clip = np.array([0, 0, 0, 0])
            predictions[:, 0:4].clip(min=min_clip, max=max_clip, out=predictions[:, 0:4])
            
            results.append({
                "xyxy": predictions[:, 0:4],
                "conf": predictions[:, 4],
                "class_id":  predictions[:, 5].astype(int),
            })
        return results

    def predict(self, frame:Frame, conf:float=0.25):
        """
        Runs the preprocessing, inference, and postprocessing over the given images.
        The prediction steps are applied in a sequence.
        Populates the detections of the frame
        """
        if not frame.image_tensors:
            raise ValidationError("Could not get image tensors to preprocess", MsgType.WARNING)

        self._stream = self._stream or self._get_new_cuda_stream()

        with frame.time_it("preprocess"), self._stream_n_sync(self._stream):
            input_tensor = self._preprocess(frame.image_tensors)
            
        with frame.time_it("inference"), self._stream_n_sync(self._stream):
            output_tensor = self._inference(input_tensor, stream=self._stream)
            
        with frame.time_it("postprocess"), self._stream_n_sync(self._stream):
            batch_detections = self._postprocess(output_tensor, frame.image_tensors, conf=conf)

        dets = DetectionsBatch([Detections(**det) for det in batch_detections])
        frame.detections_batch = dets

    def predict_optimized(self, 
        get_next_frame:Callable[[], Frame], 
        conf:float=0.25) -> Iterator[Frame]:
        
        inference_stream = self._get_new_cuda_stream()
        preprocessing_stream = self._get_new_cuda_stream()
        
        input_tensor:torch.Tensor = None # pyright: ignore
        frame:Frame = None # pyright: ignore

        def schedule_preprocessing():
            """
            Schedules the preprocessing of the next batch of images to the gpu
            """
            nonlocal input_tensor, frame
            
            frame = get_next_frame()
            
            if not frame.is_valid:
                return False

            # send the images to gpu but don't sync yet
            with torch.cuda.stream(preprocessing_stream):
                input_tensor = self._preprocess(frame.image_tensors)
            return True

        # send a first batch of images for preprocessing
        scheduling_success = schedule_preprocessing()
        if not scheduling_success:
            raise ValidationError("Could not get an image batch to start preprocessing", MsgType.WARNING)

        while True:

            with frame.time_it("pre+inference"):
                # wait for the preprocessing to complete
                preprocessing_stream.synchronize()
                
                # register an event to flag when input tensor A has been consumed by the inference
                consume_event = torch.cuda.Event()
                self._context.set_input_consumed_event(consume_event.cuda_event)
    
                # send the input tensor for inference and wait until it consumes it
                # inference keeps running after consuming the input tensor
                output_tensor = self._inference(input_tensor, stream=inference_stream)
                consume_event.synchronize()

                # once the inference has consumed the input tensor,
                # send the next image batch for preprocessing
                # This sets a new value for input_tensor and images immediately
                # as it does not wait for the preprocessing to finish.
                # so we need to reference current images before scheduling preproc.
                # NOTE: this is not part of the time taken for preprocessing and inference, because
                #       it only accounts for the time grabing and rectifying the next frame, 
                #       and this is already measured elsewhere.
                t_sp = perf_counter()
                current_frame = frame
                got_next_frame = schedule_preprocessing()
                t_sp = (perf_counter() - t_sp) * 1000
                frame.perf["pre+inference"] = -t_sp

                # now wait for the inference to complete on input tensor A
                inference_stream.synchronize()

            # with inference complete, do postprocessing. 
            # This does not need an explicit cuda sync, bc tensor is pulled to numpy
            with frame.time_it("postprocess"):
                batch_detections = self._postprocess(output_tensor, current_frame.image_tensors, conf=conf)

            dets_batch = DetectionsBatch([Detections(**dets) for dets in batch_detections])
            current_frame.detections_batch = dets_batch
            yield current_frame

            if not got_next_frame:
                raise ValidationError("Could not get a next image to preprocess", MsgType.WARNING)


if __name__ == "__main__":

    _box_ant = sv.BoxAnnotator(color=sv.Color(0,255,0))
    def annotate(img:np.ndarray, detections:Detections):

        if not detections:
            return img

        _box_ant.thickness = max(1, int(max(img.shape[0:2]) * 0.004))
        _box_ant.annotate(img, detections=detections.to_sv_detections())

        return img

    import cv2
    _source_images_dir_path = Path("/home/charles/repos/mv_app/_test_images/src_images")
    _output_images_dir_path = Path("/home/charles/repos/mv_app/_test_images/out_images")
    _yolo_engine_path = Path("/home/charles/repos/mv_app/_models/yolo26s_dynamic_batch2.engine")
    
    def image_generator():
        
        for img_path in _source_images_dir_path.iterdir():
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            yield str(img_path), cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # pyright: ignore 

    def predict_sequential(bboxes=False, bench=False, batch_n=1):
        
        model = TensorRT(_yolo_engine_path)
        img_gen = image_generator()
        

        if bboxes:
            print("bboxes...")
            repeats = 10
            
            for _id in range(repeats):
                _, image = next(img_gen)
                dets_batch = model.predict_sequential([image])
                annotate(image, dets_batch.detections_list[0])
                cv2.imwrite(str(_output_images_dir_path / f"0_{_id}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
        if bench:
            _, image = next(img_gen)
            for _ in range(15):
                img_batch = [image] * batch_n
                model.predict_sequential(img_batch)
                
            print("benchmark...")
            speeds = np.array([0, 0, 0, 0], dtype=np.float32())
            repeats = 40
            t0 = perf_counter()
            for _id in range(repeats):
                _, image = next(img_gen)
                img_batch = [image] * batch_n
                dets_batch = model.predict_sequential(img_batch)
                speeds += dets_batch.speeds2array()
    
            time_all = (perf_counter() - t0)
            speeds /= repeats
            print(f"Mean speeds per image - {repeats} samples - batch {len(img_batch)}")
            print(f"- preprocess_ms: {speeds[0]:.2f}")
            print(f"- inference_ms: {speeds[1]:.2f}")
            print(f"- postprocessing_ms: {speeds[2]:.2f}")
            print(f"Total per op: {np.sum(speeds):.2f}")
            print(f"Sample size {repeats}: {time_all:.2f} s - {(repeats / time_all):.2f} fps")

    def predict_optimized(bboxes=False, bench=False, batch_n=1):

        model = TensorRT(_yolo_engine_path)
        img_gen = image_generator()
        img_count = 0
        img_limit = 50
        def get_next_image_batch():
            nonlocal img_count
            if img_count == img_limit:
                return []
            
            img_count += 1
            _, img = next(img_gen)
            if img is None:
                return []
            
            return [img] * batch_n

        if bboxes:
            print("bboxes...")
            predictor = model.predict_generator(get_next_image_batch)
            repeats = 49
            
            for _id in range(repeats):
                dets_batch, images = next(predictor)
                image = images[0]
                annotate(image, dets_batch.detections_list[0])
                cv2.imwrite(str(_output_images_dir_path / f"0_{_id}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if bench:
            # warm up
            _, image = next(img_gen)
            def get_same_image():
                return [image] * batch_n
                
            predictor = model.predict_generator(get_same_image)
            for _ in range(10):
                dets_batch, _ = next(predictor)
                
            print("benchmark...")
            t0 = perf_counter()
            predictor = model.predict_generator(get_next_image_batch)
            repeats = 48
            for _ in range(repeats):
                dets_batch, _ = next(predictor)
    
            time_all = (perf_counter() - t0)
            print(f"Sample size {repeats}: {time_all * 1000:.2f} ms - {(repeats / time_all):.2f} fps")


    # predict_sequential(bench=True, batch_n=2)
    # print()
    predict_optimized(bench=True, batch_n=2)