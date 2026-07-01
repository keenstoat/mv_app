
import tensorrt as trt
import cvcuda
import torch
import numpy as np
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager
from typing import NamedTuple
import supervision as sv
from collections.abc import Iterator
from typing import Callable

from .utils import ValidationError, MsgType

trt.init_libnvinfer_plugins(None, "")

class Speeds(NamedTuple):
    preprocess:float
    inference:float
    postprocess:float
    total:float

class Detections:
    
    def __init__(self, xyxy:np.ndarray, conf:np.ndarray, class_id:np.ndarray):
        self.xyxy = xyxy
        self.conf = conf 
        self.class_id = class_id
        
    @property
    def is_empty(self):
        return self.xyxy.size == 0

    def to_sv_detections(self) -> sv.Detections:

        return sv.Detections(
            xyxy=self.xyxy,
            confidence=self.conf,
            class_id=self.class_id
        )
    
class DetectionsBatch:

    def __init__(self, detections:list[Detections]):
        self.detections = detections
        self.speeds = Speeds._make([0, 0, 0, 0])

    def speeds2array(self):
        return np.array(list(self.speeds._asdict().values()), dtype=np.float32)

    def speeds2dict(self):
        return self.speeds._asdict()

class TensorRT():

    def __init__(self, engine_filepath:str|Path):

        self.engine_filepath = engine_filepath
        with open(engine_filepath, "rb") as engine_file:
            try:
                # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/tensorrt.py
                # YOLO export adds a metadata block to the engine file. 
                # To load the model it has to be skipped.
                meta_len = int.from_bytes(engine_file.read(4), byteorder="little")
                # if metadata is decoded, then this engine file was exported with YOLO
                engine_file.read(meta_len).decode("utf-8") 
            except UnicodeDecodeError:
                engine_file.seek(0)
            
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self._engine = runtime.deserialize_cuda_engine(engine_file.read())

        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_filepath}")

        self._context = self._engine.create_execution_context()
        self._stream:torch.cuda.Stream = None

        self._device = torch.device("cuda")

        # Expected Input tensor shape format is NCHW
        self._input_tensor_name = self._engine.get_tensor_name(0)
        self._dynamic = -1 in tuple(self._engine.get_tensor_shape(self._input_tensor_name))
        # if input is dynamic, all profiles are equal. if not, optimal profile is good
        self._input_tensor_shape = tuple(self._engine.get_tensor_profile_shape(
            self._input_tensor_name, 0
        )[1]) #0:min, 1:optimal, 2:max shape
        
        # if dynamic, this may change at inference time to match the input tensor
        self._context.set_input_shape(self._input_tensor_name, self._input_tensor_shape)

        # yolo26 end-to-end output tensor shape is (N, 300, 6)
        output_tensor_name = self._engine.get_tensor_name(1) 
        output_tensor_shape = tuple(self._engine.get_tensor_shape(output_tensor_name)) 
        if self._dynamic:
            # output batch size is the same as expected input batch size
            output_tensor_shape = tuple(
                (self._input_tensor_shape[0], *output_tensor_shape[1:])
            )

        half_precision = self._engine.get_tensor_dtype(output_tensor_name) == trt.DataType.HALF
        self._dtype = torch.float16 if half_precision else torch.float32

        self._output_tensor = torch.empty(output_tensor_shape, dtype=self._dtype, device=self._device)
        self._context.set_tensor_address(output_tensor_name, self._output_tensor.data_ptr())

    def new_stream(self):
        return torch.cuda.Stream(device=self._device)

    @contextmanager
    def stream_n_sync(self, stream:torch.cuda.Stream):

        if not stream:
            raise RuntimeError("Passed stream is None")

        with torch.cuda.stream(stream):
            yield 
        stream.synchronize()
        
    @contextmanager
    def time_it(self, speeds:list[float]):

        t0 = perf_counter()
        yield 
        speeds.append((perf_counter() - t0) * 1000)

    def preprocess_cvcuda(self, images:list[np.ndarray]) -> torch.Tensor:

        tensor_list = []
        for image in images:
            image_tensor = torch.as_tensor(image if image.flags.writeable else image.copy(), device=self._device)
            tensor_list.append(image_tensor)

        batch_pad = self._input_tensor_shape[0] - len(tensor_list)
        if not self._dynamic and batch_pad > 0:
            tensor_list.append(torch.zeros_like(tensor_list[0], device=self._device))

        # stack
        img_tensor:cvcuda.Tensor = cvcuda.stack(
            [cvcuda.as_tensor(t, "HWC") for t in tensor_list]
        )
        
        # resize - requires tensor in NHWC format
        _, _, h, w = self._input_tensor_shape # h and w of the expected shape
        n, _, _, c = img_tensor.shape # n and c of the actual image - in case a single image was input
        img_tensor:cvcuda.Tensor = cvcuda.resize(
            img_tensor, (n, h, w, c), cvcuda.Interp.LINEAR
        )
        # normalize - requires tensor in NHWC format
        img_tensor:cvcuda.Tensor = cvcuda.convertto(
            img_tensor, np.float32, scale=1 / 255
        )

        # NHWC -> NCHW
        img_tensor:cvcuda.Tensor = cvcuda.reformat(img_tensor, "NCHW") 

        img_tensor = torch.as_tensor(img_tensor.cuda(), device=self._device).contiguous()
        return img_tensor

    def preprocess_torch(self, images:list[np.ndarray]) -> torch.Tensor:

        tensor_list = [torch.as_tensor(image if image.flags.writeable else image.copy(), dtype=torch.float32, device=self._device) for image in images]
        batch_pad = self._input_tensor_shape[0] - len(tensor_list)
        if not self._dynamic and batch_pad > 0:
            tensor_list.append(torch.zeros_like(tensor_list[0], dtype=torch.float32, device=self._device))

        # stack
        img_tensor = torch.stack(tensor_list)

        # NHWC -> NCHW
        img_tensor = img_tensor.permute(0, 3, 1, 2).contiguous()

        # resize - requires float tensor and NCHW format - this resizes to the expected input tensor shape
        _, _, h, w = self._input_tensor_shape 
        img_tensor = torch.nn.functional.interpolate(
            img_tensor, size=(h, w), mode='bilinear', 
            align_corners=False, antialias=False
        )
        # img_tensor = tv_func.resize(img_tensor, (h, w), tv_func.InterpolationMode.BILINEAR, antialias=False)

        # normalize 
        img_tensor /= 255

        return img_tensor

    def inference(self, input_tensor:torch.Tensor, stream:torch.cuda.Stream) -> torch.Tensor:
        
        n1, c1, h1, w1 = input_tensor.shape
        n0, c0, h0, w0 = self._input_tensor_shape

        if n1 > n0 or c1 != c0 or h1 != h0 or w1 != w0:
            raise RuntimeError(
                "Input tensor shape exceeds allowed model shape: "
                f"{tuple(input_tensor.shape)} > {self._input_tensor_shape}"
            )
        
        if self._dynamic and n1 < n0:

            # if model is dynamic, then set the correct input tensor shape
            self._context.set_input_shape(self._input_tensor_name, input_tensor.shape)
            # Also reduce the batch size of the output tensor, in place!
            output_tensor_shape = tuple((input_tensor.shape[0], *self._output_tensor.shape[1:]))
            self._output_tensor.resize_(output_tensor_shape)

        self._context.set_tensor_address(self._input_tensor_name, input_tensor.data_ptr())

        self._context.execute_async_v3(stream_handle=stream.cuda_stream)

        # output tensor require stream sync to be released - https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/_static/python-api/infer/Core/ExecutionContext.html#tensorrt.IExecutionContext.execute_async_v3
        # stream.synchronize()
        
        return self._output_tensor

    def _rescale_boxes(self, 
        xyxy:np.ndarray|torch.Tensor, 
        input_shape_hw:tuple[int, int], 
        target_shape_hw:tuple[int, int]):

        if type(xyxy) == np.ndarray:
            target_shape_wh = np.flip(target_shape_hw)
            input_shape_wh = np.flip(input_shape_hw)
            scale_xy = target_shape_wh / input_shape_wh

            xyxy *= np.concatenate((scale_xy, scale_xy))

            max_clip = np.concatenate((target_shape_wh, target_shape_wh))
            min_clip = np.array([0, 0, 0, 0])
            xyxy.clip(min=min_clip, max=max_clip, out=xyxy)

            return

        if type(xyxy) == torch.Tensor:

            target_shape_wh = torch.tensor(target_shape_hw).flip(dims=[0])
            input_shape_wh = torch.tensor(input_shape_hw).flip(dims=[0])
            scale_xy = target_shape_wh / input_shape_wh

            xyxy *= torch.cat((scale_xy, scale_xy))

            max_clip = torch.cat((target_shape_wh, target_shape_wh))
            min_clip = torch.tensor([0, 0, 0, 0])
            xyxy.clip_(min=min_clip, max=max_clip)
            return
        
    def postprocess(self, 
        output_tensor:torch.Tensor, 
        orig_shape_nhw:tuple[int, int, int], 
        conf:float=0.25) -> list[dict[str, np.ndarray]]:

        # yolo end2end model output_tensors.shape is (N, 300, 6) where:
        #   N is the batch size
        #   300 is the number of detections (a row per detection)
        #   6 holds the bbox=0,1,2,3; conf=4, class id=5
        # the class id are floats, so gotta turn them to ints

        results = []
        for batch_id in range(orig_shape_nhw[0]):
            
            preds = output_tensor[batch_id]
            scores = preds[:, 4]
            preds = preds[scores >= conf]

            if preds.shape[0] == 0:
                results.append({
                    "xyxy": np.empty((0, 4)), 
                    "conf": np.empty((0,)), 
                    "class_id": np.empty((0,))
                })
                continue
            
            preds = preds.cpu().numpy()
            self._rescale_boxes(preds[:, 0:4], self._input_tensor_shape[2:], orig_shape_nhw[1:])
            results.append({
                "xyxy": preds[:, 0:4],
                "conf": preds[:, 4],
                "class_id":  preds[:, 5].astype(int),
            })
        return results

    def predict(self, images:list[np.ndarray], conf:float=0.25, use_cvcuda=False) -> DetectionsBatch:
        
        if not images:
            raise ValidationError("Could not get an image to preprocess", MsgType.WARNING)

        self._stream = self._stream or self.new_stream()

        speeds = []

        t0 = perf_counter()
        
        with self.time_it(speeds), self.stream_n_sync(self._stream):
            input_tensor = self.preprocess_cvcuda(images) if use_cvcuda else self.preprocess_torch(images)
            
        with self.time_it(speeds), self.stream_n_sync(self._stream):
            output_tensor = self.inference(input_tensor, stream=self._stream)
            
        with self.time_it(speeds), self.stream_n_sync(self._stream):
            orig_shape = (len(images), *images[0].shape[:2])
            detections = self.postprocess(output_tensor, orig_shape, conf=conf)

        speeds.append((perf_counter() - t0) * 1000)
        dets = DetectionsBatch([Detections(**d) for d in detections])
        dets.speeds = Speeds._make(speeds)
        return dets

    def predict_generator(self, 
        get_images:Callable[[], list[np.ndarray]]=None, 
        conf:float=0.25, 
        use_cvcuda=False) -> Iterator[tuple[DetectionsBatch, list[np.ndarray]]]:
        
        inference_stream = self.new_stream()
        preprocessing_stream = self.new_stream()
        
        buffer = {
            True: (),
            False: ()
        }
        buff_key = True

        def preprocess():
            images = get_images()
            
            if not images:
                return False

            with torch.cuda.stream(preprocessing_stream):
                input_tensor = self.preprocess_cvcuda(images) if use_cvcuda else self.preprocess_torch(images)
                buffer[buff_key] = (input_tensor, images)
            return True
            
        
        if not preprocess():
            raise ValidationError("Could not get an image to start preprocess", MsgType.WARNING)

        while True:
            
            speeds = [0]
            t0 = perf_counter()

            preprocessing_stream.synchronize()
            input_tensor, images = buffer[buff_key]
            
            consume_event = torch.cuda.Event()
            self._context.set_input_consumed_event(consume_event.cuda_event)

            t1 = perf_counter()
            output_tensor = self.inference(input_tensor, stream=inference_stream)
            consume_event.synchronize()
            t_inf = (perf_counter() - t1) * 1000
            
            got_image = preprocess()
            
            t1 = perf_counter()
            inference_stream.synchronize()
            t_inf += (perf_counter() - t1) * 1000
            speeds.append(t_inf)
            
            with self.time_it(speeds):
                orig_shape = (len(images), *images[0].shape[:2])
                detections = self.postprocess(output_tensor, orig_shape, conf=conf)

            speeds.append((perf_counter() - t0) * 1000)
            dets = DetectionsBatch([Detections(**d) for d in detections])
            dets.speeds = Speeds._make(speeds)
            yield dets, images

            if not got_image:
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
    _imgs_path = Path("/home/charles/repos/datasets/coco/images/val2017/")
    def image_generator():
        
        for img_path in _imgs_path.iterdir():
            yield str(img_path), cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) 
        yield None

    static_engine_path = Path("/home/charles/repos/mv_app/tests_scripts/yolo26s_fp16_b2_static.engine")
    yolo_static_engine_path = Path("/home/charles/repos/mv_app/tests_scripts/yolo26s_fp16_b2_static_wmeta.engine")
    yolo_dynamic_engine_path = Path("/home/charles/repos/mv_app/_models/yolo26s_fp16.engine")
    
    def do_predict():
        
        model = TensorRT(yolo_static_engine_path)
        img_gen = image_generator()

        print("warm up...")
        # for _ in range(70): next(img_gen)
        for _ in range(10):
            _, image = next(img_gen)
            model.predict([image])

        # return
        print("bboxes...")
        repeats = 10
        img_save_path = Path("/home/charles/repos/mv_app/tests_scripts/_images")
        for _id in range(repeats):
            _, image = next(img_gen)
            dets_batch = model.predict([image], use_cvcuda=False)
            annotate(image, dets_batch.detections[0])
            cv2.imwrite(str(img_save_path / f"0_{_id}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # return
        print("benchmark...")
        speeds = np.array([0, 0, 0], dtype=np.float32())
        repeats = 1000
        t0 = perf_counter()
        for _id in range(repeats):
            _, image = next(img_gen)
            img_batch = [image]
            dets_batch = model.predict(img_batch, use_cvcuda=False)
            speeds += dets_batch.speeds2array()

        time_all = (perf_counter() - t0)
        speeds /= repeats
        print(f"Mean speeds per image - {repeats} samples - batch {len(img_batch)}")
        print(f"- preprocess_ms: {speeds[0]:.2f}")
        print(f"- inference_ms: {speeds[1]:.2f}")
        print(f"- postprocessing_ms: {speeds[2]:.2f}")
        print(f"Total per op: {np.sum(speeds):.2f}")
        print(f"Sample size {repeats}: {time_all:.2f} s - {(repeats / time_all):.2f} fps")

    def do_predict_2():

        model = TensorRT(yolo_static_engine_path)
        img_gen = image_generator()

        img_count = 0
        img_limit = 5
        def get_images():
            nonlocal img_count
            if img_count == img_limit:
                return []
            
            img_count += 1
            img = next(img_gen)[1]
            if img is None:
                return []
            
            return [img]

        predictor = model.predict_generator(get_images)
        
        print("warmup...")
        try:
            for _id in range(10):
                print("ID: ", _id)
                dets, imgs = next(predictor)
                print("imgs: ", imgs[0].shape)
        except Exception as ex:
            print("EX: ", ex)
        finally:
            print("warm up done")

        return
        print("boxes...")
        img_save_path = Path("/home/charles/repos/mv_app/tests_scripts/_images")
        for _id in range(10):
            try:
                dets, imgs = next(predictor)
                image = imgs[0]
                annotate(image, dets.detections[0])
                cv2.imwrite(str(img_save_path / f"1_{_id}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            except:
                pass

        print("benchmark...")
        t0 = perf_counter()
        repeats = 1000
        for _id in range(repeats):
            try:
                dets, imgs = next(predictor)
            except:
                pass

        time_all = (perf_counter() - t0)
        print(f"Sample size {repeats}: {time_all:.2f} s - {(repeats / time_all):.2f} fps")


    # do_predict()
    # print()
    do_predict_2()