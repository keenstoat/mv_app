
import numpy as np
import torch

from nicegui import Client, app, run, ui, events
from urllib.parse import urlparse
from pathlib import Path
from os import walk as os_walk
import logging
log = logging.getLogger(__name__)

import re
from tabulate import tabulate

from .pipeline import Pipeline
from .lib.frame import Frame
from .lib.uvc_cam import get_connected_uvc_cams, get_frame_types
from .lib.utils import (
    is_url, get_icon_path, 
    get_tensorrt_version, get_cuda_version, get_deepstream_version, is_cuda_available,
    MsgType, ValidationError,
    resource_usage_generator,
)
from .lib.ui_filepicker import FilePicker
from .lib.ui_video_webrtc import VideoWebRTC
from .lib.fps_monitor import MovingMean

_app_dirpath = Path(__file__).parent.parent.parent.absolute()

class UiUx:

    blank_image = torch.zeros((720, 720*4//3, 3), dtype=torch.uint8) # HWC layout as a np array image
    BLANK_FRAME = Frame([blank_image, blank_image], -1, -1, -1)
    
    def __init__(self, client:Client):
        
        self.client = client
        self.client_is_connected = True

        # ui configuration -------------------------------------------------
        
        self._config_components = []

        # processing -----------------------------------------------------
        self.pipeline = Pipeline()
        self.perf:dict[str, MovingMean] = {}
    
        # ui components -----------------------------------------------------        
        
        self.image_display_use_mjpeg = True
        self.display_method_toggle:ui.toggle = None
        self.display_container:ui.row = None
        self.image_viewport:ui.interactive_image = None
        self.webrtc_viewport:VideoWebRTC = None
        self.ui_log_process_stats:ui.log = None
        self.active_side:int = None
        self.input_frame_source_url:dict[int, ui.input] = {}
        self.requested_frame_source_urls:dict[int, str] = {}
        self.label_playback:ui.label = None
        self.slider_playback:ui.slider = None

        self.tree_frame_source = None
        self.tree_model_files = None
        self.button_play_pause_pipeline:ui.button 
        self.button_stop_pipeline = None

        self.label_camera_fps = None 
        self.label_processing_fps = None 
        self.label_yolo_model_name = None 

        self.resource_generator = None
        self.resource_generator_timer:ui.timer = None

        self.config_tabs:ui.tabs = None
    
    def terminate_all(self):
        self.client_is_connected = False
        self.pipeline.stop()
        self.pipeline.unload_vision_model()
        if self.resource_generator_timer:
            self.resource_generator_timer.deactivate()
        if self.resource_generator:
            self.resource_generator.close()

    def add_tooltip(self, element, text):
        with element:
            ui.tooltip(text).props('delay=1000').style('white-space: pre-line')

    def notify(self, msg, msg_type:MsgType=MsgType.WARNING):
        ui.notification(msg, type=msg_type.value, timeout=1, position='top', multi_line=True)

    # Create UI elements =========================================================
    # ============================================================================

    # Image playback elements  --------------------------------------------------------
    def set_active_side(self, active_side:int|None):
        
        if active_side == self.active_side:
            return

        glow_classes = 'rounded-md shadow-[0_0_5px_3px_#005F92FF]'
        for side, item in self.input_frame_source_url.items():
            if side == active_side:
                item.parent_slot.parent.classes(add=glow_classes)
            else:
                item.parent_slot.parent.classes(remove=glow_classes)

        self.active_side = active_side
        self.tree_frame_source.deselect()

    def create_ui_inputs_frame_source_url(self):
        
        def set_input_frame_source_url(side):

            self.input_frame_source_url[side].set_value(None)
            del self.requested_frame_source_urls[side]

        for side in (0, 1):
            # only the creation of the component must be inside a container

            with ui.column().classes('w-1/2 h-full p-0 gap-1 overflow-hidden'):
                side_name = "left" if side == 0 else "right"
                input_source_url = ui.input(placeholder=f'{side_name} frame source')
            
            input_source_url.props('outlined dense readonly')
            input_source_url.classes('w-full')
            input_source_url.on("click", lambda _, s=side: self.set_active_side(s))

            with input_source_url:
                clear_button = ui.button(icon="clear").props('flat color=white')
                clear_button.on_click(lambda _, s=side: set_input_frame_source_url(s))
            
            self._config_components.append(input_source_url)
            self.input_frame_source_url[side] = input_source_url

    def create_ui_webrtc_viewport(self, display_container:ui.row):

        self.display_container = display_container
        video = VideoWebRTC(f"/webrtc-stream/{self.client.id}")
        video.classes('w-full h-full object-contain')
        #  Use this class and style instead for the image to its actual size
        # without expanding over
        # video.classes(
        #     'max-w-full max-h-full'
        # ).style(
        #     'width: auto; height: auto; '
        #     'object-fit: contain;'
        # )
        video.classes(remove="pointer-events-none")
        
        self.webrtc_viewport = video
        self.display_method_toggle.set_value('webrtc')

    def create_ui_image_viewport(self, display_container:ui.row):
        
        self.display_container = display_container
        image = ui.interactive_image()
        image.classes('w-full h-full [&>img]:object-contain [&>img]:max-w-full [&>img]:max-h-full')
        #  Use this class and style instead for the image to its actual size
        # without expanding over
        # video.classes(
        #     'max-w-full max-h-full'
        # ).style(
        #     'width: auto; height: auto; '
        #     'object-fit: contain;'
        # )
        image.classes(remove="pointer-events-none")

        image.set_source(f'data:image/jpg;base64,{self.BLANK_FRAME.image_to_base64()}')

        self.image_viewport = image
        self.display_method_toggle.set_value('mjpeg' if self.image_display_use_mjpeg else 'jpeg')

    def create_playback_controls(self):

        with ui.row().classes("w-full"):
            label = ui.label("--/--")
            slider = ui.slider(min=0, max=1, step=1, value=0)
            slider.classes('w-full flex-1 px-4')
            slider.set_enabled(False)

        self.label_playback = label
        self.slider_playback = slider

    # Frame source tree inputs and controls -------------------------------------------

    def create_ui_video_source_tree(self):

        def on_select_tree_item(event:events.ValueChangeEventArguments):
            
            clicked_node_key = str(event.value)
            if not clicked_node_key: return

            # clicked on a video source node
            video_source_scheme = urlparse(clicked_node_key).scheme
            if video_source_scheme in ("rtsp", "https", "file", "uvc"):

                if self.active_side is None:
                    self.notify("Must select a frame to assign the frame source")
                    self.tree_frame_source.deselect()
                    return

                self.requested_frame_source_urls[self.active_side] = clicked_node_key
                
                if video_source_scheme in ["rtsp", "https"]:
                    self.input_frame_source_url[self.active_side].set_value(clicked_node_key)
                
                elif video_source_scheme == "file":
                    self.input_frame_source_url[self.active_side].set_value(Path(clicked_node_key).name)

                elif video_source_scheme == "uvc": # USB video class
                    self.input_frame_source_url[self.active_side].set_value(clicked_node_key.removeprefix("uvc://"))
                
                else:
                    log.warning(f"clicked node '{clicked_node_key}' is a video source but not recognized!")
            
            # clicked on a directory node
            else:
                # get the tree node that was clicked
                clicked_path = Path(clicked_node_key)
                segments = clicked_path.relative_to(videos_root_dirpath)
                selected_node = video_file_nodes[0]
                for part in segments.parts:
                    child_nodes = {Path(child_node['uri']).name: child_node for child_node in selected_node['children']}
                    selected_node = child_nodes[part]
                
                # then expand or collapse the node
                if selected_node['is_expanded']:
                    self.tree_frame_source.collapse([clicked_node_key])
                    selected_node['is_expanded'] = False
                else:
                    self.tree_frame_source.expand([clicked_node_key])
                    selected_node['is_expanded'] = True

                self.tree_frame_source.deselect()

        async def set_videos_root_dir():
            nonlocal videos_root_dirpath
            videos_root_dirpath = await file_picker
            if videos_root_dirpath:
                app.storage.user['videos_root_dirpath'] = str(videos_root_dirpath)
            await refresh_tree()

        async def add_new_network_stream_node():
            
            new_url = await dialog_add_new_url
            if new_url is None:
                return 
            if not is_url(new_url):
                self.notify(f"'{new_url}' is not a valid URL")
                return
            new_node = {'uri': new_url, 'name': new_url, 'icon': 'camera'}
            network_stream_nodes.append(new_node)
            await refresh_tree()

        def create_video_file_tree_nodes(root_dir:Path):
                        
            if not root_dir or not root_dir.exists():
                return []
                
            root, sub_dirs, files = next(os_walk(root_dir))
            sub_dirs[:] = [item for item in sub_dirs if not item.startswith(".")]
            files[:] = [item for item in files if not item.startswith(".")]
            root = Path(root)

            files.sort()
            file_nodes = [
                {'uri': f"file://{root / f}", 'name': f, 'icon': 'video_file'} 
                    for f in files if f.endswith(".mp4")
            ]

            sub_dir_nodes = []
            sub_dirs.sort()
            for sub_dir in sub_dirs:
                sub_dir_nodes.extend(create_video_file_tree_nodes(root / sub_dir))

            return [{
                'uri': f"{root}", 'name': root.name, 'is_expanded': False, 'icon': 'folder', 
                'children': sub_dir_nodes + file_nodes
            }]
        
        def create_webcam_nodes():
            webcams = get_connected_uvc_cams()
            return [
                {'uri': f"uvc://{device}", 'name': f"{name} ({device})", 'icon': 'usb'} 
                for device, name in webcams.items()
            ]

        def create_tree():
            nonlocal video_file_nodes
            video_file_nodes = create_video_file_tree_nodes(videos_root_dirpath)
            
            tree_nodes = []
            tree_nodes += create_webcam_nodes()
            tree_nodes += network_stream_nodes
            tree_nodes += video_file_nodes

            tree = ui.tree(
                nodes=tree_nodes, 
                node_key='uri', label_key='name', 
                on_select=on_select_tree_item,
            )
            tree.props('no-selection-unset no-connectors selected-color="primary"')
            tree.expand()
            self.tree_frame_source = tree

        async def refresh_tree():
            tree_container.clear()
            with tree_container:
                create_tree()
        
        # dialog to ask user for new video source URL
        with ui.dialog() as dialog_add_new_url, ui.column().classes('items-center'):
            with ui.row():
                user_input = ui.input(placeholder="input video source URL")
                user_input.props('outlined dense')
            with ui.row():
                ui.button('add', on_click=lambda: dialog_add_new_url.submit(user_input.value))
                ui.button('cancel', on_click=lambda: dialog_add_new_url.submit(None))

        with ui.row().classes("w-full p-0 gap-1"):
            # button to set the videos root directory
            button_select_videos_dir = ui.button("Select Videos Directory").props('rounded size=sm')
            button_select_videos_dir.on_click(set_videos_root_dir)
            self.add_tooltip(button_select_videos_dir,"Select root dir for videos")

            # button to add new RTSP URL
            button_new_source = ui.button("add stream url", on_click=add_new_network_stream_node).props('rounded size=sm')
            self.add_tooltip(button_new_source,"Add a new video source URL")

            # button to refresh the view
            button_refresh = ui.button("refresh", on_click=refresh_tree).props('rounded size=sm')
            self.add_tooltip(button_refresh,"Refresh the view")
        
        file_picker = FilePicker(select_dirs_only=True)
        network_stream_nodes = [
            {'uri': "rtsp://127.0.0.1:8554/stream", 'name': 'Dummy RTSP stream', 'icon': 'camera'},
        ]
        videos_root_dirpath:Path = app.storage.user.get('videos_root_dirpath')
        videos_root_dirpath = Path(videos_root_dirpath) if videos_root_dirpath else None
        video_file_nodes = []
        with ui.scroll_area().classes("w-full flex-1 p-0") as tree_container:
            create_tree()

    # Vision models source tree inputs and controls -------------------------------------------

    def create_ui_vision_models_tree(self):

        def on_select_tree_item(event:events.ValueChangeEventArguments):
            nonlocal selected_model_filepath

            clicked_path = str(event.value)
            if not clicked_path: return

            clicked_path = Path(clicked_path)
            if clicked_path.is_file():
                selected_model_filepath = clicked_path
                current_model_filepath = \
                    self.pipeline.vision_model.engine_filepath if self.pipeline.vision_model else ""
                if Path(current_model_filepath) != selected_model_filepath:
                    button_load_model.props('color="warning"')
                else:
                    button_load_model.props('color="primary"')
            
            else:
                segments = clicked_path.relative_to(vision_models_dirpath)
                selected_node = model_file_nodes[0]
                for part in segments.parts:
                    child_nodes = {Path(child_node['uri']).name: child_node for child_node in selected_node['children']}
                    selected_node = child_nodes[part]
                
                if selected_node['is_expanded']:
                    self.tree_model_files.collapse([str(clicked_path)])
                    selected_node['is_expanded'] = False
                else:
                    self.tree_model_files.expand([str(clicked_path)])
                    selected_node['is_expanded'] = True

                self.tree_model_files.deselect()

        def create_tree_nodes(root_dir:Path):
        
            if not root_dir or not root_dir.exists():
                return []
                
            root, sub_dirs, files = next(os_walk(root_dir))
            sub_dirs[:] = [item for item in sub_dirs if not item.startswith(".")]
            files[:] = [item for item in files if not item.startswith(".")]
            root = Path(root)

            files.sort()
            file_nodes = [
                {'uri': f"{root / f}", 'name': f, 'icon': 'video_file'} 
                    for f in files if any(f.endswith(ext) for ext in [".engine"])
            ]

            sub_dir_nodes = []
            sub_dirs.sort()
            for sub_dir in sub_dirs:
                sub_dir_nodes.extend(create_tree_nodes(root / sub_dir))

            return [{
                'uri': f"{root}", 'name': root.name, 'is_expanded': False, 'icon': 'folder', 
                'children': sub_dir_nodes + file_nodes
            }]

        async def set_models_root_dir():
            nonlocal vision_models_dirpath
            vision_models_dirpath = await file_picker
            if vision_models_dirpath:
                app.storage.user['vision_models_dirpath'] = str(vision_models_dirpath)
            await refresh_tree()

        def create_tree():
            nonlocal model_file_nodes
            model_file_nodes = create_tree_nodes(vision_models_dirpath)
            tree = ui.tree(
                nodes=model_file_nodes, 
                node_key='uri', label_key='name', 
                on_select=on_select_tree_item,
            )
            tree.props('no-selection-unset no-connectors selected-color="primary"')
            tree.expand()
            self.tree_model_files = tree

        async def refresh_tree():
            tree_container.clear()
            with tree_container:
                create_tree()
        
        async def unload_model():
            try:
                await run.io_bound(self.pipeline.unload_vision_model)
                self.label_yolo_model_name.set_text("--")
                self.label_yolo_model_name.classes(remove="text-[#00FF00] text-bold")
                self.notify(f"Model unloaded", MsgType.SUCCESS)
                button_load_model.props('color="warning"')
            except Exception as e:
                self.notify(f"Failed to unload model", MsgType.ERROR)
                log.error(e)

        async def load_model():

            try:
                if not selected_model_filepath:
                    raise ValidationError(f"Must select a model file to load", MsgType.WARNING)
                
                # always unload the model first- if the loading process fails (model not found, etc.) the UI does not 
                # return to the original config, so always unloading the model forces the user to pay attention 
                await run.io_bound(self.pipeline.unload_vision_model)
                self.label_yolo_model_name.set_text("--")
                self.label_yolo_model_name.classes(remove="text-[#00FF00] text-bold")
                button_load_model.props('color="warning"')

                loading_model_dialog.open()
                self.enable_ui(False)
                self.button_play_pause_pipeline.set_enabled(False)
                await run.io_bound(self.pipeline.load_vision_model, selected_model_filepath)
                # update the text with the model name from the model itself, so we know it was loaded
                self.label_yolo_model_name.set_text(Path(self.pipeline.vision_model.engine_filepath).name)
                self.label_yolo_model_name.classes(add="text-[#00FF00] text-bold")
                self.notify(f"Model '{selected_model_filepath.name}' loaded", MsgType.SUCCESS)
                button_load_model.props('color="primary"')
            
            except ValidationError as ve:
                self.notify(ve.message, ve.message_type)

            except Exception as e:
                self.notify(f"Model could not be loaded. Check logs for more info.", MsgType.ERROR)
                log.error(e)

            finally:
                self.enable_ui(True)
                self.button_play_pause_pipeline.set_enabled(True)
                loading_model_dialog.close()
            
        with ui.dialog() as loading_model_dialog, ui.row().classes('items-center'):
            ui.spinner("orbit", size="lg")
            ui.label("loading model...")

        with ui.row().classes("w-full p-0 gap-1"):

            # button to set the videos root directory
            button_select_models_dir = ui.button("Select Models Directory").props('rounded size=sm')
            button_select_models_dir.on_click(set_models_root_dir)
            self.add_tooltip(button_select_models_dir,"Select root dir for models")

            # button to refresh the view
            button_refresh = ui.button("refresh", on_click=refresh_tree).props('rounded size=sm')
            self.add_tooltip(button_refresh,"Refresh the view")

            # button to load selected model
            button_load_model = ui.button("Load Model", on_click=load_model).props('rounded size=sm')
            button_load_model.props('color="warning"')
            self.add_tooltip(button_load_model, "Select the vision model to use")

            # button to unload all models
            button_unload_model = ui.button("Unload Model", on_click=unload_model).props('rounded size=sm')
            self.add_tooltip(button_unload_model, "Unloads a model from CUDA and frees memory")
        
        self._config_components.append(button_load_model)
        self._config_components.append(button_unload_model)

        file_picker = FilePicker(select_dirs_only=True, include_extension=['.engine', '.pt'])
        selected_model_filepath:Path = None
        model_file_nodes = []
        vision_models_dirpath:Path = app.storage.user.get('vision_models_dirpath')
        vision_models_dirpath = Path(vision_models_dirpath) if vision_models_dirpath else None

        with ui.scroll_area().classes("w-full flex-1 p-0") as tree_container:
            create_tree()

    # Settings and configuration inputs and controls ----------------------------------
    
    def create_ui_config_panel(self):

            with ui.expansion('Image capture', value=True).classes('w-full p-0') as exp:
                exp.props('header-class="text-bold bg-[#333]"')
                self.create_ui_select_frame_type()
                self.create_ui_toggle_frame_grab_tool()
                self.create_ui_slider_grab_buffer_size()

            with ui.expansion('Inference', value=True).classes('w-full p-0') as exp:
                exp.props('header-class="text-bold bg-[#333]"')
                
                self.create_ui_toggle_inference_method()

            with ui.expansion('Visualization', value=True).classes('w-full p-0') as exp:
                exp.props('header-class="text-bold bg-[#333]"')
                
                self.create_ui_toggle_visualization_method()

    # Runtime panel view  ----------------------------------

    def create_ui_runtime_panel(self):

        with ui.column().classes("w-full h-full"):
            with ui.row().classes("w-full justify-around"):

                with ui.column().classes("items-center gap-0 p-0"):
                    knob_cpu = ui.knob(0.0, max=100, track_color='grey-2', show_value=True)
                    ui.label("CPU %")

                with ui.column().classes("items-center gap-0 p-0"):
                    knob_ram = ui.knob(0.0, max=100, track_color='grey-2', show_value=True)
                    ui.label("RAM %")

                with ui.column().classes("items-center gap-0 p-0"):
                    knob_gpu = ui.knob(0.0, max=100, track_color='grey-2', show_value=True)
                    ui.label("GPU %")

                with ui.column().classes("items-center gap-0 p-0"):
                    knob_vdd = ui.knob(0.0, max=25000, track_color='grey-2', show_value=True)
                    knob_vdd.props('color="fuchsia-600"').update()
                    ui.label("VDD mW")
            
            log = ui.log(max_lines=50).classes('w-full h-full')
            self.ui_log_process_stats = log

        def get_color(value):
            return "red-14" if value > 90 else ("yellow-9" if value > 75 else "light-green-14")
            
        self.resource_generator = resource_usage_generator(interval_ms=50)
        gpu_moving_mean = MovingMean(window_size=8) # timer goes off 4 times / sec, so take 2 second window
        def update_knobs():
            cpu_percent, ram_percent, gpu_percent, vdd_value = next(self.resource_generator)
            gpu_percent = gpu_moving_mean.update(gpu_percent)

            knob_cpu.set_value(round(cpu_percent, 1)).props(f"color={get_color(cpu_percent)}").update()
            knob_ram.set_value(round(ram_percent, 1)).props(f"color={get_color(ram_percent)}").update()
            knob_gpu.set_value(round(gpu_percent, 1)).props(f"color={get_color(gpu_percent)}").update()
            knob_vdd.set_value(int(vdd_value))
            
        self.resource_generator_timer = ui.timer(interval=0.25, active=True, callback=update_knobs)
    
    def create_ui_select_frame_type(self):

        def update_value():

            match = re.match(r'(\d+)x(\d+) @ (\d+) fps', select.value)
            if match is None or len(match.groups()) != 3:
                self.notify("Selected stream not matching")
                return

            self.pipeline.conf_cam_frame_width = int(match.group(1))
            self.pipeline.conf_cam_frame_height = int(match.group(2))
            self.pipeline.conf_cam_fps = int(match.group(3))

        cams = get_connected_uvc_cams()
        select_frame_types = ["n/a"]
        if cams:
            device_path = next(iter(cams.keys()))
            frame_types = get_frame_types(device_path)
            select_frame_types = [f"{ft['width']}x{ft['height']} @ {ft['fps']} fps" for ft in frame_types]
        
        with ui.row().classes("w-full p-0 gap-2 items-center"):
            ui.label("Video stream")
            select = ui.select(select_frame_types)
            h = self.pipeline.conf_cam_frame_height
            w = self.pipeline.conf_cam_frame_width
            fps = self.pipeline.conf_cam_fps
            select.set_value(f"{w}x{h} @ {fps} fps")

        select.on_value_change(update_value)
        self._config_components.append(select)

    def create_ui_slider_grab_buffer_size(self):

        def update_value():
            self.pipeline.conf_grab_buffer_size = int(slider.value)
            l_value.set_text(f"{self.pipeline.conf_grab_buffer_size}")

        with ui.row().classes("w-full p-0 gap-0"):
            with ui.row().classes("w-full p-0 gap-2"):
                ui.label("Grab Buffer")
                l_value = ui.label(f"{self.pipeline.conf_grab_buffer_size}")
            slider = ui.slider(
                min=1, max=10, step=1, 
                value=self.pipeline.conf_grab_buffer_size,
                on_change=update_value
            )
            slider.props('snap')

        self._config_components.append(slider)

    def create_ui_toggle_frame_grab_tool(self):

        def update_value():
            self.pipeline.conf_grab_tool = toggle.value

        with ui.row().classes("w-full p-0 gap-2 items-center"):
            ui.label("Camera grab tool")
            toggle = ui.toggle(['gst', 'cv2'])
        toggle.set_value(self.pipeline.conf_grab_tool)
        toggle.on_value_change(update_value)
        self._config_components.append(toggle)

    def create_ui_toggle_inference_method(self):

        def update_value():
            self.pipeline.conf_inference_method = toggle.value

        with ui.row().classes("w-full p-0 gap-2 items-center"):
            ui.label("Inference")
            toggle = ui.toggle(['none', 'simple', 'optimized'])

        toggle.set_value(self.pipeline.conf_inference_method)
        toggle.on_value_change(update_value)
        self._config_components.append(toggle)
    
    def create_ui_toggle_visualization_method(self):

        def update_value():

            self.image_display_use_mjpeg = toggle.value == 'mjpeg'

            if toggle.value == 'webrtc' and self.webrtc_viewport is None:
                log.info("Creating webrtc view port")
                self.display_container.clear()
                self.image_viewport = None
                with self.display_container:
                    self.create_ui_webrtc_viewport(self.display_container)
                return

            if toggle.value in ['jpeg', 'mjpeg'] and self.image_viewport is None:
                log.info("Creating image view port")
                self.display_container.clear()
                self.webrtc_viewport = None
                with self.display_container:
                    self.create_ui_image_viewport(self.display_container)
                return

        ui.label("Visualization")
        toggle = ui.toggle(['jpeg', 'mjpeg', 'webrtc'])
        # no value is set because this component is created before the viewports are created.
        # when the viewports are created, they will change the value and trigger the handler
        toggle.on_value_change(update_value)
        self._config_components.append(toggle)
        self.display_method_toggle = toggle
    
    # Running pipeline controls -------------------------------------------------

    def create_ui_buttons_run_pipeline(self):

        self.button_play_pause_pipeline = ui.button(on_click=self.run_pipeline)
        self.update_ui_play_pause_pipeline_button("start")

        self.button_stop_pipeline = ui.button(on_click=self.run_pipeline)
        self.button_stop_pipeline.set_enabled(False)
        with self.button_stop_pipeline, ui.row().classes('items-center'):
            ui.label("stop")
            ui.image(get_icon_path("stop")).classes('w-8 h-8')
    
    def create_ui_button_take_snapshot(self):

        def take_snapshot():
            frame = self.get_display_frame()
            snaps_dir = _app_dirpath / "_snapshots"
            frame.save_image(snaps_dir)
            self.notify("Snapshot saved", MsgType.SUCCESS)

        with ui.button(on_click=take_snapshot), ui.row().classes('items-center'):
            ui.label("snap")
            ui.image(get_icon_path("snapshot")).classes('w-8 h-8')

    # Status bar  -----------------------------------------------------------------

    def create_ui_status_bar(self):
        
        ui.label("Model:")
        self.label_yolo_model_name = ui.label("--")
        
        ui.space()

        cuda_version = get_cuda_version() if is_cuda_available() else "n/a"
        ui.label("CUDA:")
        ui.label(cuda_version).classes("text-info text-bold")
        
        ui.label("|")
        trt_version = get_tensorrt_version() if is_cuda_available() else "n/a"
        ui.label("TensorRT:")
        ui.label(trt_version).classes("text-info text-bold")

        ui.label("|")
        ds_version = get_deepstream_version() if is_cuda_available() else "n/a"
        ui.label("DeepStream:")
        ui.label(ds_version).classes("text-info text-bold")

    # UI update methods =========================================================
    # ===========================================================================
    
    def get_display_frame(self) -> Frame:
        
        if self.pipeline.display_frame is None:
            return self.BLANK_FRAME
        
        return self.pipeline.display_frame

    def update_ui_image_display(self):
        
        if self.image_viewport is None:
            return
        frame = self.get_display_frame()

        if frame.frame_count == -1:
            self.image_viewport.set_source("")
            return 

        if self.image_display_use_mjpeg:
            if not self.image_viewport.source.startswith('/mjpeg-stream'):
                self.image_viewport.set_source(f'/mjpeg-stream/{self.client.id}?{frame.frame_count}')
        else:
            self.image_viewport.set_source(f'/displayframe/{self.client.id}?{frame.frame_count}')

    def update_ui_playback_monitor(self):

        slider_playback = self.slider_playback
        if slider_playback is None:
            return
        
        frame = self.get_display_frame()
        label_playback = self.label_playback
        
        # frame is the blank frame
        if frame.frame_count == -1:
            slider_playback.set_value(0)
            label_playback.set_text("--/--")
            return
        
        # frame is produces by a camera or a network stream
        if frame.total_frames == np.inf:
            slider_playback.set_value(1)
            slider_playback._props["max"] = 1
            label_playback.set_text(f"{frame.frame_count}/inf")
            return

        label_playback.set_text(f"{frame.frame_count}/{frame.total_frames}")
        slider_playback._props["max"] = frame.total_frames
        slider_playback.set_value(frame.frame_count)
        slider_playback.update()

    def update_ui_log_process_stats(self):

        log_stats = self.ui_log_process_stats
        if log_stats is None:
            return
        
        log_stats.clear()

        frame = self.get_display_frame()
        if frame.frame_count == -1:
            return

        classes = 'text-m'
        
        height, width = frame.height, frame.width
        text = f"shape HxW   : {height}x{width} (full: {height}x{width * frame.slices})"
        log_stats.push(text, classes=classes)

        process_fps = self.pipeline.fps_monitor.fps
        target_fps = self.pipeline.fps_monitor.target_fps or -1.0
        text = f"process fps : {process_fps:.2f} (tops: {target_fps:.2f})"
        log_stats.push(text, classes=classes)

        
        log_stats.push("\nSpeeds:", classes=classes)
        table = []
        for key, value in frame.perf.items():
            if key not in self.perf:
                self.perf[key] = MovingMean(window_size=int(frame.fps))
            value = self.perf[key].update(value)
            table.append([key, f"{value:.2f} ms"])

        text = tabulate(table, (), tablefmt="plain")
        log_stats.push(text, classes=classes)

    def enable_ui(self, enable=True):
        """
        Enables or disables the GUI components so that the user cannot accidentally change the test parameters
        while the test is taking place.
        """
        for element in self._config_components:
            element.set_enabled(enable)
        
        if enable:
            self.tree_frame_source.classes(remove='pointer-events-none', add='opacity-100')
            self.tree_model_files.classes(remove='pointer-events-none', add='opacity-100')
        else:
            self.tree_frame_source.classes(add='pointer-events-none opacity-50')
            self.tree_model_files.classes(add='pointer-events-none opacity-50')

    def update_ui_all(self):

        if not self.client_is_connected:
            return
        
        self.update_ui_image_display()
        self.update_ui_playback_monitor()
        self.update_ui_log_process_stats()

    # Processing methods =========================================================
    # ============================================================================

    def update_ui_play_pause_pipeline_button(self, play_button_text:str):

        self.button_play_pause_pipeline.clear()
        with self.button_play_pause_pipeline, ui.row().classes('items-center'):
            ui.label(play_button_text)
            ui.image(get_icon_path(play_button_text)).classes('w-8 h-8')

    async def run_pipeline(self, events):

        if self.pipeline.is_stopped():

            if events.sender == self.button_stop_pipeline:
                return
            
            self.set_active_side(None)
            self.update_ui_play_pause_pipeline_button("pause")
            self.button_stop_pipeline.set_enabled(True)
            self.enable_ui(False)

            self.pipeline.conf_frame_source_urls = list(dict(sorted(self.requested_frame_source_urls.items())).values())

            self.perf = {}

            if self.webrtc_viewport:
                self.webrtc_viewport.start_stream()

            current_tab = self.config_tabs.value
            self.config_tabs.set_value("Runtime")
            result = await run.io_bound(self.pipeline.process, self.update_ui_all)
            if self.config_tabs.value == "Runtime":
                self.config_tabs.set_value(current_tab)

            if self.webrtc_viewport: 
                self.webrtc_viewport.stop_stream()

            self.update_ui_all()

            if result is not None:
                msg, msg_type = result
                self.notify(msg, msg_type)
            
            self.update_ui_play_pause_pipeline_button("start")
            self.button_play_pause_pipeline.set_enabled(True) # enable back the button once process actually stopped
            self.button_stop_pipeline.set_enabled(False)
            self.enable_ui(True) 

        elif self.pipeline.is_processing():
            # once the pipeline stops, await run.io_bound(self.pipeline.process...) will return
            # and after updating ui and all other visual data the toggle button is enabled back!

            if events.sender == self.button_stop_pipeline:
                # disable the buttons to prevent double clicks
                self.button_play_pause_pipeline.set_enabled(False) 
                self.button_stop_pipeline.set_enabled(False)
                self.pipeline.stop()

            elif events.sender == self.button_play_pause_pipeline:
                self.pipeline.pause()
                self.update_ui_play_pause_pipeline_button("resume")
                self.button_stop_pipeline.set_enabled(True)

        elif self.pipeline.is_paused():

            if events.sender == self.button_stop_pipeline:
                # disable the buttons to prevent double clicks
                self.button_play_pause_pipeline.set_enabled(False) 
                self.button_stop_pipeline.set_enabled(False)
                self.pipeline.stop()

            elif events.sender == self.button_play_pause_pipeline:
                self.pipeline.resume()
                self.update_ui_play_pause_pipeline_button("pause")
                self.button_stop_pipeline.set_enabled(True)


