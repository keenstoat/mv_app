
from nicegui import ui, app
from pathlib import Path

_lib_dir = Path(__file__).parent.absolute()

app.add_static_files('/static', _lib_dir / 'static')

ui.add_head_html(f'<script src="/static/webrtc_client.js"></script>', shared=True)

class VideoWebRTC(ui.video):

    def __init__(self, webrtc_endpoint:str, **kwargs) -> None:
        
        super().__init__(src="", autoplay=True, controls=False, **kwargs)

        self._webrtc_endpoint = webrtc_endpoint
        
    def start_stream(self):
        ui.run_javascript(f"startStream('{self.html_id}', '{self._webrtc_endpoint}')")

    def stop_stream(self):
        ui.run_javascript(f"stopStream('{self.html_id}', '{self._webrtc_endpoint}')")







