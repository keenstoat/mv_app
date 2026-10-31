
from nicegui import ui

class ImageViewport(ui.element, component="ui_image_ws.js"):

    def __init__(self, src: str) -> None:
        super().__init__()

        self._props["src"] = src
    
    def start_stream(self):
        self.run_method('start_stream')
        
    def stop_stream(self):
        self.run_method('stop_stream')


