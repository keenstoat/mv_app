from pathlib import Path
from nicegui import ui, events

class FilePicker(ui.dialog):

    def __init__(self, start_directory:str='.', upper_limit:str|None='.', select_dirs_only:bool=False, show_hidden_files:bool=False, include_extension:list=[]) -> None:
        
        super().__init__()

        self.open_path = Path(start_directory).resolve()
        self.upper_limit = Path(upper_limit).resolve() if upper_limit else None
        self.show_hidden_files = show_hidden_files
        self.include_extension = include_extension
        self.select_dirs_only = select_dirs_only
        self.multiple_select = False

        self.selected_path:str = None

        with self, ui.card().classes('w-[80%] h-[90%] flex flex-col justify-between'):
            # Header showing current directory path
            with ui.row().classes('w-full items-center justify-between'):
                self.header_label = ui.label(str(self.open_path))
                self.header_label.classes('truncate text-subtitle1 max-w-[80%]')
                self.button_up = ui.button(icon='arrow_upward', on_click=self.go_to_parent).props('flat dense')

            # list of files and dirs in the opened path
            grid_dict = {
                'columnDefs': [{'headerName': 'File', 'field': 'name'}],
            }
            # TODO: this is always false for now
            if self.multiple_select:
                grid_dict['rowSelection'] = {'mode': 'multiRow'}

            self.grid = ui.aggrid(grid_dict, html_columns=[0])
            self.grid.classes('w-full flex-1')
            self.grid.on('cellDoubleClicked', self.handle_double_click)
            self.grid.on('cellClicked', self.handle_click)
            
            # action buttons
            with ui.row().classes('w-full justify-between gap-2'):
                ui.button('Cancel', on_click=self.close).props('outline')
                ui.button('Ok', on_click=self.handle_ok)
        
        self.update_grid()
    
    def go_to_parent(self):
        # if at limit already, do nothing
        if self.open_path == self.upper_limit:
            return

        # if at root already, do nothing
        if self.open_path == self.open_path.parent:
            return

        self.open_path = self.open_path.parent
        self.update_grid()

    def update_grid(self) -> None:

        self.header_label.set_text(str(self.open_path))
        self.selected_path = None

        try:
            child_paths_raw = self.open_path.iterdir()
        except PermissionError:
            child_paths_raw = []

        child_paths = []
        for child_path in child_paths_raw:

            if not self.show_hidden_files and child_path.name.startswith('.'):
                continue
            if self.include_extension and child_path.is_file() and child_path.suffix not in self.include_extension:
                continue
            child_paths.append(child_path)

        child_paths.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

        self.grid.options['rowData'] = [
            {
                'name': f'📁 {path.name}' if path.is_dir() else f'📄 {path.name}',
                'path': str(path),
            }
            for path in child_paths
        ]
        
        is_root = self.open_path == self.open_path.parent
        is_upper_limit = self.open_path == self.upper_limit
        if (not self.upper_limit and not is_root) or (self.upper_limit and not is_upper_limit):
            self.grid.options['rowData'].insert(0, 
                {
                    'name': '📁 ..',
                    'path': str(self.open_path.parent),
                }
            )
            self.button_up.enable()
        else:
            self.button_up.disable()

        self.grid.update()

    async def handle_click(self, e:events.GenericEventArguments) -> None:

        self.selected_path = Path(e.args['data']['path'])

    def handle_double_click(self, e:events.GenericEventArguments) -> None:
        self.open_path = Path(e.args['data']['path'])
        if self.open_path.is_dir():
            self.update_grid()
        else:
            if not self.select_dirs_only:
                self.submit(str(self.open_path))
                self.selected_path = None

    async def handle_ok(self):
        self.submit(self.selected_path)
        self.selected_path = None
