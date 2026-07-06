
import signal
from nicegui import Client, app, core, ui, run
from pathlib import Path
import logging, colorlog
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("error", category=UserWarning, message="The given NumPy array is not writable")


import gui

def configure_logging():
    """
    Configures the logging to log to file and to stdout. 
    Just needs to be configured once in the project, and all other places where logging is used will follow.
    """
    # Create filter and logger and handler to log to terminal with colors and part of the script path only
    class CustomFilter(logging.Filter):
        def filter(self, record):
            filepath = Path(record.pathname)
            record.custom_path = f"{filepath.parent.name}/{filepath.name}"
            return True

    yellow = '\033[33m'
    magenta = '\033[35m'
    reset = '\033[0m'
    
    console_formatter = colorlog.ColoredFormatter(
        fmt=f'[{magenta}%(asctime)s{reset}] [%(log_color)s%(levelname)s%(reset)s] [{yellow}%(custom_path)s:%(lineno)d{reset}] - %(message)s',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(CustomFilter())

    # Create formatter and logger and handler to log to log file with part of the script path only
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            filepath = Path(record.pathname)
            record.custom_path = f"{filepath.parent.name}/{filepath.name}"
            return super().format(record)
    
    file_formatter = CustomFormatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(custom_path)s:%(lineno)d] - %(message)s',
    )
    file_handler = logging.FileHandler('log.log')
    file_handler.setFormatter(file_formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

    # set the noisy loggers to warning mode
    noisy_loggers = ["watchfiles.main", "aioice"]
    for logger in noisy_loggers:
        logging.getLogger(logger).setLevel(logging.WARNING)

async def disconnect() -> None:
    """Disconnect all clients from current running server."""
    for client_id in Client.instances:
        await core.sio.disconnect(client_id)

def handle_sigint(signum, frame) -> None:
    # `disconnect` is async, so it must be called from the event loop; we use `ui.timer` to do so.
    ui.timer(0.01, disconnect, once=True)
    # Delay the default handler to allow the disconnect to complete.
    ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)

@app.on_startup
async def setup():
    if run.thread_pool:
        run.thread_pool.shutdown(wait=False, cancel_futures=True)
    run.thread_pool = ThreadPoolExecutor(max_workers=1)
    logging.info("APP SETUP - run thread pool set with 1 max workers")

@app.on_shutdown
async def cleanup() -> None:
    # This prevents ugly stack traces when auto-reloading on code change,
    # because otherwise disconnected clients try to reconnect to the newly started server.
    await disconnect()

# ==============================================================================================================

configure_logging()
signal.signal(signal.SIGINT, handle_sigint)

ui.run(
    host="0.0.0.0",
    port=8000,
    dark=True,
    show=False,
    uvicorn_reload_includes='*.js, *.py',
    storage_secret="156e3e601a30"
)