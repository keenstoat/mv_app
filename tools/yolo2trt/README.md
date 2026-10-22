


create venv with uv and system site packages access

This is requried to use torch with cuda and TensorRT
```bash
uv venv --system-site-packages
```

Create UV project without linking into the workspace of the parent project

This is requried so ultralytics dependencies do not pollute the parent project
```bash
uv init --no-workspace --bare
```

Ultralytics will try to install a TensorRT package even though it exists in the .venv (through system site packs)
```log
requirements: Ultralytics requirement ['tensorrt-cu13>=10.15,<11,!=10.2.0'] not found, attempting AutoUpdate...
```

So, the env file contains `YOLO_AUTOINSTALL=False`