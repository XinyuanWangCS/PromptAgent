import importlib

def get_task(task_name):
    module = importlib.import_module(f".{task_name}", package=__package__)
    CustomTask = getattr(module, "CustomTask")
    return CustomTask