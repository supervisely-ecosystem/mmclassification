def generate_config(save_path):
    pass

def create_config(model_config_path, save_path):
    import importlib
    spec = importlib.util.spec_from_file_location("model_config", model_config_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    print(foo._base_)