from pydantic import BaseModel
import yaml
import importlib

_CLASS_PATH_KEY = "class_path"

class SerializableModel(BaseModel):
    def to_yaml(self) -> str:
        return obj_to_yaml(self)

def obj_to_yaml(obj: BaseModel) -> str:
    data = obj.dict()
    data[_CLASS_PATH_KEY] = f"{obj.__module__}.{obj.__class__.__name__}"
    return yaml.dump(data)

def load_obj_from_yaml(yaml_str: str) -> SerializableModel:
    data = yaml.safe_load(yaml_str)
    class_path = data.pop(_CLASS_PATH_KEY)
    # Dynamically import the module and class
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    class_obj = getattr(module, class_name)
    # Instantiate the class with remaining data
    return class_obj(**data)
