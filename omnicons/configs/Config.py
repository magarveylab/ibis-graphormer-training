import json
import numpy as np

class ConfigTemplate:

    def __init__(self, base: str, properties: dict):
        self.base = base
        self.properties = properties
    
    def save_to(self, save_fp: str):
        properties = self.properties.copy()
        to_export = {'base': self.base, 'properties': properties}
        json.dump(to_export, open(save_fp, 'w'))
    
    @classmethod
    def load_from(cls, json_fp: str):
        return cls(**json.load(open(json_fp))['properties'])
    
    def get_model(self) -> None:
        # method is overwritten
        return None