import tensorflow as tf
import numpy as np
import json


#class Params define how to get and set parameters needed for training in a json file
class Params():
    def __init__(self, json_path):
        self.update(json_path)

    def update(self, json_path):
        with open(json_path, 'r') as json_file:
            params = json.load(json_file)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as json_file:
            json.dump(self.dict, json_file, indent=4)

    # params.dict return dictionary of the object
    @property
    def dict(self):
        return self.__dict__

    @dict.setter
    def dict(self, new_dict):
        self.dict.update(new_dict)

#






