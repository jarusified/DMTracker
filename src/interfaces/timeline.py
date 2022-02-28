import os
import json
import numpy as np
from logger import get_logger

LOGGER = get_logger(__name__)

class Timeline():
    FILENAME = 'uvm-tracking.json'

    def __init__(self, data_dir):
        """
        Initializes a Timeline object.
        """
        LOGGER.info(f"{type(self).__name__} interface triggered.")
        self.experiments = os.listdir(data_dir)
        self.file_paths = {exp: os.path.join(os.path.abspath(data_dir), f'{exp}/{self.FILENAME}') for exp in self.experiments}
        self.timelines = {exp: self.load_timeline(self.file_paths[exp]) for exp in self.experiments}

    def load_timeline(self, file_path):
        """
        Loads a timeline from a JSON file.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        return data

    def get_timeline(self, exp):
        """
        Returns a timeline for a given experiment.
        """
        return self.timelines[exp] if exp in self.experiments else None