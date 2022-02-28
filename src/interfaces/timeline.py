import os
import json
import numpy as np
from logger import get_logger
from utils.time import epoch_to_timestamp, add_duration_to_timestamp

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

    @staticmethod
    def clean_address(d):
        return {k: v for k, v in d.items() if k != 'alloc.address'}

    def load_timeline(self, file_path):
        """
        Loads a timeline from a JSON file.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        # TODO: The timestamp is not formatted correctly. Fix it. (e.g. '2020-01-01T00:00:00.000Z')
        timeline = []
        count = 10
        for idx, d in enumerate(data):
            if(count > 10):
                break
            start_epoch = d['cupti.starttime']
            end_epoch = add_duration_to_timestamp(d['cupti.starttime'], d['time.duration'])
            timeline.append({
                "idx": idx,
                "start": epoch_to_timestamp(start_epoch),
                "end": epoch_to_timestamp(end_epoch),
            })
            count += 1
            
        return timeline

    def get_timeline(self, exp):
        """
        Returns a timeline for a given experiment.
        """
        return self.timelines[exp] if exp in self.experiments else None