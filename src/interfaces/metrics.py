import os

class Metrics():
    FILENAME = 'metrics.csv'

    def __init__(self, data_dir):
        self.experiments = os.listdir(data_dir)
        self.file_paths = [os.path.join(os.path.abspath(data_dir), f'{exp}/{self.FILENAME}') for exp in self.experiments]

    