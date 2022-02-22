import os

class Matrix():
    FILENAME = 'data_movement_matrix.json'

    def __init__(self, data_dir):
        self.experiments = os.listdir(data_dir)
        self.file_paths = [os.path.join(os.path.abspath(data_dir), f'{exp}/{self.FILENAME}') for exp in self.experiments]

    