import os
import pandas as pd

class Metrics():
    FILENAME = 'runtime_metrics.csv'

    def __init__(self, data_dir):
        """
        Initializes a Timeline object.
        """
        self.experiments = os.listdir(data_dir)
        self.file_paths = {exp: os.path.join(os.path.abspath(data_dir), f'{exp}/{self.FILENAME}') for exp in self.experiments}
        self.dfs = {exp: pd.read_csv(self.file_paths[exp], sep=", ") for exp in self.experiments}
        self.runtime_metrics = {exp: self.load_runtime_metrics(self.dfs[exp]) for exp in self.experiments}
        self.transfer_metrics = {exp: self.load_transfer_metrics(self.dfs[exp]) for exp in self.experiments}
        self.problem_size = {exp: self.load_problem_size(self.dfs[exp]) for exp in self.experiments}

    def load_runtime_metrics(self, df):
        """
        Populates the runtime (e.g., execution time) from the csv file.
        """
        metrics = []
        for jdict in df.to_dict(orient='records'):
            print(jdict)
            if jdict['units'] == 'sec':
                metrics.append(jdict)
        return metrics

    def load_transfer_metrics(self, df):
        """
        Populates the transfer metrics (e.g., throughput) from the csv file.
        """
        transfers = []
        for jdict in df.to_dict(orient='records'):
            if jdict['units'][:-2] == '/s':
                transfers.append(jdict)
        return transfers

    def load_problem_size(self, df):
        """
        Populates the attributes from the csv file.
        """
        return list(df["atts"].unique())[0]

    def get_metrics(self, exp):
        """
        Returns a timeline for a given experiment.
        """
        return self.metrics[exp] if exp in self.experiments else None    

    def get_data(self, exp):
        """
        Returns metrics for a given experiment."""
        return {
            'runtime_metrics': self.runtime_metrics[exp],
            'transfer_metrics': self.transfer_metrics[exp],
            'problem_size': self.problem_size[exp]
        }