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
        
        self.dfs = {}
        self.runtime_metrics = {}
        self.transfer_metrics = {}
        self.atts = {}
        self.total_runtime = {}
        for exp in self.experiments:
            # Check if file exists.
            if os.path.exists(self.file_paths[exp]):
                self.dfs[exp] = pd.read_csv(self.file_paths[exp], sep=", ", engine='python')
                self.runtime_metrics[exp] = self.load_runtime_metrics(self.dfs[exp])
                self.transfer_metrics[exp] = self.load_transfer_metrics(self.dfs[exp])
                self.atts[exp] =  self.load_atts(self.dfs[exp])
                self.total_runtime[exp] = self.load_total_time(self.dfs[exp])
                print(self.total_runtime[exp])

    def load_runtime_metrics(self, df):
        """
        Populates the runtime (e.g., execution time) from the csv file.
        """
        metrics = []
        for jdict in df.to_dict(orient='records'):
            # print(jdict)
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

    def load_atts(self, df):
        """
        Populates the attributes from the csv file.
        """
        return list(df["atts"].unique())[0]

    def load_total_time(self, df):
        """
        Returns the total runtime for a given experiment.
        """
        return df.loc[df['test'] == 'TotalTime'].iloc[0]['mean']

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

    def sort_by_runtime(self, exps):
        return dict(sorted(self.total_runtime.items(), key=lambda item: item[1], reverse=True))