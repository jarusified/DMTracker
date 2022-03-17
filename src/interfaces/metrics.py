import os
import pandas as pd

class Metrics():
    # TODO: Modify the filenames.
    RUNTIME_SUMMARY_FILE_NAME = 'runtime_metrics.csv'
    KERNEL_SUMMARY_FILE_NAME = 'metric_summary.csv'

    def __init__(self, data_dir):
        """
        Initializes a Metrics object.
        """
        self.experiments = os.listdir(data_dir)
        self.runtime_summary_file_paths = {exp: os.path.join(os.path.abspath(data_dir), f'{exp}/{self.RUNTIME_SUMMARY_FILE_NAME}') for exp in self.experiments}
        self.kernel_summary_file_paths = {exp: os.path.join(os.path.abspath(data_dir), f'{exp}/{self.KERNEL_SUMMARY_FILE_NAME}') for exp in self.experiments}

        # Find the kernels from the first experiment.
        self.kernels = self.get_kernels(self.experiments[0]);

        self.dfs = {}
        self.runtime_metrics = {}
        self.transfer_metrics = {}
        self.atts = {}
        self.total_runtime = {}

        # for exp in self.experiments:
        #     # Check if the runtime metric file exists.
        #     if os.path.exists(self.runtime_summary_file_paths[exp]):
        #         self.dfs[exp] = pd.read_csv(self.runtime_file_paths[exp], sep=", ", engine='python')
        #         self.runtime_metrics[exp] = self.load_runtime_metrics(self.dfs[exp])
        #         self.transfer_metrics[exp] = self.load_transfer_metrics(self.dfs[exp])
        #         self.atts[exp] =  self.load_atts(self.dfs[exp])
        #         self.total_runtime[exp] = self.load_total_time(self.dfs[exp])
    
        #     # Check if the kernel summary file exists.
        #     elif os.path.exists(self.kernel_summary_file_paths[exp]):
        #         self.dfs[exp] = pd.read_csv(self.kernel_summary_file_paths[exp], sep=", ", engine='python')
                
    # Write a function that prints 10 most expensive kernels.
    def print_expensive_kernels(self, exps):
        expensive_kernels = {}
        for exp in exps:
            expensive_kernels[exp] = self.get_expensive_kernels(exp)
        return expensive_kernels        
 
    def load_runtime_metrics(self, df):
        """
        Populates the runtime (e.g., execution time) from the csv file.
        """
        metrics = []
        for jdict in df.to_dict(orient='records'):
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
            'atts': self.atts[exp]
        }

    def get_kernels(self, exp):
        """
        Returns a list of kernels for a given ensemble.
        Note: It is assumed that the kernels are the same for all experiments.
        """
        df = pd.read_csv(self.kernel_summary_file_paths[exp], sep=",", engine='python', skiprows=6)
        return df['Kernel'].unique()

    def sort_by_runtime(self, exps):
        return dict(sorted(self.total_runtime.items(), key=lambda item: item[1], reverse=True))