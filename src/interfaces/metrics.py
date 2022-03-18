import os
from typing import OrderedDict
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
    
        # Runtime metrics
        self.runtime_summary_file_paths = {exp: os.path.join(os.path.abspath(data_dir), f'{exp}/{self.RUNTIME_SUMMARY_FILE_NAME}') for exp in self.experiments}
        self.runtime_dfs = {}
        self.runtime_metrics = {}
        self.transfer_metrics = {}
        self.atts = {}
        self.total_runtime = {}
        
        for exp in self.experiments:
            # Check if the runtime metric file exists.
            if os.path.exists(self.runtime_summary_file_paths[exp]):
                self.runtime_dfs[exp] = pd.read_csv(self.runtime_summary_file_paths[exp], sep=", ", engine='python')
                self.runtime_metrics[exp] = self.load_runtime_metrics(self.runtime_dfs[exp])
                self.transfer_metrics[exp] = self.load_transfer_metrics(self.runtime_dfs[exp])
                self.atts[exp] =  self.load_atts(self.runtime_dfs[exp])
                self.total_runtime[exp] = self.load_total_time(self.runtime_dfs[exp])

        # Kernel metrics
        self.kernel_summary_file_paths = {exp: os.path.join(os.path.abspath(data_dir), f'{exp}/{self.KERNEL_SUMMARY_FILE_NAME}') for exp in self.experiments}

        # Find the kernels from the first experiment.
        exp_0_df = pd.read_csv(self.kernel_summary_file_paths[self.experiments[0]], sep=",", engine='python', skiprows=6)
        self.kernels = exp_0_df['Kernel'].unique();
        self.metrics = exp_0_df['Metric Name'].unique();
        self.kernel_dfs = {}
        self.devices = {}
        for exp in self.experiments: 
            # Check if the kernel summary file exists.
            if os.path.exists(self.kernel_summary_file_paths[exp]):
                self.kernel_dfs[exp] = pd.read_csv(self.kernel_summary_file_paths[exp], sep=",", engine='python', skiprows=6)
                self.devices[exp] = self.kernel_dfs[exp]['Device'].unique()
                
    def get_kernel_metrics(self, metric="gst_transactions"):
        """
        Returns the metrics for a given kernel.
        Format: {
            'experiment_name': {
                'kernel_name': {
                    metric_1: val,
                    metric_2: val...
                }
            }
        }
        """
        ret = []
        for exp, k_df in self.kernel_dfs.items():
            temp = {}
            for kernel in self.kernels:
                _df = k_df.loc[k_df['Metric Name'] == metric]
                temp[kernel] = float(_df.loc[_df['Kernel'] == kernel]['Avg'].tolist()[0])
            temp["exp"] = exp
            ret.append(temp)
        return ret

        
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
        return df.loc[df['test'] == 'SGEMM-N-TotalTime'].iloc[0]['mean']

    def get_metrics(self, exp):
        """
        Returns a timeline for a given experiment.
        """
        return self.metrics[exp] if exp in self.experiments else None    

    def get_data(self):
        """
        Returns metrics for a given experiment."""
        return {
            'runtime_metrics': self.runtime_metrics,
            'transfer_metrics': self.transfer_metrics,
            'atts': self.atts,
            'kernel_metrics': self.get_kernel_metrics(),
            "kernels": self.kernels.tolist(),
            "metrics": self.metrics.tolist(),
        }

    def get_kernels(self, exp_df):
        """
        Returns a list of kernels for a given ensemble.
        Note: It is assumed that the kernels are the same for all experiments.
        """
        return exp_df['Kernel'].unique()

    def get_metrics(self, exp_df):
        """
        Returns the list of metrics recorded.
        """
        df = pd.read_csv(self.runtime_summary_file_paths[exp], sep=",", engine='python', skiprows=6)

    def sort_by_runtime(self, exps):
        return dict(sorted(self.total_runtime.items(), key=lambda item: item[1], reverse=True))