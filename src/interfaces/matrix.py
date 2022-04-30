import os
from generators import H2DUnifiedMemoryCommMatrixGenerator, P2PUnifiedMemoryCommMatrixGenerator, H2DCudaMemcpyCommMatrixGenerator, P2PCudaMemcpyCommMatrixGenerator, ZeroCopyInfoGenerator

class Matrix():
    GPU_TRACE_FILENAME = 'nvprof_gpu_trace.csv'
    METRIC_TRACE_FILENAME = 'metric_summary.csv'

    def __init__(self, data_dir):
        self.experiments = os.listdir(data_dir)
        self.gpu_trace_paths = {exp: os.path.join(os.path.abspath(data_dir), f'{exp}/{self.GPU_TRACE_FILENAME}') for exp in self.experiments}
        self.metric_trace_paths = {exp: os.path.join(os.path.abspath(data_dir), f'{exp}/{self.METRIC_TRACE_FILENAME}') for exp in self.experiments}
        self.num_devices = 4

    def get_comm(self, exp):
        if (os.path.exists(self.gpu_trace_paths[exp])):
            h2d_um_memcpy_comm = H2DUnifiedMemoryCommMatrixGenerator(self.num_devices)
            h2d_um_num_bytes_comm_matrix, h2d_um_num_times_comm_matrix = h2d_um_memcpy_comm.generate_comm_matrix(self.gpu_trace_paths[exp])
            p2p_um_memcpy_comm = P2PUnifiedMemoryCommMatrixGenerator(self.num_devices)
            p2p_um_num_bytes_comm_matrix, p2p_um_num_times_comm_matrix = p2p_um_memcpy_comm.generate_comm_matrix(self.gpu_trace_paths[exp])
        
            print("Host-device communication matrix (unified):")
            print(h2d_um_num_bytes_comm_matrix, h2d_um_num_times_comm_matrix)
            print("Peer-Peer communication matrix (unified):")
            print(p2p_um_num_bytes_comm_matrix, p2p_um_num_times_comm_matrix)

            h2d_et_memcpy_comm = H2DCudaMemcpyCommMatrixGenerator(self.num_devices)
            h2d_et_num_bytes_comm_matrix, h2d_et_num_times_comm_matrix = h2d_et_memcpy_comm.generate_comm_matrix(self.gpu_trace_paths[exp])
            p2p_et_memcpy_comm = P2PCudaMemcpyCommMatrixGenerator(self.num_devices)
            p2p_et_num_bytes_comm_matrix, p2p_et_num_times_comm_matrix = p2p_et_memcpy_comm.generate_comm_matrix(self.gpu_trace_paths[exp])

            print("Host-device communication (explicit) matrix:")
            print(h2d_et_num_bytes_comm_matrix)
            print(h2d_et_num_times_comm_matrix)
            print("Peer-Peer communication (explicit) matrix:")
            print(p2p_et_num_bytes_comm_matrix)
            print(p2p_et_num_times_comm_matrix)

        if (os.path.exists(self.metric_trace_paths[exp])):
            zc_comm = ZeroCopyInfoGenerator(self.num_devices)
            zc_num_bytes_comm_matrix, zc_num_times_comm_matrix = zc_comm.generate_comm_matrix(filepath=self.metric_trace_paths[exp])

            print("Zero copy communication matrix:")
            print(zc_num_bytes_comm_matrix, zc_num_times_comm_matrix)

        return {
            "H2D": {
                "unified": {
                    "bytes": self.matrixToJSON(h2d_um_num_bytes_comm_matrix),
                    "times": self.matrixToJSON(h2d_um_num_times_comm_matrix),
                },
                "explicit": {
                    "bytes": self.matrixToJSON(h2d_et_num_bytes_comm_matrix),
                    "times": self.matrixToJSON(h2d_et_num_times_comm_matrix),
                }
            }, 
            "P2P": {
                "unified": {
                    "bytes": self.matrixToJSON(p2p_um_num_bytes_comm_matrix),
                    "times": self.matrixToJSON(p2p_um_num_times_comm_matrix),
                },
                "explicit": {
                    "bytes": self.matrixToJSON(p2p_et_num_bytes_comm_matrix),
                    "times": self.matrixToJSON(p2p_et_num_times_comm_matrix),
                }
            },
            "ZC": {
                "bytes": self.matrixToJSON(zc_num_bytes_comm_matrix),
                "times": self.matrixToJSON(zc_num_times_comm_matrix),
            }
        }
            
    def matrixToJSON(self, matrix):
        """
        Convert communication matrix to JSON format
        {
            "nodes": [{"name": "value", group:"value"}, ...],
            "edges": [{source: num, target: num, value: num }, ...]
        }
        """
        nodes = []
        edges = []
        for i in range(len(matrix)):
            nodes.append({"id": i, "name": str(i), "group": 0})
            for j in range(len(matrix[i])):
                edges.append({"source": i, "target": j, "value": matrix[i][j]})
        return {"nodes": nodes, "edges": edges}
    
