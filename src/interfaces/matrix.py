import os
from generators import H2DUnifiedMemoryCommMatrixGenerator, P2PUnifiedMemoryCommMatrixGenerator, H2DCudaMemcpyCommMatrixGenerator, P2PCudaMemcpyCommMatrixGenerator

class Matrix():
    FILENAME = 'gpu_trace.csv'

    def __init__(self, data_dir):
        self.experiments = os.listdir(data_dir)
        self.file_paths = [os.path.join(os.path.abspath(data_dir), f'{exp}/{self.FILENAME}') for exp in self.experiments]
        num_devices = 4
        for file_path in self.file_paths:
            print(file_path, os.path.exists(file_path))
            if (os.path.exists(file_path)):
                h2d_um_memcpy_comm = H2DUnifiedMemoryCommMatrixGenerator(num_devices)
                h2d_um_num_bytes_comm_matrix, h2d_um_num_times_comm_matrix = h2d_um_memcpy_comm.generate_comm_matrix(file_path)
                p2p_um_memcpy_comm = P2PUnifiedMemoryCommMatrixGenerator(num_devices)
                p2p_um_num_bytes_comm_matrix, p2p_um_num_times_comm_matrix = p2p_um_memcpy_comm.generate_comm_matrix(file_path)
            
                print("Host-device communication matrix:")
                print(h2d_um_num_bytes_comm_matrix, h2d_um_num_times_comm_matrix)
                print("Peer-Peer communication matrix:")
                print(p2p_um_num_bytes_comm_matrix, p2p_um_num_times_comm_matrix)

                h2d_et_memcpy_comm = H2DCudaMemcpyCommMatrixGenerator(num_devices)
                h2d_et_num_bytes_comm_matrix, h2d_et_num_times_comm_matrix = h2d_et_memcpy_comm.generate_comm_matrix(file_path)
                p2p_et_memcpy_comm = P2PCudaMemcpyCommMatrixGenerator(num_devices)
                p2p_et_num_bytes_comm_matrix, p2p_et_num_times_comm_matrix = p2p_et_memcpy_comm.generate_comm_matrix(file_path)
    
                print("Host-device communication (explicit) matrix:")
                print(h2d_et_num_bytes_comm_matrix, h2d_et_num_times_comm_matrix)
                print("Peer-Peer communication (explicit) matrix:")
                print(p2p_et_num_bytes_comm_matrix, p2p_et_num_times_comm_matrix)

                
        # all_um_num_bytes_comm_matrix = merge_matrices(h2d_um_num_bytes_comm_matrix, p2p_um_num_bytes_comm_matrix)
        # all_um_num_times_comm_matrix = merge_matrices(h2d_um_num_times_comm_matrix, p2p_um_num_times_comm_matrix)
    
    

class Node:
    def __init__(self, name, children):
        self.name = name
        self.children = children
    
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    
    def bfs(self):
        queue = [self]
        while queue:
            node = queue.pop(0)
            print(node)
            for child in node.children:
                queue.append(child)
    
    def bellman_ford(self, source):
        """
        Bellman-Ford algorithm
        :param source:
        :return:
        """
        distances = {node: float('inf') for node in self.children}
        distances[source] = 0
        for _ in range(len(self.children) - 1):
            for node in self.children:
                for child in node.children:
                    if distances[node] + 1 < distances[child]:
                        distances[child] = distances[node] + 1
        return distances


