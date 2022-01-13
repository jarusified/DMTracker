import os
import subprocess

from logger import get_logger
from generators import *
from utils import create_dir_after_check

LOGGER = get_logger(__name__)

NCCL_SHARED_LIBRARY_PATH = "./nccl/build/lib/libnccl.so"

class Tracer:
    """
    Tracer Class.
    """

    def __init__(self, args):
        LOGGER.info(f"{type(self).__name__} triggered.")

        self.cmd = args.args["cmd"]
        self.app_name = args.args["app_name"]
        self.num_gpus = args.args["num_gpus"]
        self.output_dir = os.path.join(os.path.abspath(args.args["output_dir"]), self.app_name)
        

        # Run app with nvprof GPU Trace
        self.gpu_trace_file = os.path.join(self.output_dir, "gpu_trace.csv")
        
        # Run app with Metric Trace
        # self.metrics = ["nvlink_user_data_received", "nvlink_user_data_transmitted", "sysmem_read_bytes", "sysmem_write_bytes"]
        self.metrics = ["crop__busy_cycles_avg"]
        self.metric_trace_file = os.path.join(self.output_dir, "metric_trace.csv")
        
        self.runtime_metrics_file = os.path.join(self.output_dir, "runtime_metrics.csv")
        self.nsys_trace_qdrep_file = os.path.join(self.output_dir, "nsys_trace.qdrep")
        self.nsys_trace_json_file = os.path.join(self.output_dir, "nsys_trace.json")

        create_dir_after_check(self.output_dir)
        
        # # Run app with NCCL library
        # if(with_nccl): 
        #     check_nccl(NCCL_SHARED_LIBRARY_PATH)
        #     preload = f'LD_PRELOAD={NCCL_SHARED_LIBRARY_PATH}'

        #     file_regex = f'nccl_{args.coll_type}_*.csv'
        #     file_paths = glob.glob(file_regex)
        #     remove_existing_files(file_paths)
        #     nccl_cmd = f'{preload} {args.ifile}'
        #     subprocess.run([nccl_cmd], shell=True)

    def start(self):
        """
        TODO: Run all the commands in parallel.
        """
        gpu_trace_cmd = f'nvprof --print-gpu-trace --csv --log-file {self.gpu_trace_file} {self.cmd}'
        subprocess.run([gpu_trace_cmd], shell=True)

        metric_trace_cmd = f'nvprof --print-gpu-trace --metrics {",".join(self.metrics)} --csv --log-file {self.metric_trace_file} {self.cmd}'
        subprocess.run([metric_trace_cmd], shell=True)

        runtime_metrics_cmd = f'{self.cmd} -m  {self.runtime_metrics_file}'
        subprocess.run([runtime_metrics_cmd], shell=True)

        nsys_metrics_cmd = f'nsys profile --trace=cuda,nvtx -d 20 --sample=none -o {self.nsys_trace_qdrep_file} {self.cmd}'
        subprocess.run([nsys_metrics_cmd], shell=True)
        
        nsys_convert_to_json_cmd = f'nsys export {self.nsys_trace_qdrep_file} -o {self.nsys_trace_json_file} -t json'
        subprocess.run([nsys_convert_to_json_cmd], shell=True)

    @staticmethod
    def merge_matrices(matrix1, matrix2):
        """
        Merge the matrix. Need to redo this.
        """
        for x in range(0, len(matrix1)):
            for y in range(0, len(matrix2)):
                matrix1[x + 1][y + 1] = matrix2[x][y]

        return matrix1

    def get_unified_memory_transfers(self):
        # Unified Memory
        h2d_um_memcpy_comm = H2DUnifiedMemoryCommMatrixGenerator(self.num_gpus)
        h2d_um_num_bytes_comm_matrix, h2d_um_num_times_comm_matrix = h2d_um_memcpy_comm.generate_comm_matrix(self.gpu_trace_file)
        p2p_um_memcpy_comm = P2PUnifiedMemoryCommMatrixGenerator(self.num_gpus)
        p2p_um_num_bytes_comm_matrix, p2p_um_num_times_comm_matrix = p2p_um_memcpy_comm.generate_comm_matrix(self.gpu_trace_file)

        all_um_num_bytes_comm_matrix = self.merge_matrices(h2d_um_num_bytes_comm_matrix, p2p_um_num_bytes_comm_matrix)
        all_um_num_times_comm_matrix = self.merge_matrices(h2d_um_num_times_comm_matrix, p2p_um_num_times_comm_matrix)

        if max(map(max, all_um_num_bytes_comm_matrix)) != 0 and max(map(max, all_um_num_times_comm_matrix)) !=0:
            print("Unified Memory Bytes: \n", all_um_num_bytes_comm_matrix)
            print("Unified Memory Transfers: \n", all_um_num_times_comm_matrix)

            outputfile_um_num_bytes_comm_matrix = "um_num_bytes_comm_matrix"
            outputfile_um_num_times_comm_matrix = "um_num_times_comm_matrix"

            # plot_comm_matrix(all_um_num_bytes_comm_matrix, self.num_gpus, outputfile_um_num_bytes_comm_matrix, self.scale)
            # plot_comm_matrix(all_um_num_times_comm_matrix, self.num_gpus, outputfile_um_num_times_comm_matrix, self.scale)

    def get_explicit_memory_transfers(self):
        # Explicit Transfers
        h2d_et_memcpy_comm = H2DCudaMemcpyCommMatrixGenerator(self.num_gpus)
        h2d_et_num_bytes_comm_matrix, h2d_et_num_times_comm_matrix = h2d_et_memcpy_comm.generate_comm_matrix(self.gpu_trace_file)
        p2p_et_memcpy_comm = P2PCudaMemcpyCommMatrixGenerator(self.num_gpus)
        p2p_et_num_bytes_comm_matrix, p2p_et_num_times_comm_matrix = p2p_et_memcpy_comm.generate_comm_matrix(self.gpu_trace_file)

        all_et_num_bytes_comm_matrix = self.merge_matrices(h2d_et_num_bytes_comm_matrix, p2p_et_num_bytes_comm_matrix)
        all_et_num_times_comm_matrix = self.merge_matrices(h2d_et_num_times_comm_matrix, p2p_et_num_times_comm_matrix)

        if max(map(max, all_et_num_bytes_comm_matrix)) != 0 and max(map(max, all_et_num_times_comm_matrix)) !=0:
            print("Explicit Transfers Bytes: \n", all_et_num_bytes_comm_matrix)
            print("Explicit Transfers Transfers: \n", all_et_num_times_comm_matrix)

            outputfile_et_num_bytes_comm_matrix = "et_num_bytes_comm_matrix"
            outputfile_et_num_times_comm_matrix = "et_num_times_comm_matrix"

            # plot_comm_matrix(all_et_num_bytes_comm_matrix, args.num_gpus, outputfile_et_num_bytes_comm_matrix, args.scale)
            # plot_comm_matrix(all_et_num_times_comm_matrix, args.num_gpus, outputfile_et_num_times_comm_matrix, args.scale)

    def get_zero_copy_memory_transfers():
        # Zero-Copy Memory Transfers
        all_zc_comm = ZeroCopyInfoGenerator(self.num_gpus)
        all_zc_num_bytes_comm_matrix, all_zc_num_times_comm_matrix = all_zc_comm.generate_comm_matrix(metric_trace_file)

        if max(map(max, all_zc_num_bytes_comm_matrix)) != 0 and max(map(max, all_zc_num_times_comm_matrix)) !=0:
            print("ZeroCopy Memory Bytes: \n", all_zc_num_bytes_comm_matrix)
            print("ZeroCopy Memory Transfers: \n", all_zc_num_times_comm_matrix)
            # plot_bar_chart(all_zc_num_bytes_comm_matrix, self.num_gpus)
            # plot_bar_chart(all_zc_num_times_comm_matrix, self.num_gpus)

            # Intra-node Memory Transfers
            outputfile_intra_node_num_bytes_comm_matrix = "intra_node_num_bytes_comm_matrix"
            outputfile_intra_node_num_times_comm_matrix = "intra_node_num_times_comm_matrix"
            all_intra_node_num_bytes_comm_matrix = merge_matrices_for_intranode(all_et_num_bytes_comm_matrix, nccl_num_bytes_comm_matrix)
            all_intra_node_num_transfers_comm_matrix = merge_matrices_for_intranode(all_et_num_times_comm_matrix, nccl_num_times_comm_matrix)
            if max(map(max, all_intra_node_num_bytes_comm_matrix)) != 0 and max(map(max, all_intra_node_num_transfers_comm_matrix)) !=0:
                print("Intra-node Memory Bytes: \n", all_intra_node_num_bytes_comm_matrix)
                print("Intra-node Memory Transfers: \n", all_intra_node_num_transfers_comm_matrix)
                # plot_comm_matrix(all_intra_node_num_bytes_comm_matrix, num_devices, outputfile_intra_node_num_bytes_comm_matrix, scale)
                # plot_comm_matrix(all_intra_node_num_transfers_comm_matrix, num_devices, outputfile_intra_node_num_times_comm_matrix, scale)

    def get_nccl_memory_transfers(self):
        nccl_comm = NcclCommMatrixGenerator(args.num_gpus)
        nccl_num_bytes_comm_matrix, nccl_num_times_comm_matrix = nccl_comm.generate_comm_matrix(filepath_prefix=file_regex)

        print("Nccl Memory Bytes: \n", nccl_num_bytes_comm_matrix)
        print("Nccl Memory Transfers: \n", nccl_num_times_comm_matrix)

        outputfile_nccl_num_bytes_comm_matrix = "nccl_num_bytes_comm_matrix"
        outputfile_nccl_num_times_comm_matrix = "nccl_num_times_comm_matrix"

        # plot_comm_matrix(nccl_num_bytes_comm_matrix, args.num_gpus, outputfile_nccl_num_bytes_comm_matrix, args.scale)
        # plot_comm_matrix(nccl_num_times_comm_matrix, args.num_gpus, outputfile_nccl_num_times_comm_matrix, args.scale)
