import os
import subprocess

from logger import get_logger
from generators import *
from utils.general import create_dir_after_check, get_latest_file

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
        self.output_dir = os.path.join(os.path.abspath(args.args["data_dir"]), self.app_name)

        # Run app with nvprof GPU Trace
        # self.gpu_trace_file = os.path.join(self.output_dir, "gpu_trace.csv")
        
        # Run app with Metric Trace
        self.metrics = [
            "dram_read_throughput",
            "dram_read_transactions",
            "dram_write_throughput",
            "dram_write_transactions",
            "gld_efficiency",
            "gld_throughput",
            "gld_transactions",
            "gst_efficiency",
            "gst_throughput",
            "gst_transactions",
            "sysmem_read_bytes",
            "sysmem_write_bytes",
            "dram_utilization",
            "l2_utilization",
            "double_precision_fu_utilization",
            "achieved_occupancy",
            "nvlink_user_data_received", 
            "nvlink_user_data_transmitted", 
        ]
        self.metrics_summary_file = os.path.join(self.output_dir, "metric_summary.csv")
        
        self.events = [
            "active_cycles",
            "active_warps",
            "active_cycles_pm",
            "active_warps_pm",
            "warps_launched",
            "divergent_branch",
            "shared_ld_bank_conflict",
            "shared_st_bank_conflict",
            "active_cycles",
        ]
        self.events_summary_file = os.path.join(self.output_dir, "events_summary.csv")

        self.runtime_metrics_file = os.path.join(self.output_dir, "runtime_metrics.csv")
        self.nsys_trace_qdrep_file = os.path.join(self.output_dir, "nsys_trace.qdrep")
        self.nsys_trace_json_file = os.path.join(self.output_dir, "nsys_trace.json")
        self.lstopo_svg_file = os.path.join(self.output_dir, "topology.svg")
        self.uvm_tracking_file = os.path.join(self.output_dir, "uvm-tracking.json")
        self.nvbit_trace_file = os.path.join(self.output_dir, "nvbit_trace.json")
        self.region_profile_file = os.path.join(self.output_dir, "region_profile.json")
        self.runtime_report_file = os.path.join(self.output_dir, "runtime_report.txt")

        create_dir_after_check(self.output_dir)
        
    def cali_to_json(self, from_file, to_file):
        cmd = f'cali-query -j {from_file} >> {to_file}'
        subprocess.run([cmd], shell=True)

    def start(self, nvprof_metrics="select"):
        """
        TODO: Run all the commands in parallel.
        """
        LOGGER.info("[Tracer] Runtime summary")
        runtime_metrics_cmd = f'{self.cmd} -m {self.runtime_metrics_file}'
        subprocess.run([runtime_metrics_cmd], shell=True)

        LOGGER.info("[Tracer] Metrics summary")
        if(nvprof_metrics == "all"):
            metric_summary_cmd = f'nvprof --metrics all --csv --log-file {self.metrics_summary_file} {self.cmd}'
        else:
            metric_summary_cmd = f'nvprof --metrics {",".join(self.metrics)} --csv --log-file {self.metrics_summary_file} {self.cmd}'
        subprocess.run([metric_summary_cmd], shell=True)

        if(len(self.events) > 0):
            LOGGER.info("[Tracer] Events summary")
            event_summary_cmd = f'nvprof --events {",".join(self.events)} --csv --log-file{self.events_summary_file} {self.cmd}'
            subprocess.run([event_summary_cmd], shell=True)

        LOGGER.info("[Tracer] Nsys trace summary")
        nsys_metrics_cmd = f'nsys profile --trace=cuda,nvtx -d 20 --sample=none -o {self.nsys_trace_qdrep_file} {self.cmd}'
        subprocess.run([nsys_metrics_cmd], shell=True)

        LOGGER.info("[Tracer] Caliper NV-Bit")
        caliper_configs ="nvbit-trace"
        caliper_metrics_cmd = f'CALI_CONFIG_PROFILE={caliper_configs} {self.cmd}'
        subprocess.run([caliper_metrics_cmd], shell=True)

        f = get_latest_file(os.getcwd())
        self.cali_to_json(f, self.nvbit_trace_file)
        os.remove(f)

        LOGGER.info("[Tracer] Caliper UVM tracking total")
        caliper_configs ="uvm-tracking-total"
        caliper_metrics_cmd = f'CALI_CONFIG_PROFILE={caliper_configs},output=stdout {self.cmd}'
        subprocess.run([caliper_metrics_cmd], shell=True)

        f = get_latest_file(os.getcwd())
        self.cali_to_json(f, self.uvm_tracking_file)
        os.remove(f)

        LOGGER.info("[Tracer] Caliper Hatchet region profile")
        caliper_configs ="Hatchet-region-profile"
        caliper_metrics_cmd = f'CALI_CONFIG={caliper_configs},output=stdout {self.cmd} >> {self.region_profile_file}'
        subprocess.run([caliper_metrics_cmd], shell=True)

        LOGGER.info("[Tracer] Caliper Hatchet runtime-report")
        caliper_configs ="runtime-report"
        caliper_metrics_cmd = f'CALI_CONFIG={caliper_configs},output=stdout {self.cmd} >> {self.runtime_report_file}'
        subprocess.run([caliper_metrics_cmd], shell=True)

        LOGGER.info("[Tracer] LSTOPO SVG dump")
        topology_cmd = f'lstopo --of svg >> {self.lstopo_svg_file}'
        subprocess.run([topology_cmd], shell=True)
    
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
