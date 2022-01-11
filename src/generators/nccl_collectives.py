import re
import glob

class NcclCommMatrixGenerator():
    def __init__(self, num_devices):
        self.headers = ['src', 'dst', 'size']
        self.num_bytes_comm_matrix = [[0] * num_devices for _ in range(num_devices)]
        self.num_times_comm_matrix = [[0] * num_devices for _ in range(num_devices)]

    def generate_comm_matrix(self, filepath_prefix="comscribe_*_*.csv"):
        file_paths = glob.glob(filepath_prefix)

        for file_path in file_paths:
            with open(file_path) as fp:
                lines = fp.readlines()

                for line in lines:
                    nodeName, commId, deviceId, src, dst, size, algo = line.split(",")
                    self.num_bytes_comm_matrix[int(dst)][int(src)] += int(size)
                    self.num_times_comm_matrix[int(dst)][int(src)] += 1

        return self.num_bytes_comm_matrix, self.num_times_comm_matrix

class ZeroCopyInfoGenerator():
    def __init__(self, num_devices):
        # Needed headers for zerocopy memory
        self.headers = [
            'Device', 'nvlink_user_data_received',
            'nvlink_user_data_transmitted', 'sysmem_read_bytes',
            'sysmem_write_bytes']

        self.num_bytes_comm_matrix = [[0] * (4) for _ in range(num_devices)]
        self.num_times_comm_matrix = [[0] * (4) for _ in range(num_devices)]

    def has_all_headers(self, line):
        for header in self.headers:
            if not re.search(header, line):
                return False
        return True

    def get_indices_of_headers(self, line):
        name_to_index = {}
        for header in self.headers:
            name_to_index[header] = line.index(header)
        return name_to_index

    def get_size_and_gpu_ids(self, splitted_line, name_to_index, num_of_elems):
        device_id,nvlink_user_data_received,nvlink_user_data_transmitted,sysmem_read_bytes,sysmem_write_bytes = None, None, None, None, None
        if len(splitted_line) == num_of_elems + 1:
            device_id = splitted_line[name_to_index['Device']]
            nvlink_user_data_received = splitted_line[name_to_index['nvlink_user_data_received'] + 1]
            nvlink_user_data_transmitted = splitted_line[name_to_index['nvlink_user_data_transmitted'] + 1]
            sysmem_read_bytes = splitted_line[name_to_index['sysmem_read_bytes'] + 1]
            sysmem_write_bytes = splitted_line[name_to_index['sysmem_write_bytes'] + 1]
        return (device_id,nvlink_user_data_received,nvlink_user_data_transmitted,sysmem_read_bytes,sysmem_write_bytes)

    def get_num_of_elems(self, splitted_line):
        return len(splitted_line)

    def _clean_and_split_line(self, line):
        clean_line = line.replace('"', '')
        splitted_line = clean_line.split(',')
        return splitted_line

    def generate_comm_matrix(self, filepath):
        multiply_by = 1
        find_headers = True
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                line = fp.readline()
                stripped_line = line.strip()
                if find_headers:
                    if self.has_all_headers(stripped_line):
                        splitted_line = self._clean_and_split_line(stripped_line)
                        name_to_index = self.get_indices_of_headers(splitted_line)
                        num_of_elems = self.get_num_of_elems(splitted_line)
                        line = fp.readline()
                        find_headers = False
                else:
                    splitted_line = self._clean_and_split_line(stripped_line)
                    zerocopy_memory_info = self.get_size_and_gpu_ids(splitted_line, name_to_index, num_of_elems)
                    if not None in zerocopy_memory_info:
                        device = zerocopy_memory_info[0]
                        device_id = int(re.findall('\((.*?)\)', device)[0])
                        self.num_bytes_comm_matrix[device_id][0] += int(zerocopy_memory_info[1])
                        self.num_bytes_comm_matrix[device_id][1] += int(zerocopy_memory_info[2])
                        self.num_bytes_comm_matrix[device_id][2] += int(zerocopy_memory_info[3])
                        self.num_bytes_comm_matrix[device_id][3] += int(zerocopy_memory_info[4])
                        if int(zerocopy_memory_info[1]):
                            self.num_times_comm_matrix[device_id][0] += 1.0
                        if int(zerocopy_memory_info[2]):
                            self.num_times_comm_matrix[device_id][1] += 1.0
                        if int(zerocopy_memory_info[3]):
                            self.num_times_comm_matrix[device_id][2] += 1.0
                        if int(zerocopy_memory_info[4]):
                            self.num_times_comm_matrix[device_id][3] += 1.0
        return self.num_bytes_comm_matrix, self.num_times_comm_matrix
