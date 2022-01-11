import re
import glob

class H2DCudaMemcpyCommMatrixGenerator():

    def __init__(self, num_devices):
        # Needed headers for H2D memcpy
        self.headers = ['Device', 'Size', 'Name']
        self.num_bytes_comm_matrix = [[0] * (num_devices + 1) for _ in range(num_devices + 1)]
        self.num_times_comm_matrix = [[0] * (num_devices + 1) for _ in range(num_devices + 1)]

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
        size, src_index, dst_index = None, None, None
        if len(splitted_line) == num_of_elems:
            mem_transfer_type = splitted_line[name_to_index['Name']]
            if mem_transfer_type == "[CUDA memcpy HtoD]":
                size = splitted_line[name_to_index['Size']]
                dst_index = splitted_line[name_to_index['Device']]
            elif mem_transfer_type == "[CUDA memcpy DtoH]":
                size = splitted_line[name_to_index['Size']]
                src_index = splitted_line[name_to_index['Device']]
        return size, src_index, dst_index

    def get_num_of_elems(self, splitted_line):
        return len(splitted_line)

    def get_size_type(self, line, name_to_index):
        splitted_line = self._clean_and_split_line(line)
        size_type = splitted_line[name_to_index['Size']]
        return size_type

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
                        size_type = self.get_size_type(line, name_to_index)
                        if size_type == "KB":
                            multiply_by = 1024
                        elif size_type == "MB":
                            multiply_by = 1024 * 1024
                        elif size_type == "GB":
                            multiply_by = 1024 * 1024 * 1024
                        find_headers = False
                else:
                    splitted_line = self._clean_and_split_line(stripped_line)
                    comm_size, src_dev, dst_dev = self.get_size_and_gpu_ids(splitted_line, name_to_index, num_of_elems)
                    if comm_size:
                        if not src_dev and dst_dev:
                            dst_id = int(re.findall('\((.*?)\)', dst_dev)[0])
                            self.num_bytes_comm_matrix[dst_id + 1][0] += float(comm_size) * multiply_by
                            self.num_times_comm_matrix[dst_id + 1][0] += 1.0
                        elif src_dev and not dst_dev:
                            src_id = int(re.findall('\((.*?)\)', src_dev)[0])
                            self.num_bytes_comm_matrix[0][src_id + 1] += float(comm_size) * multiply_by
                            self.num_times_comm_matrix[0][src_id + 1] += 1.0
        return self.num_bytes_comm_matrix, self.num_times_comm_matrix