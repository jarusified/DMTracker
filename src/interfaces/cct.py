import json
import os
import hatchet as ht
import networkx as nx

class CCT():
    FILENAME = 'region_profile.json'
    COLUMNS = ["time (inc)", "time", "name", "module"]

    def __init__(self, data_dir):
        self.experiments = os.listdir(data_dir)
        self.file_paths = [os.path.join(os.path.abspath(data_dir), f'{exp}/{self.FILENAME}') for exp in self.experiments]
        self.graph = nx.DiGraph()
        self.hts = [ht.GraphFrame.from_caliper_json(fp) for fp in self.file_paths]
        self.nxgs = [self.ht_to_nxg(ht) for ht in self.hts]

    def ht_to_nxg(self, ht) -> nx.DiGraph:
        return nx.DiGraph()

    @staticmethod
    def _create_nxg_from_paths(paths):
        """
        :param paths:
        :return:
        """
        assert isinstance(paths, list)

        nxg = nx.DiGraph()

        # go over all path
        for i, path in enumerate(paths):

            # go over the callsites in this path
            plen = len(path)

            for j in range(plen - 1):
                source = Sanitizer.sanitize(path[j])
                target = Sanitizer.sanitize(path[j + 1])

                if not nxg.has_edge(source, target):
                    nxg.add_edge(source, target)

        return nxg

    def _edge_attributes(self):
        pass

    def _node_attributes(self):
        pass

    def nxg_to_json(self, nxg):
        return nxg.to_json()