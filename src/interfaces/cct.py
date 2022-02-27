import os
import hatchet as ht
import pandas as pd
import networkx as nx
from logger import get_logger
from utils.sanitizer import Sanitizer
from utils.df import df_factorize_column, df_add_column

LOGGER = get_logger(__name__)

class CCT():
    FILENAME = 'region_profile.json'
    COLUMNS = ["time (inc)", "time", "name", "module"]

    def __init__(self, data_dir):
        LOGGER.info(f"{type(self).__name__} interface triggered.")
        self.experiments = os.listdir(data_dir)
        self.file_paths = { exp: os.path.join(os.path.abspath(data_dir), f'{exp}/{self.FILENAME}') for exp in self.experiments }
        self.graph = nx.DiGraph()
        self.hts = { exp: ht.GraphFrame.from_caliper_json(self.file_paths[exp]) for exp in self.experiments }
        self.dfs = { exp: self.add_path_columns(self.hts[exp]) for exp in self.experiments }
        self.nxgs = { exp: self.ht_graph_to_nxg(self.hts[exp].graph) for exp in self.experiments }

    def add_path_columns(self, ht) -> pd.DataFrame:
        """
        Add path columns to the GraphFrame.
        """        
        df = ht.dataframe
        self.idx2callsite, self.callsite2idx = df_factorize_column(df, "name")

        paths = {}
        callers = {}
        callees = {}

        _csidx = lambda _: self.callsite2idx[_]  # noqa E731
        for node in ht.graph.traverse():
            node_name = Sanitizer.from_htframe(node.frame)
            cs_idx = _csidx(node_name)
            paths[cs_idx] = [_csidx(Sanitizer.from_htframe(_)) for _ in node.paths()[0]]
            callers[cs_idx] = [_csidx(Sanitizer.from_htframe(_.frame)) for _ in node.parents]
            callees[cs_idx] = [_csidx(Sanitizer.from_htframe(_.frame)) for _ in node.children]

        df = df_add_column(df, "callees", apply_dict=callees, dict_default=[], apply_on="nid")
        df = df_add_column(df, "callers", apply_dict=callers, dict_default=[], apply_on="nid")
        df = df_add_column(df, "path", apply_dict=paths, dict_default=[], apply_on="nid")

        return df

    @staticmethod
    def ht_graph_to_nxg(ht_graph) -> nx.DiGraph:
        """
        Constructs a networkX graph from hatchet graph.
        :param ht_graph: (hatchet.Graph) Hatchet Graph
        :return: (NetworkX.nxg) NetworkX graph
        """
        assert isinstance(ht_graph, ht.graph.Graph)

        nxg = nx.DiGraph()
        for root in ht_graph.roots:
            node_gen = root.traverse()
            node = root

            try:
                while node:

                    # Get all node paths from hatchet.
                    node_paths = node.paths()

                    # Loop through all the node paths.
                    for node_path in node_paths:
                        if len(node_path) >= 2:
                            src_name = Sanitizer.from_htframe(node_path[-2])
                            trg_name = Sanitizer.from_htframe(node_path[-1])
                            nxg.add_edge(src_name, trg_name)
                    node = next(node_gen)

            except StopIteration:
                pass
            finally:
                del root

        return nxg


    @staticmethod
    def _create_nxg_from_paths(paths) -> nx.DiGraph:
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
                source = path[j]
                target = path[j + 1]

                if not nxg.has_edge(source, target):
                    nxg.add_edge(source, target)

        return nxg

    def _edge_attributes(self):
        pass

    def _node_attributes(self):
        pass

    def get_nxg(self, exp):
        return self.nxgs[exp].to_json()