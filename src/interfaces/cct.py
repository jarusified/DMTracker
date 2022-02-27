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
        self.file_paths = [os.path.join(os.path.abspath(data_dir), f'{exp}/{self.FILENAME}') for exp in self.experiments]
        self.graph = nx.DiGraph()
        self.hts = [ht.GraphFrame.from_caliper_json(fp) for fp in self.file_paths]
        self.dfs = [self.add_path_columns(ht) for ht in self.hts]
        self.nxgs = [self.df_to_nxg(df) for df in self.dfs]

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

        print(paths)
        df = df_add_column(df, "callees", apply_dict=callees, dict_default=[], apply_on="nid")
        df = df_add_column(df, "callers", apply_dict=callers, dict_default=[], apply_on="nid")
        df = df_add_column(df, "path", apply_dict=paths, dict_default=[], apply_on="nid")

        return df

    def df_to_nxg(self, df) -> nx.DiGraph:
        pass

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

    def nxg_to_json(self, nxg):
        return nxg.to_json()