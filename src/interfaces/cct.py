import os
import hatchet as ht
import pandas as pd
import networkx as nx
from logger import get_logger
from utils.sanitizer import Sanitizer
from utils.df import df_factorize_column, df_add_column, df_lookup_by_column
from utils.general import get_sorted_files
from networkx.readwrite import json_graph

LOGGER = get_logger(__name__)

class CCT():
    FILENAME = 'region_profile.json'
    COLUMNS = ["time (inc)", "time", "label", "class"]

    def __init__(self, data_dir):
        LOGGER.info(f"{type(self).__name__} interface triggered.")
        self.experiments = get_sorted_files(data_dir)
        self.file_paths = { exp: os.path.join(os.path.abspath(data_dir), f'{exp}/{self.FILENAME}') for exp in self.experiments }
        self.graph = nx.DiGraph()

        self.hts = {}
        self.nxgs = {}
        self.dfs = {}
        for exp in self.experiments:
            # LOGGER.info("adding graph for experiment:", exp)
            # Check if file exists.
            if os.path.exists(self.file_paths[exp]):
                self.hts[exp] = ht.GraphFrame.from_caliper_json(self.file_paths[exp])
                self.dfs[exp] = self.add_path_columns(self.hts[exp])
                self.nxgs[exp] = self.ht_graph_to_nxg(self.hts[exp])

    def get_idx(self, callsite):
        return self.callsite2idx[callsite]

    def get_callsite(self, idx):
        return self.idx2callsite[idx]

    def get_mean_runtime(self, exp, tag):
        pass
    
    def add_path_columns(self, ht) -> pd.DataFrame:
        """
        Add path columns to the GraphFrame.
        """        
        df = ht.dataframe
        self.idx2callsite, self.callsite2idx = df_factorize_column(df, "name")

        paths = {}
        callers = {}
        callees = {}

        for node in ht.graph.traverse():
            node_name = Sanitizer.from_htframe(node.frame)
            cs_idx = self.get_idx(node_name)
            paths[cs_idx] = [self.get_idx(Sanitizer.from_htframe(_)) for _ in node.paths()[0]]
            callers[cs_idx] = [self.get_idx(Sanitizer.from_htframe(_.frame)) for _ in node.parents]
            callees[cs_idx] = [self.get_idx(Sanitizer.from_htframe(_.frame)) for _ in node.children]

        df = df_add_column(df, "callees", apply_dict=callees, dict_default=[], apply_on="nid")
        df = df_add_column(df, "callers", apply_dict=callers, dict_default=[], apply_on="nid")
        df = df_add_column(df, "path", apply_dict=paths, dict_default=[], apply_on="nid")

        return df

    def ht_graph_to_nxg(self, ht) -> nx.DiGraph:
        """
        Constructs a networkX graph from hatchet graph.
        :param ht_graph: (hatchet.Graph) Hatchet Graph
        :return: (NetworkX.nxg) NetworkX graph
        """
        ht_graph = ht.graph
        ht_df = ht.dataframe
        assert isinstance(ht_df, pd.DataFrame)
        # assert isinstance(ht_graph, ht.graph.Graph)

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

        self._node_attributes(nxg, ht_df)
        self._edge_attributes(nxg)

        return nxg

    def _node_attributes(self, nxg, df):
        datamap = {}
        for callsite in nxg.nodes():
            for column in CCT.COLUMNS:
                if column not in datamap:
                    datamap[column] = {}

                callsite_idx = self.get_idx(callsite)
                _df = df_lookup_by_column(df, "name", callsite_idx)

                if column == "time (inc)":
                    datamap[column][callsite] = self.get_mean_runtime(_df, column)
                elif column == "time":
                    datamap[column][callsite] = self.get_mean_runtime(_df, column)
                elif column == "label":
                    datamap[column][callsite] = callsite
                elif column == "module":
                    datamap[column][callsite] = 'CPU'

        for idx, key in enumerate(datamap):
            nx.set_node_attributes(nxg, name=key, values=datamap[key])


    def _edge_attributes(self, nxg):
        pass

    def get_nxg(self, exp):
        if exp not in self.nxgs:
            raise Exception(f"Experiment {exp} not found.")
        return json_graph.node_link_data(self.nxgs[exp])
