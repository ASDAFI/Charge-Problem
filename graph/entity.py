
import random
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
import numpy as np


class Graph(nx.DiGraph):
    def __init__(self) -> None:
        super().__init__()
        
        
        self.capacities: List[float] = list()
        self.charges: List[float] = list()
        self.consume_per_turn: List[float] = list()
        self.is_battery: List[bool] = list()
        self.color_map: List[str] = list()
        self.nodes_count: int = 0

        # TODO: load from file
        self.battery_color: str = "red"
        self.non_battery_color: str = "blue"
         
    def add_node(self, is_battery: bool = False, capacity: float = float("inf"), charge: float = None, consume_per_turn: float = 1.0) -> int:
        """
        return node_id value
        """
        self.capacities.append(capacity)
        self.is_battery.append(is_battery)

        if charge == None:
            charge = capacity
        self.charges.append(charge)

        self.consume_per_turn.append(consume_per_turn)
        
        node_id: int = self.nodes_count
        
        if is_battery:
            self.color_map.append(self.battery_color)
        else:
            self.color_map.append(self.non_battery_color)
        
        super().add_node(node_id)

        self.nodes_count += 1

        return node_id

    def get_last_node_id(self) -> int:
        return self.nodes_count - 1

    def get_nodes_count(self) -> int:
        return self.nodes_count
    
    def plot(self) -> None:
        nx.draw(self, node_color=self.color_map, with_labels=True, font_weight='bold')
        plt.show()
        

class CompleteGraph(Graph):
    def __init__(self) -> None:
        super().__init__()
    
    def randomize(self, min_nodes_count: int, max_nodes_count: int, min_batteries_count: int, max_batteries_count: int, min_capacity: int,
                   max_capacity: int, min_charge: float, max_charge: float,
                   min_consume_per_turn: float = 1.0, max_consume_per_turn: float = 1.0,
                   is_int_charge: bool = True, is_int_consume_per_turn: bool = True, is_int_capacity: bool = True,  is_all_full_charge: bool = True):
        
        nodes_count: int = random.randint(min_nodes_count, max_nodes_count)
        batteries_count: int = random.randint(min_batteries_count, min(max_batteries_count, nodes_count)) # TODO: fix specific situations
        charge: float
        consume_per_turn: float
        capacity: float

        for _ in range(batteries_count):
            if is_int_capacity:
                capacity = float(random.randint(min_capacity, max_capacity))
            else:
                capacity = random.randint(min_capacity, max_capacity - 1) + random.random()

            if is_all_full_charge:
                charge = capacity
            elif is_int_charge:
                charge = float(random.randint(min_charge, max_charge))
            else:
                charge = random.randint(min_charge, max_charge - 1) + random.random()
            
            if is_int_consume_per_turn:
                consume_per_turn = float(random.randint(min_consume_per_turn, max_consume_per_turn))
            else:
                consume_per_turn = random.randint(min_consume_per_turn, max_consume_per_turn) + random.random()
            
            self.add_node(True, capacity, charge, consume_per_turn)
        
        for _ in range(nodes_count - batteries_count):
            self.add_node()
        
    def add_node(self, is_battery: bool = False, capacity: float = float("inf"), charge: float = None, consume_per_turn: float = 1.0) -> int:
        super().add_node(is_battery, capacity, charge, consume_per_turn)
        
        u: int
        v: int = super().get_last_node_id()

        for u in range(v):
            super().add_edge(u, v)
            super().add_edge(v, u)
        
        return v


class HamiltonianCycle(Graph):
    def __init__(self) -> None:
        super().__init__()

    def randomize(self, min_nodes_count: int, max_nodes_count: int, min_batteries_count: int, max_batteries_count: int, min_capacity: int,
                   max_capacity: int, min_charge: float, max_charge: float,
                   min_consume_per_turn: float = 1.0, max_consume_per_turn: float = 1.0,
                   is_int_charge: bool = True, is_int_consume_per_turn: bool = True, is_int_capacity: bool = True,  is_all_full_charge: bool = True):
        
        nodes_count: int = random.randint(min_nodes_count, max_nodes_count)
        batteries_count: int = random.randint(min_batteries_count, min(max_batteries_count, nodes_count)) # TODO: fix specific situations
        charge: float
        consume_per_turn: float
        capacity: float

        for _ in range(batteries_count):
            if is_int_capacity:
                capacity = float(random.randint(min_capacity, max_capacity))
            else:
                capacity = random.randint(min_capacity, max_capacity - 1) + random.random()

            if is_all_full_charge:
                charge = capacity
            elif is_int_charge:
                charge = float(random.randint(min_charge, max_charge))
            else:
                charge = random.randint(min_charge, max_charge - 1) + random.random()
            
            if is_int_consume_per_turn:
                consume_per_turn = float(random.randint(min_consume_per_turn, max_consume_per_turn))
            else:
                consume_per_turn = random.randint(min_consume_per_turn, max_consume_per_turn) + random.random()
            
            super().add_node(True, capacity, charge, consume_per_turn)
        
        for _ in range(nodes_count - batteries_count):
            super().add_node()

        # TODO make more efficient
        u: int = 0
        v: int
        not_used_nodes: List[int] = list(range(1, nodes_count))
        while not_used_nodes != []:
            v = random.choice(not_used_nodes)
            not_used_nodes.remove(v)
            super().add_edge(u, v)
            u = v
        
        super().add_edge(u, 0)


class PreProcessedGraph:
    def __init__(self):
        self.nodes_count: np.int32
        self.floyd_matrix: np.ndarray
        self.capacities: np.ndarray
        self.charges: np.ndarray
        self.consume_per_turn: np.ndarray
        self.adj_matrix: np.ndarray
        self.is_battery: np.ndarray

    def load(self, graph: Graph):
        self.nodes_count = np.int32(graph.get_nodes_count())
        self.capacities = np.array(graph.capacities, dtype=np.float32)
        self.charges = np.array(graph.charges, dtype=np.float32)
        self.consume_per_turn = np.array(graph.consume_per_turn, dtype=np.float32)
        self.is_battery = np.array(graph.is_battery, dtype=np.int8)
        self.adj_matrix = nx.adjacency_matrix(graph).toarray().astype(dtype=np.int8)
        self.floyd_matrix = nx.floyd_warshall_numpy(graph).astype(dtype=np.int32)


def preprocess(graph: Graph) -> PreProcessedGraph:
    preprocessed_graph: PreProcessedGraph = PreProcessedGraph()
    preprocessed_graph.load(graph)
    return preprocessed_graph

