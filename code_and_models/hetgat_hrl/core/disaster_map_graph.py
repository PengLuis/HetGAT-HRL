from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class MapComplexitySpec:
    name: str
    map_size_m: float
    node_count: int
    min_node_spacing_m: float
    redundant_edge_radius_m: float
    redundant_edge_prob: float


# 当前先落地 M；S/L 后续按同结构追加即可。
MAP_COMPLEXITY_PRESETS: Dict[str, MapComplexitySpec] = {
    "M": MapComplexitySpec(
        name="M",
        map_size_m=5000.0,
        node_count=40,
        min_node_spacing_m=300.0,
        redundant_edge_radius_m=1000.0,
        redundant_edge_prob=0.8,
    )
}


class DisasterMapGraph:
    """
    灾后路网图生成器（UGV/UAV 共享底图）：
    1) 节点：5000x5000 连续空间 + 排斥采样 + Node0 中心锚定
    2) 连边：MST 保底连通 + 半径阈值内随机冗余干道
    3) 缓存：欧氏距离矩阵 + 拓扑最短路矩阵（O(1) 查表）
    """

    def __init__(
        self,
        seed: int = 0,
        map_size_m: float = 5000.0,
        node_count: int = 40,
        min_node_spacing_m: float = 300.0,
        redundant_edge_radius_m: float = 1000.0,
        redundant_edge_prob: float = 0.8,
    ) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.map_size_m = float(map_size_m)
        self.node_count = int(node_count)
        self.min_node_spacing_m = float(min_node_spacing_m)
        self.redundant_edge_radius_m = float(redundant_edge_radius_m)
        self.redundant_edge_prob = float(np.clip(redundant_edge_prob, 0.0, 1.0))

        self.node_xy: Dict[int, Tuple[float, float]] = {}
        self.graph: nx.Graph = nx.Graph()
        self.euclidean_dist_matrix: Optional[np.ndarray] = None
        self.shortest_path_matrix: Optional[np.ndarray] = None

        self._generate_nodes()
        self._generate_edges()
        self._cache_distances()

    @classmethod
    def from_complexity(cls, complexity: str, seed: int = 0) -> "DisasterMapGraph":
        key = str(complexity).upper().strip()
        if key not in MAP_COMPLEXITY_PRESETS:
            raise ValueError(f"未知复杂度: {complexity}, 可选: {list(MAP_COMPLEXITY_PRESETS.keys())}")
        spec = MAP_COMPLEXITY_PRESETS[key]
        return cls(
            seed=seed,
            map_size_m=spec.map_size_m,
            node_count=spec.node_count,
            min_node_spacing_m=spec.min_node_spacing_m,
            redundant_edge_radius_m=spec.redundant_edge_radius_m,
            redundant_edge_prob=spec.redundant_edge_prob,
        )

    def _generate_nodes(self) -> None:
        """
        节点生成：
        - Node 0 固定在 (2500, 2500)
        - 其余节点使用带最小间距约束的拒绝采样（Poisson-like）
        """
        if self.node_count < 1:
            raise ValueError("node_count 必须 >= 1")
        center = 0.5 * self.map_size_m
        self.node_xy[0] = (center, center)

        max_trials = 200_000
        trials = 0
        next_id = 1
        while next_id < self.node_count:
            trials += 1
            if trials > max_trials:
                raise RuntimeError(
                    f"节点排斥采样失败：已尝试 {max_trials} 次，"
                    f"请放宽 min_node_spacing_m 或减小 node_count。"
                )
            x = float(self.rng.uniform(0.0, self.map_size_m))
            y = float(self.rng.uniform(0.0, self.map_size_m))

            ok = True
            for _, (px, py) in self.node_xy.items():
                if float(np.hypot(x - px, y - py)) < self.min_node_spacing_m:
                    ok = False
                    break
            if not ok:
                continue
            self.node_xy[next_id] = (x, y)
            next_id += 1

        self.graph.clear()
        for i in range(self.node_count):
            self.graph.add_node(i, x=float(self.node_xy[i][0]), y=float(self.node_xy[i][1]))

    def _generate_edges(self) -> None:
        """
        连边生成：
        Step 1: 全连接图上做 MST，保底全连通
        Step 2: d<=阈值 的未连接点对，以概率 p 加冗余边
        """
        n = self.node_count
        if n <= 1:
            return
        # 完全距离矩阵（供 MST + 冗余边判断复用）
        xy = np.array([self.node_xy[i] for i in range(n)], dtype=np.float64)
        diff = xy[:, None, :] - xy[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        full = nx.Graph()
        full.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                full.add_edge(i, j, weight=float(dist[i, j]))

        mst = nx.minimum_spanning_tree(full, algorithm="kruskal", weight="weight")
        for u, v, data in mst.edges(data=True):
            self.graph.add_edge(int(u), int(v), weight=float(data["weight"]))

        # 冗余干道
        for i in range(n):
            for j in range(i + 1, n):
                if self.graph.has_edge(i, j):
                    continue
                if float(dist[i, j]) <= self.redundant_edge_radius_m:
                    if float(self.rng.uniform(0.0, 1.0)) <= self.redundant_edge_prob:
                        self.graph.add_edge(i, j, weight=float(dist[i, j]))

    def _cache_distances(self) -> None:
        """
        距离缓存层：
        - euclidean_dist_matrix: UAV 飞行耗能/时延查表
        - shortest_path_matrix: UGV 路网行驶查表
        """
        n = self.node_count
        xy = np.array([self.node_xy[i] for i in range(n)], dtype=np.float64)
        diff = xy[:, None, :] - xy[None, :, :]
        self.euclidean_dist_matrix = np.linalg.norm(diff, axis=2).astype(np.float64)

        sp = np.full((n, n), np.inf, dtype=np.float64)
        for i in range(n):
            sp[i, i] = 0.0
        all_pairs = nx.all_pairs_dijkstra_path_length(self.graph, weight="weight")
        for src, dmap in all_pairs:
            s = int(src)
            for dst, d in dmap.items():
                sp[s, int(dst)] = float(d)
        self.shortest_path_matrix = sp

    def get_euclidean_distance(self, src: int, dst: int) -> float:
        if self.euclidean_dist_matrix is None:
            raise RuntimeError("euclidean_dist_matrix 未初始化")
        return float(self.euclidean_dist_matrix[int(src), int(dst)])

    def get_shortest_path_distance(self, src: int, dst: int) -> float:
        if self.shortest_path_matrix is None:
            raise RuntimeError("shortest_path_matrix 未初始化")
        return float(self.shortest_path_matrix[int(src), int(dst)])

    def get_neighbors(self, node_id: int) -> List[int]:
        return sorted(int(v) for v in self.graph.neighbors(int(node_id)))

    def average_degree(self) -> float:
        if self.node_count <= 0:
            return 0.0
        return float(2.0 * self.graph.number_of_edges() / float(self.node_count))

    def to_topology_payload(
        self,
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Set[int]], np.ndarray, np.ndarray]:
        adjacency: Dict[int, Set[int]] = {i: set() for i in range(self.node_count)}
        for u, v in self.graph.edges():
            adjacency[int(u)].add(int(v))
            adjacency[int(v)].add(int(u))
        if self.euclidean_dist_matrix is None or self.shortest_path_matrix is None:
            raise RuntimeError("距离矩阵未就绪")
        return (
            dict(self.node_xy),
            adjacency,
            self.euclidean_dist_matrix.copy(),
            self.shortest_path_matrix.copy(),
        )
