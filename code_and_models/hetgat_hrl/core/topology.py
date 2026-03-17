from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from hetgat_hrl.core.disaster_map_graph import DisasterMapGraph
from hetgat_hrl.core.mdp_spec import EnvConfig


@dataclass(frozen=True)
class Node:
    node_id: int
    x: float
    y: float
    elevation_m: float = 0.0
    slope_norm: float = 0.0


@dataclass
class EdgeAttr:
    roughness_norm: float = 0.0
    building_density_norm: float = 0.0
    infra_bottleneck_norm: float = 0.0
    base_vulnerability: float = 0.0


class GraphTopology:
    """
    Topology layer with two sources:
    1) synthetic random connected graph
    2) OSM + DEM driven graph (fallback to synthetic when unavailable)
    """

    def __init__(
        self,
        nodes: Dict[int, Node],
        adjacency: Dict[int, Set[int]],
        edge_attrs: Optional[Dict[Tuple[int, int], EdgeAttr]] = None,
        euclidean_dist_matrix: Optional[np.ndarray] = None,
        shortest_path_matrix: Optional[np.ndarray] = None,
    ):
        self.nodes = nodes
        self.adjacency = {int(k): set(int(x) for x in v) for k, v in adjacency.items()}
        self.blocked_edges: Set[Tuple[int, int]] = set()
        self.edge_attrs: Dict[Tuple[int, int], EdgeAttr] = edge_attrs or {}
        self.euclidean_dist_matrix: Optional[np.ndarray] = euclidean_dist_matrix
        self.shortest_path_matrix: Optional[np.ndarray] = shortest_path_matrix
        if not self.edge_attrs:
            self._init_default_edge_attrs()

    @staticmethod
    def edge_key(src: int, dst: int) -> Tuple[int, int]:
        return (min(int(src), int(dst)), max(int(src), int(dst)))


    @staticmethod
    def _edge_u01(src: int, dst: int, salt: int = 0) -> float:
        """
        Deterministic pseudo-random U(0,1) from edge key.
        Keeps defaults reproducible across runs/platforms.
        """
        a = int(min(src, dst))
        b = int(max(src, dst))
        x = ((a * 73856093) ^ (b * 19349663) ^ (int(salt) * 83492791)) & 0xFFFFFFFF
        x ^= (x >> 13)
        x = (x * 1274126177) & 0xFFFFFFFF
        x ^= (x >> 16)
        return float((x + 0.5) / 4294967296.0)

    def _default_edge_attr(
        self,
        src: int,
        dst: int,
        cut_edges: Optional[Set[Tuple[int, int]]] = None,
    ) -> EdgeAttr:
        a = self.nodes[int(src)]
        b = self.nodes[int(dst)]
        slope = float(np.clip(0.5 * (a.slope_norm + b.slope_norm), 0.0, 1.0))

        # Roughness prior: steeper links are rougher.
        rough = float(np.clip(0.15 + 0.70 * slope, 0.0, 1.0))

        # Building-density default prior (literature-aligned urban morphology bands):
        # low/mid/high coverage ~= 20% / 60% / 20%.
        # Values mapped to normalized density in [0,1].
        # Midpoint closer to map center tends to be denser.
        xs = np.array([n.x for n in self.nodes.values()], dtype=np.float64)
        ys = np.array([n.y for n in self.nodes.values()], dtype=np.float64)
        cx = float(0.5 * (xs.min() + xs.max())) if xs.size else 0.0
        cy = float(0.5 * (ys.min() + ys.max())) if ys.size else 0.0
        diag = float(
            np.hypot(
                max(xs.max() - xs.min(), 1e-6),
                max(ys.max() - ys.min(), 1e-6),
            )
        ) if xs.size else 1.0
        mx = 0.5 * (a.x + b.x)
        my = 0.5 * (a.y + b.y)
        r_norm = float(np.clip(np.hypot(mx - cx, my - cy) / max(0.5 * diag, 1e-6), 0.0, 1.5))
        urbanity = float(np.exp(-((r_norm / 0.95) ** 2)))

        u_b = self._edge_u01(src, dst, salt=1)
        if u_b < 0.20:
            b_base = 0.22
        elif u_b < 0.80:
            b_base = 0.40
        else:
            b_base = 0.58
        bldg = float(np.clip(b_base + 0.12 * (urbanity - 0.5), 0.0, 1.0))

        # Bridge/tunnel/critical-corridor default ratio:
        # tunnel 2%, bridge 6%, other critical 12%, regular 80%.
        # These are structure-type priors, then boosted by graph bottleneck structure.
        u_i = self._edge_u01(src, dst, salt=2)
        if u_i < 0.02:
            infra = 1.00  # tunnel-like critical segment
        elif u_i < 0.08:
            infra = 0.85  # bridge-like critical segment
        elif u_i < 0.20:
            infra = 0.60  # other key corridor
        else:
            infra = 0.20  # regular road segment

        # Structural bottleneck boost from topology.
        deg_src = len(self.adjacency.get(int(src), set()))
        deg_dst = len(self.adjacency.get(int(dst), set()))
        if deg_src <= 2 or deg_dst <= 2:
            infra = max(infra, 0.45)
        if cut_edges is not None and self.edge_key(src, dst) in cut_edges:
            infra = max(infra, 0.60)
        infra = float(np.clip(infra, 0.0, 1.0))

        # V_base = clip((Slope/35)*(1+0.65*D_bldg)*(1+C_infra), 0, 1)
        # slope_norm already corresponds to normalized slope term.
        v_base = float(np.clip((slope) * (1.0 + 0.65 * bldg) * (1.0 + infra), 0.0, 1.0))
        return EdgeAttr(
            roughness_norm=rough,
            building_density_norm=bldg,
            infra_bottleneck_norm=infra,
            base_vulnerability=v_base,
        )

    def _init_default_edge_attrs(self) -> None:
        cut_edges = self._bridge_edges(len(self.nodes), self.adjacency)
        for src, nbs in self.adjacency.items():
            for dst in nbs:
                if src >= dst:
                    continue
                self.edge_attrs[(src, dst)] = self._default_edge_attr(
                    src=src, dst=dst, cut_edges=cut_edges
                )

    @staticmethod
    def _mean_degree(adjacency: Dict[int, Set[int]]) -> float:
        n = max(len(adjacency), 1)
        m2 = float(sum(len(v) for v in adjacency.values()))
        return m2 / float(n)

    def average_degree(self) -> float:
        return self._mean_degree(self.adjacency)

    @staticmethod
    def _target_edge_count(
        num_nodes: int, num_edges: int, avg_degree_min: float, avg_degree_max: float
    ) -> int:
        min_edges = int(np.ceil(0.5 * avg_degree_min * num_nodes))
        max_edges = int(np.floor(0.5 * avg_degree_max * num_nodes))
        if min_edges > max_edges:
            min_edges = max_edges = max(1, int(round(0.5 * 3.5 * num_nodes)))
        if num_edges <= 0:
            return int(round(0.5 * 0.5 * (avg_degree_min + avg_degree_max) * num_nodes))
        return int(np.clip(num_edges, min_edges, max_edges))

    @staticmethod
    def _as_undirected_edges(adjacency: Dict[int, Set[int]]) -> Set[Tuple[int, int]]:
        es: Set[Tuple[int, int]] = set()
        for a, nbs in adjacency.items():
            for b in nbs:
                if a == b:
                    continue
                es.add((min(a, b), max(a, b)))
        return es

    @staticmethod
    def _rebuild_adjacency(num_nodes: int, edges: Set[Tuple[int, int]]) -> Dict[int, Set[int]]:
        out: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
        for a, b in edges:
            out[a].add(b)
            out[b].add(a)
        return out

    @staticmethod
    def _is_connected(num_nodes: int, adjacency: Dict[int, Set[int]]) -> bool:
        if num_nodes <= 1:
            return True
        seen = set()
        stack = [0]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for nb in adjacency.get(cur, set()):
                if nb not in seen:
                    stack.append(nb)
        return len(seen) == num_nodes

    @staticmethod
    def _connected_components(adjacency: Dict[int, Set[int]]) -> List[Set[int]]:
        nodes = set(adjacency.keys())
        comps: List[Set[int]] = []
        while nodes:
            root = nodes.pop()
            comp = {root}
            stack = [root]
            while stack:
                cur = stack.pop()
                for nb in adjacency.get(cur, set()):
                    if nb not in comp:
                        comp.add(nb)
                        if nb in nodes:
                            nodes.remove(nb)
                        stack.append(nb)
            comps.append(comp)
        return comps

    @staticmethod
    def _bridge_edges(num_nodes: int, adjacency: Dict[int, Set[int]]) -> Set[Tuple[int, int]]:
        bridges: Set[Tuple[int, int]] = set()
        base_edges = GraphTopology._as_undirected_edges(adjacency)
        for e in list(base_edges):
            a, b = e
            adj2 = {k: set(v) for k, v in adjacency.items()}
            adj2[a].discard(b)
            adj2[b].discard(a)
            if not GraphTopology._is_connected(num_nodes, adj2):
                bridges.add(e)
        return bridges

    @staticmethod
    def _ensure_connected(
        nodes: Dict[int, Node], adjacency: Dict[int, Set[int]]
    ) -> Dict[int, Set[int]]:
        out = {k: set(v) for k, v in adjacency.items()}
        num_nodes = len(nodes)
        if num_nodes <= 1:
            return out
        comps = GraphTopology._connected_components(out)
        while len(comps) > 1:
            c1 = comps[0]
            c2 = comps[1]
            best = None
            best_d = float("inf")
            for a in c1:
                for b in c2:
                    na, nb = nodes[a], nodes[b]
                    d = float(np.hypot(na.x - nb.x, na.y - nb.y))
                    if d < best_d:
                        best_d = d
                        best = (a, b)
            if best is None:
                break
            a, b = best
            out[a].add(b)
            out[b].add(a)
            comps = GraphTopology._connected_components(out)
        return out

    @staticmethod
    def _rebalance_degree_band(
        nodes: Dict[int, Node],
        adjacency: Dict[int, Set[int]],
        target_edges: int,
    ) -> Dict[int, Set[int]]:
        out = GraphTopology._ensure_connected(nodes, adjacency)
        num_nodes = len(nodes)
        edges = GraphTopology._as_undirected_edges(out)

        # Add edges if sparse.
        while len(edges) < target_edges:
            best = None
            best_d = float("inf")
            for a in range(num_nodes):
                for b in range(a + 1, num_nodes):
                    if (a, b) in edges:
                        continue
                    na, nb = nodes[a], nodes[b]
                    d = float(np.hypot(na.x - nb.x, na.y - nb.y))
                    if d < best_d:
                        best_d = d
                        best = (a, b)
            if best is None:
                break
            edges.add(best)

        # Remove edges if dense while preserving connectivity.
        while len(edges) > target_edges:
            adj_tmp = GraphTopology._rebuild_adjacency(num_nodes, edges)
            bridges = GraphTopology._bridge_edges(num_nodes, adj_tmp)
            removable = [e for e in edges if e not in bridges]
            if not removable:
                break
            # Prefer removing long non-bridge edges.
            removable.sort(
                key=lambda e: np.hypot(
                    nodes[e[0]].x - nodes[e[1]].x, nodes[e[0]].y - nodes[e[1]].y
                ),
                reverse=True,
            )
            edges.remove(removable[0])

        return GraphTopology._rebuild_adjacency(num_nodes, edges)

    @staticmethod
    def _normalize_xy(
        raw_xy: Dict[int, Tuple[float, float]], world_size_m: float
    ) -> Dict[int, Tuple[float, float]]:
        xs = np.array([xy[0] for xy in raw_xy.values()], dtype=np.float64)
        ys = np.array([xy[1] for xy in raw_xy.values()], dtype=np.float64)
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        dx = max(x1 - x0, 1e-6)
        dy = max(y1 - y0, 1e-6)
        out: Dict[int, Tuple[float, float]] = {}
        for k, (x, y) in raw_xy.items():
            xn = (x - x0) / dx
            yn = (y - y0) / dy
            out[k] = (float(xn * world_size_m), float(yn * world_size_m))
        return out

    @staticmethod
    def _load_dem(dem_npy_path: str) -> Optional[np.ndarray]:
        p = Path(str(dem_npy_path)) if dem_npy_path else None
        if p is None or not p.exists():
            return None
        if p.suffix.lower() != ".npy":
            return None
        try:
            arr = np.load(str(p))
            if arr.ndim != 2:
                return None
            return arr.astype(np.float64)
        except Exception:
            return None

    @staticmethod
    def _annotate_dem(
        nodes_xy: Dict[int, Tuple[float, float]],
        world_size_m: float,
        dem: Optional[np.ndarray],
    ) -> Dict[int, Tuple[float, float]]:
        if dem is None:
            return {k: (0.0, 0.0) for k in nodes_xy}
        gy, gx = np.gradient(dem)
        slope_field = np.hypot(gx, gy)
        s_ref = float(np.percentile(slope_field, 95))
        s_ref = max(s_ref, 1e-6)
        h, w = dem.shape
        out: Dict[int, Tuple[float, float]] = {}
        for k, (x, y) in nodes_xy.items():
            u = float(np.clip(x / max(world_size_m, 1e-6), 0.0, 1.0))
            v = float(np.clip(y / max(world_size_m, 1e-6), 0.0, 1.0))
            j = int(round(u * (w - 1)))
            i = int(round(v * (h - 1)))
            elev = float(dem[i, j])
            slope = float(np.clip(slope_field[i, j] / s_ref, 0.0, 1.0))
            out[k] = (elev, slope)
        return out

    @staticmethod
    def _safe_float(d: dict, keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in d:
                try:
                    return float(d[k])
                except Exception:
                    continue
        return None

    @staticmethod
    def generate_random_connected(
        num_nodes: int,
        num_edges: int,
        seed: int,
        world_size_m: float = 3000.0,
        avg_degree_min: float = 3.2,
        avg_degree_max: float = 3.8,
    ) -> "GraphTopology":
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0.0, world_size_m, size=(num_nodes, 2))
        nodes = {
            i: Node(node_id=i, x=float(coords[i, 0]), y=float(coords[i, 1]))
            for i in range(num_nodes)
        }
        adjacency: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}

        # Build a connected backbone.
        order = list(rng.permutation(num_nodes))
        for i in range(num_nodes - 1):
            a, b = int(order[i]), int(order[i + 1])
            adjacency[a].add(b)
            adjacency[b].add(a)

        target_edges = GraphTopology._target_edge_count(
            num_nodes, num_edges, avg_degree_min, avg_degree_max
        )
        adjacency = GraphTopology._rebalance_degree_band(nodes, adjacency, target_edges)
        return GraphTopology(nodes=nodes, adjacency=adjacency)

    @staticmethod
    def generate_from_osm_dem(
        num_nodes: int,
        num_edges: int,
        seed: int,
        osm_graphml_path: str,
        dem_npy_path: str,
        world_size_m: float = 3000.0,
        avg_degree_min: float = 3.2,
        avg_degree_max: float = 3.8,
    ) -> "GraphTopology":
        try:
            import networkx as nx
        except Exception:
            return GraphTopology.generate_random_connected(
                num_nodes=num_nodes,
                num_edges=num_edges,
                seed=seed,
                world_size_m=world_size_m,
                avg_degree_min=avg_degree_min,
                avg_degree_max=avg_degree_max,
            )

        p = Path(str(osm_graphml_path)) if osm_graphml_path else None
        if p is None or not p.exists():
            return GraphTopology.generate_random_connected(
                num_nodes=num_nodes,
                num_edges=num_edges,
                seed=seed,
                world_size_m=world_size_m,
                avg_degree_min=avg_degree_min,
                avg_degree_max=avg_degree_max,
            )

        rng = np.random.default_rng(seed)
        try:
            g_raw = nx.read_graphml(str(p))
        except Exception:
            return GraphTopology.generate_random_connected(
                num_nodes=num_nodes,
                num_edges=num_edges,
                seed=seed,
                world_size_m=world_size_m,
                avg_degree_min=avg_degree_min,
                avg_degree_max=avg_degree_max,
            )

        g = nx.Graph()
        g.add_nodes_from(g_raw.nodes(data=True))
        g.add_edges_from(g_raw.edges())
        if g.number_of_nodes() == 0:
            return GraphTopology.generate_random_connected(
                num_nodes=num_nodes,
                num_edges=num_edges,
                seed=seed,
                world_size_m=world_size_m,
                avg_degree_min=avg_degree_min,
                avg_degree_max=avg_degree_max,
            )

        # Largest connected component.
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
        all_nodes = list(g.nodes())
        if not all_nodes:
            return GraphTopology.generate_random_connected(
                num_nodes=num_nodes,
                num_edges=num_edges,
                seed=seed,
                world_size_m=world_size_m,
                avg_degree_min=avg_degree_min,
                avg_degree_max=avg_degree_max,
            )

        target_n = min(int(num_nodes), len(all_nodes))
        start = all_nodes[int(rng.integers(0, len(all_nodes)))]
        bfs_order = list(nx.bfs_tree(g, start).nodes())
        selected = bfs_order[:target_n]
        if len(selected) < target_n:
            rem = [n for n in all_nodes if n not in set(selected)]
            rng.shuffle(rem)
            selected.extend(rem[: target_n - len(selected)])
        sub = g.subgraph(selected).copy()

        # Re-index nodes to 0..n-1.
        old_to_new = {old: i for i, old in enumerate(sub.nodes())}
        raw_xy: Dict[int, Tuple[float, float]] = {}
        for old, new in old_to_new.items():
            data = sub.nodes[old]
            x = GraphTopology._safe_float(
                data, ["x", "lon", "lng", "longitude", "X", "LONGITUDE"]
            )
            y = GraphTopology._safe_float(
                data, ["y", "lat", "latitude", "Y", "LATITUDE"]
            )
            if x is None or y is None:
                x = float(rng.uniform(0.0, 1.0))
                y = float(rng.uniform(0.0, 1.0))
            raw_xy[new] = (float(x), float(y))
        xy = GraphTopology._normalize_xy(raw_xy, world_size_m=world_size_m)

        adjacency: Dict[int, Set[int]] = {i: set() for i in range(len(old_to_new))}
        for a_old, b_old in sub.edges():
            a = old_to_new[a_old]
            b = old_to_new[b_old]
            if a == b:
                continue
            adjacency[a].add(b)
            adjacency[b].add(a)

        dem = GraphTopology._load_dem(dem_npy_path)
        dem_anno = GraphTopology._annotate_dem(xy, world_size_m=world_size_m, dem=dem)
        nodes: Dict[int, Node] = {}
        for i in range(len(old_to_new)):
            elev, slope = dem_anno.get(i, (0.0, 0.0))
            nodes[i] = Node(
                node_id=i,
                x=float(xy[i][0]),
                y=float(xy[i][1]),
                elevation_m=float(elev),
                slope_norm=float(slope),
            )

        target_edges = GraphTopology._target_edge_count(
            len(nodes), num_edges, avg_degree_min, avg_degree_max
        )
        adjacency = GraphTopology._rebalance_degree_band(nodes, adjacency, target_edges)
        return GraphTopology(nodes=nodes, adjacency=adjacency)

    @staticmethod
    def generate_from_disaster_map(cfg: EnvConfig) -> "GraphTopology":
        """
        浣跨敤 DisasterMapGraph 鐢熸垚鍣細
        - 鑺傜偣: Node0 涓績閿氬畾 + 鏈€灏忛棿璺濇帓鏂ラ噰鏍?        - 杩炶竟: MST 淇濆簳 + 鍐椾綑骞查亾
        - 璺濈: 娆ф皬/鏈€鐭矾鍙岀煩闃电紦瀛?        """
        if str(cfg.map_complexity).upper() in {"M"}:
            gen = DisasterMapGraph.from_complexity(
                complexity=str(cfg.map_complexity),
                seed=int(cfg.seed),
            )
        else:
            gen = DisasterMapGraph(
                seed=int(cfg.seed),
                map_size_m=float(cfg.map_size_m),
                node_count=int(cfg.num_nodes or cfg.n_nodes),
                min_node_spacing_m=float(cfg.min_node_spacing_m),
                redundant_edge_radius_m=float(cfg.redundant_edge_radius_m),
                redundant_edge_prob=float(cfg.redundant_edge_prob),
            )
        node_xy, adjacency, euclid_m, sp_m = gen.to_topology_payload()
        nodes = {
            int(i): Node(node_id=int(i), x=float(x), y=float(y))
            for i, (x, y) in node_xy.items()
        }
        return GraphTopology(
            nodes=nodes,
            adjacency=adjacency,
            euclidean_dist_matrix=euclid_m,
            shortest_path_matrix=sp_m,
        )

    @staticmethod
    def build_from_config(cfg: EnvConfig) -> "GraphTopology":
        source = str(cfg.map_source).strip().lower()
        if source in {"disaster_map", "disaster", "m_complexity"}:
            return GraphTopology.generate_from_disaster_map(cfg)
        if source == "osm_dem":
            return GraphTopology.generate_from_osm_dem(
                num_nodes=int(cfg.num_nodes or cfg.n_nodes),
                num_edges=int(cfg.num_edges),
                seed=int(cfg.seed),
                osm_graphml_path=str(cfg.osm_graphml_path or ""),
                dem_npy_path=str(cfg.dem_npy_path or ""),
                avg_degree_min=float(cfg.avg_degree_min),
                avg_degree_max=float(cfg.avg_degree_max),
            )
        return GraphTopology.generate_random_connected(
            num_nodes=int(cfg.num_nodes or cfg.n_nodes),
            num_edges=int(cfg.num_edges),
            seed=int(cfg.seed),
            avg_degree_min=float(cfg.avg_degree_min),
            avg_degree_max=float(cfg.avg_degree_max),
        )

    def neighbors(self, node_id: int) -> List[int]:
        nbs = []
        for nb in self.adjacency.get(node_id, set()):
            if not self.is_blocked(node_id, nb):
                nbs.append(nb)
        return nbs

    def is_connected(self, src: int, dst: int) -> bool:
        return dst in self.adjacency.get(src, set())

    def edge_distance(self, src: int, dst: int) -> float:
        if self.euclidean_dist_matrix is not None:
            return float(self.euclidean_dist_matrix[int(src), int(dst)])
        a = self.nodes[src]
        b = self.nodes[dst]
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def edge_attr(self, src: int, dst: int) -> EdgeAttr:
        k = self.edge_key(src, dst)
        if k not in self.edge_attrs:
            self.edge_attrs[k] = self._default_edge_attr(src=int(k[0]), dst=int(k[1]))
        return self.edge_attrs[k]

    def is_blocked(self, src: int, dst: int) -> bool:
        k = (min(src, dst), max(src, dst))
        return k in self.blocked_edges

    def set_blocked(self, src: int, dst: int, blocked: bool) -> None:
        k = (min(src, dst), max(src, dst))
        if blocked:
            self.blocked_edges.add(k)
        else:
            self.blocked_edges.discard(k)

    def blocked_ratio(self) -> float:
        total = sum(len(v) for v in self.adjacency.values()) // 2
        if total <= 0:
            return 0.0
        return float(len(self.blocked_edges) / total)

    def path_exists(self, src: int, dst: int, ignore_blocked: bool = False) -> bool:
        if int(src) == int(dst):
            return True
        seen = set()
        stack = [int(src)]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for nb in self.adjacency.get(cur, set()):
                if (not ignore_blocked) and self.is_blocked(cur, nb):
                    continue
                if nb == int(dst):
                    return True
                if nb not in seen:
                    stack.append(nb)
        return False

    def shortest_path_distance(
        self, src: int, dst: int, ignore_blocked: bool = False
    ) -> float:
        src = int(src)
        dst = int(dst)
        if src == dst:
            return 0.0
        # O(1) cache path length when blocked edges are ignored
        # or when the graph currently has no blocked edges.
        if self.shortest_path_matrix is not None and (
            ignore_blocked or len(self.blocked_edges) == 0
        ):
            return float(self.shortest_path_matrix[src, dst])
        inf = float("inf")
        dist = {k: inf for k in self.nodes}
        dist[src] = 0.0
        visited: Set[int] = set()
        while True:
            cur = None
            cur_d = inf
            for n, d in dist.items():
                if n in visited:
                    continue
                if d < cur_d:
                    cur_d = d
                    cur = n
            if cur is None or cur_d >= inf:
                break
            if cur == dst:
                return float(cur_d)
            visited.add(cur)
            for nb in self.adjacency.get(cur, set()):
                if (not ignore_blocked) and self.is_blocked(cur, nb):
                    continue
                nd = cur_d + self.edge_distance(cur, nb)
                if nd < dist[nb]:
                    dist[nb] = nd
        return inf
