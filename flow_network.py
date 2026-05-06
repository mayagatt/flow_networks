
"""
flow_network.py
===============
Toy model for contrastive learning on flow networks.

Two parts:
  1. Network — build, inspect, and visualize a weighted flow network
  2. Optimization — run contrastive Hebbian learning to meet a target edge flow

Usage
-----
    from flow_network import FlowNetwork, ContrastiveOptimizer
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class FlowNetwork:
    """
    A weighted undirected flow network.

    Parameters
    ----------
    W : np.ndarray, shape (n, n)
        Weighted adjacency matrix. Off-diagonal entries are edge conductances;
        diagonal is ignored. The matrix should be symmetric.

    Example
    -------
    >>> W = np.array([[0, 0.8, 0.4, 0.0],
    ...               [0.8, 0, 1.0, 0.2],
    ...               [0.4, 1.0, 0, 0.7],
    ...               [0.0, 0.2, 0.7, 0]])
    >>> net = FlowNetwork(W)
    >>> net.plot_conductances()
    """

    def __init__(self, W: np.ndarray):
        self.G = nx.from_numpy_array(W)
        self.pos = nx.spring_layout(self.G, seed=42)
        self._build_matrices()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_matrices(self):
        """Recompute B, K, L, L_pinv from current G edge weights."""
        self.edges = list(self.G.edges())
        self.B = -nx.incidence_matrix(self.G, oriented=True).toarray()  # (n_nodes, n_edges)
        K_vec = np.array([self.G[u][v]['weight'] for u, v in self.edges])
        self.K = np.diag(K_vec)
        self.L = nx.laplacian_matrix(self.G).toarray().astype(float)
        self.L_pinv = np.linalg.pinv(self.L)

    # ── Physics ───────────────────────────────────────────────────────────────

    def solve(self, Q_in: np.ndarray) -> dict:
        """
        Solve pressures and edge flows given external current injections.

        Parameters
        ----------
        Q_in : np.ndarray, shape (n_nodes,)
            External current injection per node. Must sum to zero.

        Returns
        -------
        dict with keys 'pressures', 'flows', 'power_dissipation'
        """
        p = self.L_pinv @ Q_in
        p -= p.min()                        # gauge: min pressure = 0
        flows = self.K @ self.B.T @ p
        K_vec = np.diag(self.K)
        dp = self.B.T @ p
        power_dissipation = K_vec * dp ** 2
        return dict(pressures=p, flows=flows, power_dissipation=power_dissipation)

    def edge_index(self, edge: tuple) -> int:
        """Return the index of (u, v) in the edge list."""
        return self.edges.index(edge)

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _normalize(self, values, log_scale=False):
        v = np.array(values, dtype=float)
        if log_scale:
            v = np.log10(np.abs(v) + 1e-10)
        v_min, v_max = v.min(), v.max()
        v_range = v_max - v_min
        norm = (v - v_min) / v_range if v_range > 1e-10 else np.full_like(v, 0.5)
        return norm, v_min, v_max

    def plot_edge_property(self, edge_values, title='', label='',
                           cmap=plt.cm.coolwarm, log_scale=False, ax=None):
        """Plot the network with edges colored and sized by edge_values."""
        norm_v, v_min, v_max = self._normalize(edge_values, log_scale)
        cb_label = f"log10({label})" if log_scale else label

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(7, 5))

        nx.draw(self.G, self.pos, ax=ax,
                with_labels=True,
                node_color='steelblue',
                node_size=800,
                font_color='white',
                font_size=14,
                edge_color=cmap(norm_v),
                width=[1 + 4 * v for v in norm_v])
        ax.set_title(title)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(v_min, v_max))
        plt.colorbar(sm, ax=ax, label=cb_label)

        if standalone:
            plt.show()

    def plot_before_after(self, values_before, values_after,
                          G_before=None, G_after=None,
                          title='', label='',
                          cmap=plt.cm.coolwarm, log_scale=False,
                          label_fontsize=18):
        """Side-by-side before/after comparison on a shared color scale."""
        G_before = G_before or self.G
        G_after  = G_after  or self.G

        v_b = np.array(values_before, dtype=float)
        v_a = np.array(values_after,  dtype=float)
        if log_scale:
            v_b = np.log10(np.abs(v_b) + 1e-10)
            v_a = np.log10(np.abs(v_a) + 1e-10)
            label = f"log10({label})"

        v_min = min(v_b.min(), v_a.min())
        v_max = max(v_b.max(), v_a.max())
        v_range = v_max - v_min

        def norm(v):
            return (v - v_min) / v_range if v_range > 1e-10 else np.full_like(v, 0.5)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, G_plot, values, subtitle in zip(
                axes, [G_before, G_after], [v_b, v_a], ['Initial', 'Trained']):
            nv = norm(values)
            nx.draw(G_plot, self.pos, ax=ax,
                    with_labels=True,
                    node_color='steelblue',
                    node_size=800,
                    font_color='white',
                    font_size=14,
                    edge_color=cmap(nv),
                    width=[1 + 4 * v for v in nv])
            ax.set_title(f"{subtitle} — {title}")

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(v_min, v_max))
        fig.colorbar(sm, cax=cbar_ax, label=label)
        cb = fig.colorbar(sm, cax=cbar_ax, label=label)
        cb.set_label(label, fontsize=label_fontsize)

        plt.show()

    def plot_conductances(self, log_scale=False):
        K_vec = np.diag(self.K)
        self.plot_edge_property(K_vec, title='Edge Conductances',
                                label='Conductance', cmap=plt.cm.YlOrRd,
                                log_scale=log_scale)

    def plot_flows(self, Q_in: np.ndarray, log_scale=False):
        state = self.solve(Q_in)
        self.plot_edge_property(state['flows'], title='Edge Flows',
                                label='Flow', cmap=plt.cm.coolwarm,
                                log_scale=log_scale)

    def print_edges(self, Q_in: np.ndarray):
        """Print edge index, nodes, flow, and conductance."""
        state = self.solve(Q_in)
        print(f"{'idx':>4}  {'edge':>8}  {'flow':>10}  {'conductance':>12}")
        print("-" * 40)
        for i, (u, v) in enumerate(self.edges):
            print(f"{i:>4}  ({u},{v}):  {state['flows'][i]:>10.4f}  "
                  f"{self.G[u][v]['weight']:>12.4f}")


