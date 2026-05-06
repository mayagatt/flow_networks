import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch



def _draw_panel(ax, G, pos, norm_values, cmap,
                signed=False, directions=None):
    """
    Draw one graph panel onto `ax`.

    Parameters
    ----------
    ax          : matplotlib Axes
    G           : networkx graph
    pos         : node positions dict
    norm_values : array in [0,1] for color + width encoding
    cmap        : colormap
    signed      : whether to flip arrow direction by sign
    directions  : array of +1/-1 per edge (required if signed=True)
    """
    edge_colors = cmap(norm_values)
    edge_widths = 1 + 4 * norm_values
    edges = list(G.edges())

    if not signed:
        nx.draw(
            G, pos, ax=ax,
            with_labels=True,
            node_color='steelblue',
            node_size=800,
            font_color='white',
            font_size=14,
            edge_color=edge_colors,
            width=edge_widths,
        )
    else:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='steelblue', node_size=800)
        nx.draw_networkx_labels(G, pos, ax=ax, font_color='white', font_size=14)

        for i, (u, v) in enumerate(edges):
            src, dst = (u, v) if directions[i] >= 0 else (v, u)
            ax.add_patch(FancyArrowPatch(
                posA=pos[src], posB=pos[dst],
                arrowstyle='-|>',
                color=edge_colors[i],
                linewidth=edge_widths[i],
                mutation_scale=20,
                connectionstyle='arc3,rad=0.08',
                shrinkA=18, shrinkB=18,
                zorder=2,
            ))

    ax.set_axis_off()




def _prepare_values(values, cmap, signed, log_scale, label, v_min=None, v_max=None):
    """
    Returns (plot_values, norm_values, v_min, v_max, cbar_label, directions, cmap).
    
    v_min, v_max : optional fixed range for colorbar and normalization.
                   If None, inferred from data.
    """
    values = np.array(values, dtype=float)

    if signed:
        directions = np.sign(values)
        values     = np.abs(values)
        cbar_label = f"|{label}|"
        if cmap is plt.cm.coolwarm:
            cmap = plt.cm.viridis
    else:
        directions = None
        cbar_label = label

    if log_scale:
        values     = np.log10(values + 1e-10)
        cbar_label = f"log({cbar_label})"

    # use overrides if provided, otherwise infer from data
    v_min = v_min if v_min is not None else values.min()
    v_max = v_max if v_max is not None else values.max()

    v_range     = v_max - v_min
    norm_values = (
        (values - v_min) / v_range
        if v_range > 1e-10
        else np.full_like(values, 0.5)
    )

    return values, norm_values, v_min, v_max, cbar_label, directions, cmap

def plot_edge_property(
    G, pos, edge_values, title='', label='',
    v_min=None, v_max=None,
    cmap=plt.cm.cool, log_scale=False, signed=False,
):
    _, norm_values, v_min, v_max, cbar_label, directions, cmap = _prepare_values(
        edge_values, cmap, signed, log_scale, label, v_min=v_min, v_max=v_max
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_panel(ax, G, pos, norm_values, cmap,
                signed=signed, directions=directions)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(v_min, v_max))
    sm.set_array([])

    cb = fig.colorbar(sm, ax=ax)
    cb.ax.tick_params(labelsize=14)
    cb.set_label(cbar_label, fontsize=16)

    ax.set_title(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def conductance_update_PD(p_C,p_F, B, lr):
    """ Update the conductances according to d/dk(Power_dissipation_F - Power_dissipation_C)
    Power_dissipation_ij = (p_j - p_i^2) * K_ij
    => d/dk(Power_dissipation_ij) = (p_j - p_i^2)
    """
    dp_F = B.T @ p_F   # pressure drop across each edge, free phase
    dp_C = B.T @ p_C   # pressure drop across each edge, clamped phase

    delta_dp = dp_F**2 - dp_C**2
    return lr * delta_dp

def conductance_update_SR(q_C,q_F, K, lr):
    k = np.diag(K)
    dk = k**2 * (q_F - q_C)
    return lr * dk


def get_graph_conductances_and_inverse_laplacian(G):
    """ Extract conductances and pressures from the graph """
    edges = list(G.edges())
    K_adj = np.diag([G[u][v]['weight'] for u, v in edges])
    
    L_weighted = nx.laplacian_matrix(G).toarray()
    L_pinv = np.linalg.pinv(L_weighted)
    return K_adj, L_pinv

def loss_function(Q, Q_goal):
    """ Simple squared error loss on the flow of a specific edge """
    return (Q - Q_goal)**2





# ─────────────────────────────────────────────────────────────────────────────
# HELPER: draw a single graph panel
# ─────────────────────────────────────────────────────────────────────────────

from matplotlib.pyplot import tick_params


def _draw_panel(ax, G, pos, norm_values, cmap,
                signed=False, directions=None):
    """
    Draw one graph panel onto `ax`.

    Parameters
    ----------
    ax          : matplotlib Axes
    G           : networkx graph
    pos         : node positions dict
    norm_values : array in [0,1] for color + width encoding
    cmap        : colormap
    signed      : whether to flip arrow direction by sign
    directions  : array of +1/-1 per edge (required if signed=True)
    """
    edge_colors = cmap(norm_values)
    edge_widths = 1 + 4 * norm_values
    edges = list(G.edges())

    if not signed:
        nx.draw(
            G, pos, ax=ax,
            with_labels=True,
            node_color='steelblue',
            node_size=800,
            font_color='white',
            font_size=14,
            edge_color=edge_colors,
            width=edge_widths,
        )
    else:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='steelblue', node_size=800)
        nx.draw_networkx_labels(G, pos, ax=ax, font_color='white', font_size=14)

        for i, (u, v) in enumerate(edges):
            src, dst = (u, v) if directions[i] >= 0 else (v, u)
            ax.add_patch(FancyArrowPatch(
                posA=pos[src], posB=pos[dst],
                arrowstyle='-|>',
                color=edge_colors[i],
                linewidth=edge_widths[i],
                mutation_scale=20,
                connectionstyle='arc3,rad=0.08',
                shrinkA=18, shrinkB=18,
                zorder=2,
            ))

    ax.set_axis_off()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: prepare values (sign split, log, normalize)
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_values(values, cmap, signed, log_scale, label):
    """
    Returns (plot_values, norm_values, v_min, v_max, cbar_label, directions, cmap).
    """
    values = np.array(values, dtype=float)

    if signed:
        directions = np.sign(values)
        values     = np.abs(values)
        cbar_label = f"|{label}|"
        if cmap is plt.cm.coolwarm:
            cmap = plt.cm.viridis
    else:
        directions = None
        cbar_label = label

    if log_scale:
        values     = np.log10(values + 1e-10)
        cbar_label = f"log({cbar_label})"

    v_min, v_max = values.min(), values.max()
    v_range      = v_max - v_min
    norm_values  = (
        (values - v_min) / v_range
        if v_range > 1e-10
        else np.full_like(values, 0.5)
    )

    return values, norm_values, v_min, v_max, cbar_label, directions, cmap


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: single-panel plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_edge_property(
    G, pos, edge_values, title='', label='',
    cmap=plt.cm.cool, log_scale=False, signed=False,
):
    _, norm_values, v_min, v_max, cbar_label, directions, cmap = _prepare_values(
        edge_values, cmap, signed, log_scale, label
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_panel(ax, G, pos, norm_values, cmap,
                signed=signed, directions=directions)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(v_min, v_max))
    sm.set_array([])

    cb = fig.colorbar(sm, ax=ax)
    cb.ax.tick_params(labelsize=14)
    cb.set_label(cbar_label, fontsize=16)

    ax.set_title(title, fontsize=20)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: side-by-side before/after plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_before_after(
    G_before, G_after, pos,
    values_before, values_after,
    title='', label='',
    cmap=plt.cm.cool, log_scale=False, signed=False,
    subtitles=('Initial', 'Trained'),
):
    # prepare each side independently for directions, then reconcile scale
    _, nv_b, v_min_b, v_max_b, cbar_label, dir_b, cmap = _prepare_values(
        values_before, cmap, signed, log_scale, label
    )
    _, nv_a, v_min_a, v_max_a, _,          dir_a, _    = _prepare_values(
        values_after,  cmap, signed, log_scale, label
    )

    # shared scale across both panels for fair comparison
    v_min, v_max = min(v_min_b, v_min_a), max(v_max_b, v_max_a)
    v_range      = v_max - v_min

    def _renorm(v_min_local, nv_local, v_min_local_raw, values_raw):
        # recompute norm_values against the shared scale
        vals = np.array(values_raw, dtype=float)
        if signed:
            vals = np.abs(vals)
        if log_scale:
            vals = np.log10(vals + 1e-10)
        return (vals - v_min) / v_range if v_range > 1e-10 else np.full_like(vals, 0.5)

    nv_b = _renorm(v_min_b, nv_b, v_min_b, values_before)
    nv_a = _renorm(v_min_a, nv_a, v_min_a, values_after)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),dpi=300)

    for ax, G_plot, nv, dirs, subtitle in zip(
        axes,
        [G_before,  G_after],
        [nv_b,      nv_a   ],
        [dir_b,     dir_a  ],
        subtitles,
    ):
        _draw_panel(ax, G_plot, pos, nv, cmap, signed=signed, directions=dirs)
        ax.set_title(f"{subtitle} {title}", fontsize=20)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(v_min, v_max))#), tick_params={'labelsize': 12})
    sm.set_array([])
    # fig.colorbar(sm, cax=cbar_ax, label=cbar_label)

    cb = fig.colorbar(sm, cax=cbar_ax, label=cbar_label)
    cb.ax.tick_params(labelsize=14)
    cb.set_label(cbar_label, fontsize=16)
    
    # plt.setbox(False)
    plt.show()


def plot_history(history, edges, target_edge_nodes, target_current):
    history_flows = np.array([hi['flows'] for hi in history])
    history_steps = [hi['step'] for hi in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # All edge flows over time
    for i, (u, v) in enumerate(edges):
        is_target = (u, v) == target_edge_nodes
        if is_target:
            axes[0].plot(history_steps, history_flows[:, i],
                        lw=2.5, ls='-', color='crimson',
                        label=f"target edge", zorder=5)
                        #  label=f"({u},{v}) target edge", zorder=5)
        else:
            axes[0].plot(history_steps, history_flows[:, i],
                        lw=1, color='steelblue', alpha=0.4) #, label=f"({u},{v})")
        # lw = 2.5 if (u, v) == target_edge_nodes else 1
        # ls = '-' if (u, v) == target_edge_nodes else '--'
        # axes[0].plot(history_steps, history_flows[:, i], label=f"({u},{v})", lw=lw, ls=ls)


    font_size = 18
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[0].axhline(target_current, color='k', lw=1.5, ls=':', label=f"goal={target_current}")
    axes[0].set_xlabel("Iteration", fontsize=font_size)
    axes[0].set_ylabel("Edge flow", fontsize=font_size)
    # axes[0].set_yscale('log'

    # axes[0].set_title("Edge Flows over Training")
    axes[0].legend()

    # Conductance over time
    history_conductances = [hi['conductances'] for hi in history]
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    for i, (u, v) in enumerate(edges):
        if (u,v) == target_edge_nodes:
            axes[1].plot(history_steps, [h[i] for h in history_conductances], label=f"target edge ({u},{v})", lw=2.5, color='crimson')
        else:   
            axes[1].plot(history_steps, [h[i] for h in history_conductances], color='steelblue', alpha=0.4, label=f"({u},{v})", lw=1)
    axes[1].set_xlabel("Iteration", fontsize=font_size)
    axes[1].set_ylabel("Conductance", fontsize=font_size)
    # axes[1].set_yscale('log')
    # fix legend position
    axes[1].legend(loc='lower right')

    # # Loss over time
    # history_losses = [h['loss'] for h in history]
    # axes[2].tick_params(axis='both', which='major', labelsize=12)
    # axes[2].scatter(history_steps, history_losses) #, color='tomato') #, linewidth=3)
    # axes[2].set_xlabel("Iteration", fontsize=font_size)
    # axes[2].set_ylabel("Loss", fontsize=font_size)
    # axes[2].set_yscale('log')

    plt.tight_layout()
    plt.show()