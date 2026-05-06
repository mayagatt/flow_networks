
# ══════════════════════════════════════════════════════════════════════════════
# QUICK-START EXAMPLE  (run as a script or paste into a notebook)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 1. Define the network ─────────────────────────────────────────────────
    W = np.array([
        [0,   0.8, 0.4, 0.0],
        [0.8, 0,   1.0, 0.2],
        [0.4, 1.0, 0,   0.7],
        [0.0, 0.2, 0.7, 0  ],
    ])

    net = FlowNetwork(W)
    net.plot_conductances()

    Q_in = np.array([1., 0., 0., -1.])
    net.print_edges(Q_in)
    net.plot_flows(Q_in)

    # ── 2. Run contrastive optimization ───────────────────────────────────────
    opt = ContrastiveOptimizer(
        network=net,
        Q_in=Q_in,
        target_edge=(1, 2),
        target_flow=0.3,
        learning_rate=0.01,
        nudge_strength=0.01,
        loss_tol=1e-4,
    )

    opt.run(max_iter=100_000, log_every=1000)
    opt.plot_loss()
    opt.plot_history_flows()
    opt.plot_results()