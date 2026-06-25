# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class ContrastiveOptimizer:
    """
    Contrastive Hebbian learning on a FlowNetwork.

    Adjusts edge conductances so that the flow on `target_edge` matches
    `target_flow`, using the difference in power dissipation between a
    free phase and a nudged (clamped) phase.

    Parameters
    ----------
    network      : FlowNetwork
    Q_in         : np.ndarray  — external current injections (source/sink vector)
    target_edge  : tuple       — (u, v) edge to control
    target_flow  : float       — desired flow on that edge
    update_func  : str         — 'PD' = power dissipation difference, 'SR' = shear rate difference
    learning_rate: float       — step size for conductance updates
    nudge_strength: float      — scales how strongly Q_in is nudged in clamped phase
    loss_tol     : float       — stop when loss < loss_tol

    Example
    -------
    >>> opt = ContrastiveOptimizer(
    ...     network=net,
    ...     Q_in=np.array([1., 0., 0., -1.]),
    ...     target_edge=(1, 2),
    ...     target_flow=0.3,
    ...     learning_rate=0.01,
    ...     nudge_strength=0.01,
    ...     loss_tol=1e-4,
    ... )
    >>> opt.run(max_iter=100_000, log_every=1000)
    >>> opt.plot_loss()
    >>> opt.plot_results()
    """

    def __init__(self, network: FlowNetwork, Q_in: np.ndarray,
                 target_edge: tuple, target_flow: float,
                 update_func: str,  # 'PD' = power dissipation difference, 'SR' = shear rate difference
                 learning_rate: float = 0.01,
                 nudge_strength: float = 0.01,
                 loss_tol: float = 1e-4):

        self.net_init = network                     # original (never mutated)
        self.net = copy.deepcopy(network)           # working copy
        self.Q_in = Q_in.astype(float)
        self.target_edge = target_edge
        self.desired_flow_target = target_flow
        self.lr = learning_rate
        self.eps = nudge_strength
        self.loss_tol = loss_tol
        self.update_func = update_func

        self.target_idx = self.net.edge_index(target_edge)
        self.history = []
        self.losses = []

    # ── Loss ──────────────────────────────────────────────────────────────────

    def loss(self, flow_on_target: float) -> float:
        return 0.5 * (flow_on_target - self.desired_flow_target) ** 2

    # ── Update rule ───────────────────────────────────────────────────────────

    def _conductance_update_PD(self, p_F: np.ndarray, p_C: np.ndarray) -> np.ndarray:
        """
        dk_ij = lr * (dp_C_ij² - dp_F_ij²)
        Increase conductance where clamped phase dissipates more → drives flow
        toward target.
        """
        B = self.net.B
        dp_F = B.T @ p_F
        dp_C = B.T @ p_C
        return self.lr * (dp_C ** 2 - dp_F ** 2)
    
    def _conductance_update_SR(self, q_F: np.ndarray, q_C: np.ndarray) -> np.ndarray:
        """
        dk_ij = lr * k_ij² * (q_F_ij - q_C_ij)
        Increase conductance where free phase has more flow → drives flow
        toward target.
        """
        k = np.diag(self.net.K)
        dk = k**2 * (q_F - q_C)
        return self.lr * dk

    # ── Single step ───────────────────────────────────────────────────────────

    def step(self, p_F: np.ndarray,q_F: np.ndarray , current_loss: float):
        """Run one free→clamp→update→rebuild cycle. Returns updated p_F, loss."""
        net = self.net

        # ── Clamped phase: nudge Q_in to push more flow through target edge ──
        state_F = net.solve_q_p(self.Q_in)
        p_F     = state_F['pressures']
        q_F     = state_F['flows']

        # distance of the current flow from the desired flow of the target edge
        loss   = (q_F[self.target_idx] - self.desired_flow_target)**2 

        # add flow to the target edge to the source/sink vector, according to the loss
        Qin_C   = self.Q_in + self.eps * loss * net.B[:, self.target_idx]

        # resolve the state of the network in its clamped state
        state_C = net.solve_q_p(Qin_C)
        p_C     = state_C['pressures']

        # ── Conductance update ────────────────────────────────────────────────
        # compute dk for each edge based on the difference between the free and clamped phases
        if self.update_func == 'PD':
            dk = self._conductance_update_PD(p_F, p_C)
        elif self.update_func == 'SR':
            dk = self._conductance_update_SR(q_F=q_F, q_C=state_C['flows'])
        else:
            raise ValueError(f"Invalid update_func: {self.update_func}. Choose 'PD' or 'SR'.")

        # apply dk to the edge weights, ensuring they remain positive
        k = net.get_K()
        new_k = np.maximum(k + dk, 1e-5)
        net.set_K(new_k)

        # ── Free phase with updated network ───────────────────────────────────
        state_F = net.solve_q_p(self.Q_in)
        p_F_new = state_F['pressures']
        new_loss = self.loss(state_F['flows'][self.target_idx])

        return p_F_new, new_loss, state_F

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, max_iter: int = 100_000, log_every: int = 1000):
        """
        Run the contrastive optimization loop.

        Parameters
        ----------
        max_iter  : maximum number of iterations
        log_every : print and record history every N steps
        """
        # Initial free phase
        state_F = self.net.solve_q_p(self.Q_in)
        p_F     = state_F['pressures']
        current_loss = self.loss(state_F['flows'][self.target_idx])

        print(f"Initial flow on {self.target_edge}: "
              f"{state_F['flows'][self.target_idx]:.4f}  |  "
              f"target: {self.target_flow}  |  loss: {current_loss:.6f}")

        # Save initial state for before/after plots
        self._state_init = state_F
        self._K_init     = np.diag(self.net_init.K).copy()

        print("\n=== Optimization start ===")
        for ii in range(1, max_iter + 1):
            p_F, current_loss, state_F = self.step(p_F, state_F['flows'], current_loss)
            self.losses.append(current_loss)

            if ii % log_every == 0:
                flow = state_F['flows'][self.target_idx]
                print(f"Iter {ii:>7}  |  flow: {flow:.5f}  |  loss: {current_loss:.6f}")
                self.history.append({
                    'step':         ii,
                    'loss':         current_loss,
                    'flow_target':  flow,
                    'conductances': np.diag(self.net.K).copy(),
                    'flows':        state_F['flows'].copy(),
                    'pressures':    state_F['pressures'].copy(),
                })

            if current_loss < self.loss_tol:
                print(f"\n✓ Converged at iteration {ii}  |  loss: {current_loss:.2e}")
                break
        else:
            print(f"\n✗ Did not converge in {max_iter} iterations  |  "
                  f"final loss: {current_loss:.6f}")

        # Always record final state
        self._state_final = state_F
        self._K_final     = np.diag(self.net.K).copy()

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_loss(self):
        """Plot loss curve over all iterations."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.losses, color='tomato', lw=1.5)
        ax.axhline(self.loss_tol, color='steelblue', ls='--', lw=1,
                   label=f'tolerance = {self.loss_tol}')
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Contrastive Learning — Loss Curve')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_history_flows(self):
        """Plot flow on each edge over logged history steps."""
        if not self.history:
            print("No history recorded. Run with log_every < max_iter.")
            return

        steps  = [h['step']        for h in self.history]
        flows  = np.array([h['flows'] for h in self.history])

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        for i, (u, v) in enumerate(self.net.edges):
            is_target = (u, v) == self.target_edge
            axes[0].plot(steps, flows[:, i],
                         lw=2.5 if is_target else 1,
                         ls='-'  if is_target else '--',
                         label=f"({u},{v})")
        axes[0].axhline(self.target_flow, color='red', lw=1.5, ls=':',
                        label=f"target={self.target_flow}")
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Flow')
        axes[0].set_title('Edge Flows over Training')
        axes[0].legend(fontsize=9)

        losses = [h['loss'] for h in self.history]
        axes[1].plot(steps, losses, color='tomato')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss (logged steps)')

        plt.tight_layout()
        plt.show()

    def plot_results(self, log_scale=False, label_fontsize=18):
        """Before/after plots of conductances and flows."""
        self.net.plot_before_after(
            values_before=self._K_init,
            values_after=self._K_final,
            G_before=self.net_init.G,
            G_after=self.net.G,
            title='Conductances', label='Conductance',
            cmap=plt.cm.coolwarm, log_scale=log_scale,
            label_fontsize=label_fontsize)

        self.net.plot_before_after(
            values_before=self._state_init['flows'],
            values_after=self._state_final['flows'],
            G_before=self.net_init.G,
            G_after=self.net.G,
            title='Edge Flows', label='Flow',
            cmap=plt.cm.managua.reversed(), log_scale=log_scale,
            label_fontsize=label_fontsize)
        
      # ── Loss + history flows side by side ─────────────────────────────────
        if not self.history:
            self.plot_loss()
            return

        steps = [h['step']  for h in self.history]
        flows = np.array([h['flows'] for h in self.history])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        axes[0].plot(self.losses, color='tomato', lw=1.5)
        axes[0].axhline(self.loss_tol, color='steelblue', ls='--', lw=1,
                        label=f'tolerance = {self.loss_tol}')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Iteration', fontsize=label_fontsize - 4)
        axes[0].set_ylabel('Loss', fontsize=label_fontsize - 4)
        axes[0].set_title('Loss Curve', fontsize=label_fontsize - 2)
        axes[0].legend(fontsize=label_fontsize - 6)
        axes[0].tick_params(labelsize=label_fontsize - 6)

        # Edge flows over history
        for i, (u, v) in enumerate(self.net.edges):
            is_target = (u, v) == self.target_edge
            axes[1].plot(steps, flows[:, i],
                        lw=2.5 if is_target else 1,
                        ls='-'  if is_target else '--',
                        label=f"({u},{v}){' ← target' if is_target else ''}")
        axes[1].axhline(self.target_flow, color='red', lw=1.5, ls=':',
                        label=f"target = {self.target_flow}")
        axes[1].set_xlabel('Iteration', fontsize=label_fontsize - 4)
        axes[1].set_ylabel('Flow', fontsize=label_fontsize - 4)
        axes[1].set_title('Edge Flows over Training', fontsize=label_fontsize - 2)
        axes[1].legend(fontsize=label_fontsize - 6)
        axes[1].tick_params(labelsize=label_fontsize - 6)

        plt.tight_layout()
        plt.show()


