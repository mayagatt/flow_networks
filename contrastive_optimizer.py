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
    clamp_strength: float      — scales how strongly Q_in is nudged in clamped phase
    loss_tol     : float       — stop when loss < loss_tol

    Example
    -------
    >>> opt = ContrastiveOptimizer(
    ...     network=net,
    ...     Q_in=np.array([1., 0., 0., -1.]),
    ...     target_edge=(1, 2),
    ...     target_flow=0.3,
    ...     learning_rate=0.01,
    ...     clamp_strength=0.01,
    ...     loss_tol=1e-4,
    ... )
    >>> opt.run(max_iter=100_000, log_every=1000)
    >>> opt.plot_loss()
    >>> opt.plot_results()
    """

    def __init__(self, network: FlowNetwork, Q_in: np.ndarray,
                 target_edge: tuple, desired_flow_target: float,
                 update_func: str,  # 'PD' = power dissipation difference, 'SR' = shear rate difference, 'SR_global_shear' = global shear rate difference
                 learning_rate: float = 0.01,
                 clamp_strength: float = 0.01,
                 loss_tol: float = 1e-4):

        self.net_init = network                     # original (never mutated)
        self.net = copy.deepcopy(network)           # working copy
        self.Q_in = Q_in.astype(float)
        self.target_edge = target_edge
        self.desired_flow_target = desired_flow_target
        self.lr = learning_rate
        self.eps = clamp_strength
        self.loss_tol = loss_tol
        self.update_func = update_func

        self.tau_0 = self.net.get_K()**(-3/4) * self.net.solve_q_p(self.Q_in)['flows']

        self.target_idx = self.net.edge_index(target_edge)
        self.hist_scalars = {
            'step': [],
            'loss': [],
            'flow_target': []
        }
        
        self.hist_arrays = {
            'conductances': [],
            'flows':        [],
            'pressures':    [],
            'shear_rates':  []
        }

    # ── Loss ──────────────────────────────────────────────────────────────────

    def get_flows(self):
        return self.net.solve_q_p(self.Q_in)['flows']

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
        dk_ij = lr * k_ij^(1/4) * (q_F_ij - q_C_ij)
        Increase conductance where free phase has more flow → drives flow
        toward target.
        """
        k = np.diag(self.net.K)
        # dk = k**(1/4) * (q_F - q_C)  # when we take tau in the update rule

        dk = k**(-1/2) * (q_F**2 - q_C**2)    # when taking delta_k ~ k tau^2
        return self.lr * dk
    
    def _conductance_update_SR_global_shear(self, q_F: np.ndarray, q_C: np.ndarray) -> np.ndarray:
        k = np.diag(self.net.K)
        # tau = k**(1/4) * np.abs(q_F) # delta_k ~ k tau
        
        tau_0 = self.tau_0
        return self.lr * (k**(-1/2) * q_F**2 - k * tau_0**2)

    # ── Single step ───────────────────────────────────────────────────────────

    def step(self, p_F: np.ndarray,q_F: np.ndarray, loss: float) -> tuple:
        """Run one free→clamp→update→rebuild cycle. Returns updated p_F, loss."""
        net = self.net

        # The contrastive learning way: add target edge flow to the source/sink vector, according to the loss
        Qin_C   = self.Q_in + self.eps * loss * net.B[:, self.target_idx]

        # resolve the state of the network in its clamped state
        state_C = net.solve_q_p(Qin_C)

        # ── Conductance update ────────────────────────────────────────────────
        # compute dk for each edge based on the difference between the free and clamped phases
        if self.update_func == 'PD':
            dk = self._conductance_update_PD(p_F=p_F, p_C=state_C['pressures'])
        elif self.update_func == 'SR':
            dk = self._conductance_update_SR(q_F=q_F, q_C=state_C['flows'])
        elif self.update_func == 'SR_global_shear':
            dk = self._conductance_update_SR_global_shear(q_F=q_F, q_C=state_C['flows'])
        else:
            raise ValueError(f"Invalid update_func: {self.update_func}. Choose 'PD' or 'SR'.")

        # apply dk to the edge weights, ensuring they remain positive
        k = net.get_K()
        new_k = np.maximum(k + dk, 1e-5)
        net.set_K(new_k)  # might be a bug - not sure if the network is saved

        # ── Recompute free phase with updated network ───────────────────────────────────
        state_F_new = net.solve_q_p(self.Q_in)
        q_F_new = state_F_new['flows']
        loss_new = self.loss(q_F_new[self.target_idx])

        return state_F_new, loss_new

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, max_iter: int = 100_000, log_every: int = 1000):
        """
        Run the contrastive optimization loop.

        Parameters
        ----------
        max_iter  : maximum number of iterations
        log_every : print and record history every N steps
        """
        n_log = max_iter // log_every
        arr = {
            'conductances': np.empty((n_log, len(self.net.edges))),
            'flows':        np.empty((n_log, len(self.net.edges))),
            'pressures':    np.empty((n_log, len(self.net.nodes))),
            'shear_rates':  np.empty((n_log, len(self.net.edges)))
        }
        t = 0

        # Initial free phase
        state_F = self.net.solve_q_p(self.Q_in)
        p_F     = state_F['pressures']
        q_F     = state_F['flows']
        loss    = self.loss(q_F[self.target_idx])

        print(f"Initial flow on {self.target_edge}: "
            f"{q_F[self.target_idx]:.4f}  |  "
            f"target: {self.desired_flow_target}  |  loss: {loss:.6f}")

        self._state_init = state_F
        self._K_init     = np.diag(self.net_init.K).copy()

        print("\n=== Optimization start ===")
        try:
            for ii in range(1, max_iter + 1):
                state_F, loss = self.step(p_F, q_F, loss)
                p_F = state_F['pressures']
                q_F = state_F['flows']

                if ii % log_every == 0:
                    flow = state_F['flows'][self.target_idx]
                    print(f"Iter {ii:>7}  |  flow: {flow:.5f}  |  loss: {loss:.6f}")

                    self.hist_scalars['step'].append(ii)
                    self.hist_scalars['loss'].append(loss)
                    self.hist_scalars['flow_target'].append(flow)

                    arr['conductances'][t] = np.diag(self.net.K)
                    arr['flows'][t]        = state_F['flows']
                    arr['pressures'][t]    = state_F['pressures']
                    arr['shear_rates'][t]  = self.net.get_K()**(-3/4) * state_F['flows']
                    t += 1

                if loss < self.loss_tol:
                    print(f"\n✓ Converged at iteration {ii}  |  loss: {loss:.2e}")
                    break
            else:
                print(f"\n✗ Did not converge in {max_iter} iterations  |  "
                    f"final loss: {loss:.6f}")
        except KeyboardInterrupt:
            print(f"\n⚠ Interrupted manually at iteration {ii}  |  final loss: {loss:.6f}")

        # This now ALWAYS runs, whether converged, maxed out, or manually interrupted
        self.hist_scalars['step']        = np.array(self.hist_scalars['step'])
        self.hist_scalars['loss']        = np.array(self.hist_scalars['loss'])
        self.hist_scalars['flow_target'] = np.array(self.hist_scalars['flow_target'])

        self.hist_arrays = {k: v[:t] for k, v in arr.items()}

        self._state_final = state_F
        self._K_final     = np.diag(self.net.K).copy()

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_loss(self):
        """Plot loss curve over all iterations."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.hist_scalars['loss'], color='tomato', lw=1.5)
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
        if not self.hist_scalars['step']:
            print("No history recorded. Run with log_every < max_iter.")
            return

        steps  = self.hist_scalars['step']
        flows  = self.hist_arrays['flows']

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

        losses = self.hist_scalars['loss']
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
            values_after=self.hist_arrays['conductances'][-1],
            G_before=self.net_init.G,
            G_after=self.net.G,
            title='Conductances', label='Conductance',
            cmap=plt.cm.coolwarm, log_scale=log_scale,
            label_fontsize=label_fontsize)

        self.net.plot_before_after(
            values_before=self.hist_arrays['flows'][0],
            values_after=self.hist_arrays['flows'][-1],
            G_before=self.net_init.G,
            G_after=self.net.G,
            title='Edge Flows', label='Flow',
            cmap=plt.cm.managua.reversed(), log_scale=log_scale,
            label_fontsize=label_fontsize)
        
# change plot history to include also the shear rates and pressures over time, in addition to the flows and loss. This will give a more complete picture of how the network evolves during optimization.

# change the legends so that onlt the target edge is highlighed, and the others are shown in the same color (gray?)

    def plot_history(self, title, label_fontsize=18):
        """Plot loss, flows, and conductances over logged history steps."""
      # ── Loss + history flows side by side ─────────────────────────────────
        # if self.hist_scalars['step'].size == 0:
        #     self.plot_loss()
        #     return

        steps = self.hist_scalars['step']
        flows = self.hist_arrays['flows']
        shear_rates = self.hist_arrays['shear_rates']

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        # set tiel for the whole panel
        fig.suptitle(title, fontsize=label_fontsize)

        # Loss curve
        axes[0][0].plot(self.hist_scalars['loss'], color='tomato', lw=1.5)
        axes[0][0].axhline(self.loss_tol, color='steelblue', ls='--', lw=1,
                           label=f'tolerance = {self.loss_tol}')
        axes[0][0].set_yscale('log')
        axes[0][0].set_xlabel('Iteration', fontsize=label_fontsize - 4)
        axes[0][0].set_ylabel('Loss', fontsize=label_fontsize - 4)
        axes[0][0].legend(fontsize=label_fontsize - 6)
        axes[0][0].tick_params(labelsize=label_fontsize - 6)

        # Edge flows over history
        for i, (u, v) in enumerate(self.net.edges):
            is_target = (u, v) == self.target_edge
            axes[0][1].plot(steps, flows[:, i],
                        lw=2.5 if is_target else 1,
                        ls='-'  if is_target else '--',
                        label=f"({u},{v}) target" if is_target else '',
                        color='tomato' if is_target else 'gray')
            
        axes[0][1].axhline(self.desired_flow_target, color='k', lw=1.5, ls=':',
                        label=f"desired flow = {self.desired_flow_target}")
        axes[0][1].set_xlabel('Iteration', fontsize=label_fontsize - 4)
        axes[0][1].set_ylabel('Flow', fontsize=label_fontsize - 4)
        axes[0][1].legend(fontsize=label_fontsize - 6)
        axes[0][1].tick_params(labelsize=label_fontsize - 6)

        # plot conductance history
        conductances = self.hist_arrays['conductances']
        for i, (u, v) in enumerate(self.net.edges):
            is_target = (u, v) == self.target_edge
            axes[1][0].plot(steps, conductances[:, i], 
                        lw=1.5 if is_target else 1,
                        ls='-'  if is_target else '--', 
                        label=f"({u},{v}) target" if is_target else '',
                        color='tomato' if is_target else 'gray')
            
        axes[1][0].set_xlabel('Iteration', fontsize=label_fontsize - 4)
        axes[1][0].set_ylabel('Conductance', fontsize=label_fontsize - 4)
        axes[1][0].legend(fontsize=label_fontsize - 6)
        axes[1][0].tick_params(labelsize=label_fontsize - 6)


        # shear rate history
        for i, (u, v) in enumerate(self.net.edges):
            is_target = (u, v) == self.target_edge
            axes[1][1].plot(steps, shear_rates[:, i], 
                        lw=1.5 if is_target else 1,
                        ls='-'  if is_target else '--', 
                        label=f"({u},{v}) target" if is_target else '',
                        color='tomato' if is_target else 'gray')
            
        axes[1][1].set_xlabel('Iteration', fontsize=label_fontsize - 4)
        axes[1][1].set_ylabel('Shear Rate', fontsize=label_fontsize - 4)
        axes[1][1].legend(fontsize=label_fontsize - 6)
        axes[1][1].tick_params(labelsize=label_fontsize - 6)

        plt.tight_layout()
        plt.show()


