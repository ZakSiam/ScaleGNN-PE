# gnn_ss.py
"""GNN-SS: GNN-powered Sub-sampled Bayesian Optimization.

Wang-Henderson, Soyuer, Kassraie, Krause & Bogunovic, "Graph Neural Bayesian Optimization for
Virtual Screening", NeurIPS 2023 Workshop on Adaptive Experimental Design and Active Learning
in the Real World. Implements Algorithm 1 of that paper.

Kept in its own module so algorithms.py is untouched; GnnSS subclasses GnnUCB from there.
"""

import numpy as np
import torch

from algorithms import GnnUCB
from config import *


class GnnSS(GnnUCB):
    """GNN-SS: GNN-powered Sub-sampled Bayesian Optimization.

    Wang-Henderson, Soyuer, Kassraie, Krause & Bogunovic, "Graph Neural Bayesian Optimization
    for Virtual Screening", NeurIPS 2023 Workshop on Adaptive Experimental Design and Active
    Learning in the Real World. Implements Algorithm 1.

    Two ideas, both aimed at the O(|G|) scan and the O(p) covariance of GnnUCB:

    1. Random sub-sampling (SS). At each adaptive round we draw G_t ~ Unif(G) with |G_t| = K,
       score only those K arms, and query the top-b. This replaces the full-domain scan and,
       per the paper, also diversifies the training set (Fig. 2 there shows it can beat the
       full scan on sample efficiency, not just cost).

    2. Data-independent random projection of the NTK features. With S: R^p -> R^d (d << p),
       g~(G) = S g(G) and

           sigma~_t^2(G) = g~(G)^T [ lambda I + sum_i g~(G_i) g~(G_i)^T ]^{-1} g~(G)   (Eq. 2)

       so the inverse is d x d instead of p x p, and -- crucially here -- only d floats per arm
       need to be cached instead of p. We implement the paper's 'Sub' variant, where S = P is
       the sparse selection map with a single nonzero sqrt(p/d) per row (Weinberger et al. 2009;
       Yang et al. 2015), costing O(d) per arm. The paper's FJL variant is deliberately not
       implemented: per its own Table 2 it costs O(N p log p), which at N = 1e6 is far worse
       than Sub for a reported gain of 14.38 vs 14.27 on their benchmark.

       uncertainty='diag' reproduces the GnnUCB/Kassraie et al. diagonal approximation instead,
       so the two can be compared under an otherwise identical algorithm.

    Sub-sampling composes with the projection: the per-arm gradient cache drops from
    p floats (4.24 MB at neuron_per_layer=1024) to d floats (8 KB at d=2048).
    """

    def __init__(self, net: str, num_nodes: int, feat_dim: int, num_actions: int, action_domain: list,
                 subsample_K: int = 4000, batch_size: int = 1, explore_rounds: int = 0,
                 acquisition: str = 'ucb', uncertainty: str = 'sub', proj_dim: int = 2048, cov_scaling: str = 'alg1',
                 ss_train_batch: int = 32, paper_init: bool = False,
                 **kwargs):
        super().__init__(net=net, num_nodes=num_nodes, feat_dim=feat_dim, num_actions=num_actions,
                         action_domain=action_domain, **kwargs)

        self.name = 'GNN-SS'
        self.ss_train_batch = int(ss_train_batch)

        # Appendix A / C state that theta_0 has "iid. unit Gaussian" entries. That is stated for
        # an NTK-parameterised network, whose forward pass carries a 1/sqrt(m) factor per layer;
        # nets.py uses STANDARD parameterisation (plain nn.Linear, no 1/sqrt(m)), so unit-variance
        # weights blow the output up by ~m^(L/2). Measured on rtcb at m=384, L=3: f(G) ~= -8.9e3
        # against rewards of mean 0, std 1.
        #
        # GnnUCB's normalize_init (Kaiming, fan_in, relu gain) is the standard-parameterisation
        # EQUIVALENT of the paper's init -- same function class and same O(1) output scale, just
        # the other convention. So paper_init defaults to False: that is faithful in effect, and
        # it keeps GNN-SS comparable with the GNN-PE / PEFT runs, which all use normalize_init.
        # Set paper_init=True only if nets.py is also switched to NTK parameterisation.
        self.paper_init = bool(paper_init)
        if self.paper_init:
            import copy as _copy
            with torch.no_grad():
                for p in self.func.parameters():
                    p.normal_(mean=0.0, std=1.0)
            self.f0 = _copy.deepcopy(self.func)
            for p in self.f0.parameters():
                p.requires_grad_(True)
            self.num_net_params = sum(p.numel() for p in self.f0.parameters() if p.requires_grad)
            self.U = self.alg_lambda * torch.ones((self.num_net_params,)).to(device)
            self._grad_cache = {}

        self.subsample_K = int(subsample_K)
        self.batch_size = int(batch_size)
        self.explore_rounds = int(explore_rounds)
        self.acquisition = str(acquisition).lower()
        if self.acquisition not in {'ucb', 'greedy', 'ts'}:
            raise ValueError(f"Unknown acquisition: {self.acquisition}. Use ucb, greedy or ts.")
        self.uncertainty = str(uncertainty).lower()
        if self.uncertainty not in {'sub', 'diag', 'fjl'}:
            raise ValueError(f"Unknown uncertainty: {self.uncertainty}. Use sub, fjl or diag.")
        # The paper is internally inconsistent about the covariance normalisation: Eq. (1)/(2)
        # write  lambda*I + (1/t) sum g g^T,  while Algorithm 1 and Appendix B write
        # lambda*I + sum g g^T.  Expose both rather than silently picking one.
        self.cov_scaling = str(cov_scaling).lower()
        if self.cov_scaling not in {'alg1', 'eq2'}:
            raise ValueError("cov_scaling must be alg1 (no 1/t) or eq2 (with 1/t).")

        # Round counter: drives the exploration stage (Algorithm 1, warm start).
        self._t = 0

        # --- Sub projection S = P: one nonzero of sqrt(p/d) per row, i.e. a scaled selection.
        # S is data-independent and drawn ONCE, so g~ is a fixed feature of each arm (as with g).
        self.proj_dim = int(min(proj_dim, self.num_net_params))
        if self.uncertainty == 'fjl':
            # S_FJL = P H D  (Ailon & Chazelle 2009):  D Rademacher diagonal, H Walsh-Hadamard
            # applied by the O(p log p) fast transform, P a d-of-p subsample scaled by sqrt(p/d).
            p = self.num_net_params
            self._fjl_n = 1 << (p - 1).bit_length()          # zero-pad to a power of two
            sign = self._rds.randint(0, 2, size=self._fjl_n) * 2 - 1
            self._fjl_D = torch.from_numpy(sign).float().to(device)
            self._fjl_idx = torch.from_numpy(
                np.sort(self._rds.choice(self._fjl_n, size=self.proj_dim, replace=False))
            ).long().to(device)
            # H/sqrt(n) is orthogonal, so <Sg,Sg'> is unbiased for <g,g'> with the sqrt(p/d) gain.
            self._fjl_scale = float(np.sqrt(p / self.proj_dim) / np.sqrt(self._fjl_n))
            d = self.proj_dim
            self.Kt = self.alg_lambda * torch.eye(d, device=device, dtype=torch.float64)
            self.Kt_inv = torch.eye(d, device=device, dtype=torch.float64) / self.alg_lambda
            self.M = torch.zeros((d, d), device=device, dtype=torch.float64)
            self._proj_idx = None
            self._proj_scale = 1.0
        elif self.uncertainty == 'sub':
            idx = self._rds.choice(self.num_net_params, size=self.proj_dim, replace=False)
            self._proj_idx = torch.from_numpy(np.sort(idx)).long().to(device)
            self._proj_scale = float(np.sqrt(self.num_net_params / self.proj_dim))
            # K_0 = lambda I  (Algorithm 1), maintained together with its inverse.
            d = self.proj_dim
            self.Kt = self.alg_lambda * torch.eye(d, device=device, dtype=torch.float64)
            self.Kt_inv = torch.eye(d, device=device, dtype=torch.float64) / self.alg_lambda
            self.M = torch.zeros((d, d), device=device, dtype=torch.float64)
        else:
            self._proj_idx = None
            self._proj_scale = 1.0

        # Projected-gradient cache. Same lazy contract as GnnUCB._grad_cache, but each entry is
        # d floats rather than p, which is what makes a 1e6-arm domain fit in memory at all.
        self._proj_cache = {}

    # ---- NTK features -------------------------------------------------------------------

    def _raw_grad(self, idx):
        """g(G)/sqrt(m): the NTK feature at initialization, under GnnUCB's 1/sqrt(m) scaling."""
        return self._get_grad(idx) / np.sqrt(self.neuron_per_layer)

    @staticmethod
    def _fwht(x):
        """In-place-style fast Walsh-Hadamard transform, O(n log n), n a power of two."""
        n = x.numel()
        h = 1
        while h < n:
            x = x.view(-1, 2, h)
            a, b = x[:, 0, :], x[:, 1, :]
            x = torch.stack((a + b, a - b), dim=1)
            h <<= 1
        return x.reshape(n)

    def _fjl_apply(self, g):
        """g~ = P H D g  -- the Fast Johnson-Lindenstrauss variant (paper Appendix A)."""
        v = torch.zeros(self._fjl_n, device=device, dtype=torch.float32)
        v[:g.numel()] = g.float()
        v = self._fwht(v * self._fjl_D)
        return (v[self._fjl_idx] * self._fjl_scale).to(torch.float64)

    def _proj_grad(self, idx, cache: bool = False):
        """g~(G) = S g(G)/sqrt(m).

        `cache=True` only for arms that enter the posterior (Algorithm 1 step 4), which is at
        most b*T of them. Scored-but-unqueried candidates are NOT cached: each adaptive round
        draws a fresh G_t ~ Unif(G), so with K << |G| a redraw is rare and caching every scored
        arm would grow the cache to O(K*T*d) -- 16 GB at K=4000, T=300, d=2048 -- to save
        almost no recompute. Queried arms stay at O(b*T*d), i.e. megabytes.
        """
        g = self._proj_cache.get(idx)
        if g is not None:
            return g
        raw = self._raw_grad(idx)
        if self.uncertainty == 'fjl':
            g = self._fjl_apply(raw)
        else:
            g = (raw[self._proj_idx] * self._proj_scale).to(torch.float64)
        # The full-p gradient was only needed to build the projection; drop it so the retained
        # footprint is O(d) per arm rather than O(p).
        self._grad_cache.pop(idx, None)
        if cache:
            self._proj_cache[idx] = g
        return g

    # ---- posterior ----------------------------------------------------------------------

    def add_data(self, indices, rewards):
        """Append observations and rank-1 update the (projected or diagonal) covariance."""
        for idx, reward in zip(indices, rewards):
            self.data['graph_indices'].append(idx)
            self.data['rewards'].append(reward)

            if self.uncertainty == 'diag':
                g_idx = self._get_grad(idx)
                self.U += g_idx * g_idx / self.neuron_per_layer
            else:
                u = self._proj_grad(idx, cache=True).reshape(-1, 1)   # (d, 1)
                self.M += u @ u.T
                if self.cov_scaling == 'alg1':
                    # K_t = lambda I + sum g g^T  -> exact rank-1 inverse update.
                    self.Kt += u @ u.T
                    # Sherman-Morrison: (A + uu^T)^-1 = A^-1 - (A^-1 u u^T A^-1)/(1 + u^T A^-1 u)
                    Ainv_u = self.Kt_inv @ u
                    denom = 1.0 + float((u.T @ Ainv_u).item())
                    self.Kt_inv -= (Ainv_u @ Ainv_u.T) / denom
                else:
                    # K_t = lambda I + (1/t) sum g g^T. The 1/t rescales every past term, so
                    # this is not a rank-1 update; refactor the d x d inverse instead (once per
                    # observation, negligible against scoring K candidates).
                    t = max(1, len(self.data['graph_indices']))
                    d = self.proj_dim
                    self.Kt = self.alg_lambda * torch.eye(d, device=device, dtype=torch.float64) + self.M / t
                    self.Kt_inv = torch.linalg.inv(self.Kt)

    def get_post_var(self, idx):
        if self.uncertainty == 'diag':
            g = self._get_grad(idx)
            return torch.sqrt(torch.sum(g * g / self.U) / self.neuron_per_layer).item()
        u = self._proj_grad(idx).reshape(-1, 1)
        return float(torch.sqrt(torch.clamp(u.T @ self.Kt_inv @ u, min=0.0)).item())

    # ---- action selection ---------------------------------------------------------------

    def _acq_scores(self, cand):
        """Acquisition value for each candidate index (Algorithm 1, step 2)."""
        beta = np.sqrt(self.exploration_coef)
        scores = []
        for ix in cand:
            mean = self.func(self.action_domain[ix]).item()
            if self.acquisition == 'greedy':
                scores.append(mean)
            else:
                sd = self.get_post_var(ix)
                if self.acquisition == 'ucb':
                    scores.append(mean + beta * sd)
                else:  # thompson
                    scores.append(mean + beta * sd * self._rds.normal())
        return np.asarray(scores, dtype=float)

    def subsample(self):
        """G_t ~ Unif(G), |G_t| = K (Algorithm 1, step 1). Drawn fresh every round."""
        if self.subsample_K >= self.num_actions:
            return np.arange(self.num_actions)
        return self._rds.choice(self.num_actions, size=self.subsample_K, replace=False)

    def select_batch(self):
        """Return the b arms queried this round; b = 1 recovers sequential selection."""
        self._t += 1
        b = min(self.batch_size, self.num_actions)

        # Exploration stage: uniform random queries to warm-start the surrogate.
        if self._t <= self.explore_rounds:
            return list(self._rds.choice(self.num_actions, size=b, replace=False))

        cand = self.subsample()
        scores = self._acq_scores(cand)
        top = np.argsort(-scores)[:b]
        return [int(cand[i]) for i in top]

    def select(self):
        """Single-action interface, for compatibility with the sequential runners."""
        return self.select_batch()[0]

    def best_predicted(self):
        """Highest posterior mean over a fresh sub-sample (the full scan is what SS avoids)."""
        cand = self.subsample()
        means = [self.func(self.action_domain[ix]).item() for ix in cand]
        return int(cand[int(np.argmax(means))])

    # ---- training (paper Appendix A loss + Algorithm 2) ---------------------------------

    def train(self):
        """TrainGNN (Algorithm 2) on the Appendix A loss.

        Overrides GnnUCB.train, which minimises the plain squared error. The paper's objective
        adds a proximal term anchoring the weights to their initialization:

            L(theta; H_t) = (1/t) sum_i ( f_GNN(G_i; theta) - y_i )^2  +  m * lambda * ||theta - theta_0||^2

        The regulariser is not cosmetic here. It is what keeps theta in the lazy/NTK regime near
        theta_0, which is the assumption under which g(G) = grad f(G; theta_0) is a valid
        linearisation and Eq. (2) is a meaningful posterior variance. GNN-SS never resets to
        theta_0 (it warm-starts from theta_{t-1}, unlike Kassraie et al. 2022), so without this
        term the surrogate drifts away from the point the covariance is built at, and the mean
        and uncertainty end up describing different functions.

        Algorithm 2 runs E epochs of Adam over minibatches drawn from H_t; we take E from
        max_train_steps so the runner's existing knob keeps working.
        """
        n = len(self.data['graph_indices'])
        if n == 0:
            return 0.0

        optimizer = self._get_optimizer(reset=self.train_from_scratch)
        theta0 = [p.detach().clone() for p in self.f0.parameters() if p.requires_grad]
        bsz = max(1, min(int(self.ss_train_batch), n))
        E = max(1, int(self.max_train_steps))

        tot = 0.0
        for _ in range(E):
            batch = self._rds.choice(n, size=bsz, replace=False)
            optimizer.zero_grad()

            # (1/t) sum (f(G_i) - y_i)^2  -- mean over the sampled batch
            mse = torch.zeros((), device=device)
            for j in batch:
                pred = self.func(self.action_domain[self.data['graph_indices'][j]]).to(device)
                y = torch.tensor(self.data['rewards'][j], dtype=torch.float32, device=device)
                mse = mse + (pred - y) ** 2
            mse = mse / len(batch)

            # m * lambda * ||theta - theta_0||^2
            prox = torch.zeros((), device=device)
            for p, p0 in zip((q for q in self.func.parameters() if q.requires_grad), theta0):
                prox = prox + ((p - p0) ** 2).sum()
            loss = mse + self.neuron_per_layer * self.alg_lambda * prox

            loss.backward()
            optimizer.step()
            tot += float(loss.item())
        return tot / E
