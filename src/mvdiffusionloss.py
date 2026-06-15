import numpy as np
import tensorflow as tf


class MVDiffusionLoss(tf.keras.losses.Loss):
    """
    Out-of-sample loss for Multi-view Diffusion Trajectories (MDT, arXiv:2512.01484).

    Given a precomputed MDT operator W = W_t...W_1 (n x n, row-stochastic and
    generally asymmetric), the trajectory-dependent diffusion map of Prop. 2 is
    obtained from the pi_t-symmetrized operator

        A = Pi^{1/2} W Pi^{-1/2},   Pi = diag(pi_t),   pi_t^T W = pi_t^T

    whose top singular vector is exactly sqrt(pi_t). The encoder embedding Psi
    is matched (up to rotation) by requiring the Gram matrix of its sqrt(pi_t)-
    scaled outputs to equal

        G = A A^T - sqrt(pi) sqrt(pi)^T

    This mirrors `DiffusionLoss` (single-view: A symmetric, target A^{2t} -
    sqrt(pi)sqrt(pi)^T). The pi_t weighting is essential, not cosmetic: a
    collapsed (constant) embedding maps onto exactly the removed sqrt(pi)sqrt(pi)^T
    component, so collapse is penalised -- without it the parametric encoder
    collapses to a constant on multi-cluster data.

    pi_t is the stationary distribution of the fixed operator W, found once by
    power iteration (cheap; W is fixed for the out-of-sample stage).
    """

    def __init__(self, W: np.ndarray, name: str = "mv_diffusion_loss"):
        if not isinstance(W, np.ndarray) or W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be a square (n, n) NumPy array.")
        super().__init__(name=name)

        pi = self._stationary_distribution(W)
        sqrt_pi = np.sqrt(np.maximum(pi, 1e-12))
        # Pi^{1/2} W Pi^{-1/2}: symmetrised operator whose top singular mode is sqrt(pi).
        A = sqrt_pi[:, None] * W / sqrt_pi[None, :]
        G = A @ A.T - np.outer(sqrt_pi, sqrt_pi)

        self.G = tf.constant(G, dtype=tf.float32)
        self.sqrt_pi = tf.constant(sqrt_pi, dtype=tf.float32)

    @staticmethod
    def _stationary_distribution(W: np.ndarray, iters: int = 500) -> np.ndarray:
        # Left Perron eigenvector of the row-stochastic operator: pi^T W = pi^T.
        p = np.ones(W.shape[0]) / W.shape[0]
        for _ in range(iters):
            p = p @ W
            p /= p.sum()
        return p

    def call(self, y_true, y_pred):
        # y_true: integer indices of the batch points; y_pred: encoder embeddings.
        ids = tf.cast(y_true, tf.int32)
        G = tf.gather(tf.gather(self.G, ids), ids, axis=1)
        scaled = tf.expand_dims(tf.gather(self.sqrt_pi, ids), axis=1) * y_pred
        gram = tf.matmul(scaled, scaled, transpose_b=True)
        return tf.reduce_mean(tf.square(gram - G))
