import unittest

import numpy as np
import torch

from experiments.gnn_mdt.closure import (
    MultiViewSAGE,
    build_graphs,
    consensus_embedding,
    knn_transitions,
    make_synthetic,
    mdt_oos_operator,
    mdt_operator,
    scipy_to_torch,
    shuffle_graphs,
    trajectory,
)


class GNNMDTTests(unittest.TestCase):
    def test_transitions_are_stochastic(self):
        split = make_synthetic(n_train=30, n_test=12)
        train, test = knn_transitions(split.train[0], split.test[0], 5)
        np.testing.assert_allclose(np.asarray(train.sum(1)).ravel(), 1, atol=1e-6)
        np.testing.assert_allclose(np.asarray(test.sum(1)).ravel(), 1, atol=1e-6)
        self.assertEqual(test.shape, (12, 30))

    def test_alpha_normalisation_is_inert_at_zero_and_reweights_density(self):
        split = make_synthetic(n_train=40, n_test=15)
        plain, plain_oos = build_graphs(split, 5, 0.0)
        alpha, alpha_oos = build_graphs(split, 5, 1.0)
        default = build_graphs(split, 5)[0]
        for index, (base, normalised) in enumerate(zip(plain, alpha)):
            # alpha=0 must reproduce the historical operator exactly, so the
            # baseline arm of the alpha sweep is the earlier locked run.
            np.testing.assert_allclose(base.toarray(), default[index].toarray(), atol=1e-6)
            np.testing.assert_allclose(np.asarray(normalised.sum(1)).ravel(), 1, atol=1e-6)
            self.assertEqual(base.nnz, normalised.nnz)
            # Same support, different mass: dividing by the neighbour degree
            # moves weight away from high-density columns.
            self.assertGreater(abs(base - normalised).sum(), 1e-3)
        for base, normalised in zip(plain_oos, alpha_oos):
            np.testing.assert_allclose(np.asarray(normalised.sum(1)).ravel(), 1, atol=1e-6)
            self.assertGreater(abs(base - normalised).sum(), 1e-3)

    def test_mdt_and_oos_shapes(self):
        split = make_synthetic(n_train=30, n_test=12)
        train, test = build_graphs(split, 5)
        weights = trajectory(len(train), 4, 0)
        self.assertEqual(mdt_operator(train, weights).shape, (30, 30))
        self.assertEqual(mdt_oos_operator(train, test, weights).shape, (12, 30))
        ctr, cte = consensus_embedding(train, test, 4, 3, 2, 0)
        self.assertEqual(ctr.shape, (30, 3))
        self.assertEqual(cte.shape, (12, 3))

    def test_shuffled_graphs_are_capacity_matched_nulls(self):
        split = make_synthetic(n_train=30, n_test=12)
        train, test = build_graphs(split, 5)
        s_train, s_test = shuffle_graphs(train, test, 0)
        for real, null in zip(train, s_train):
            # Same stochastic operator up to relabelling: rows still sum to one,
            # the degree multiset and the weight multiset are untouched.
            np.testing.assert_allclose(np.asarray(null.sum(1)).ravel(), 1, atol=1e-6)
            self.assertEqual(real.nnz, null.nnz)
            np.testing.assert_allclose(np.sort(np.diff(real.indptr)),
                                       np.sort(np.diff(null.indptr)))
            np.testing.assert_allclose(np.sort(real.data), np.sort(null.data), atol=1e-6)
            # ...but the neighbours of a given node changed, which is the point.
            self.assertGreater(abs(real - null).sum(), 0)
            # Rows and columns are permuted together, so self-loops stay on the
            # diagonal; a one-sided permutation would scatter them.
            self.assertTrue((null.diagonal() > 0).all())
        for real, null in zip(test, s_test):
            np.testing.assert_allclose(np.asarray(null.sum(1)).ravel(), 1, atol=1e-6)
            self.assertEqual(real.shape, null.shape)
        # Same seed, same permutation: the control is reproducible across runs.
        again, _ = shuffle_graphs(train, test, 0)
        for first, second in zip(s_train, again):
            self.assertEqual((first != second).nnz, 0)

    def test_recycling_starts_as_identity_and_pair_target_is_gauge_free(self):
        split = make_synthetic(n_train=30, n_test=12)
        train, test = build_graphs(split, 5)
        ptr = [scipy_to_torch(p, torch.device("cpu")) for p in train]
        xtr = [torch.tensor(x) for x in split.train]
        model = MultiViewSAGE([x.shape[1] for x in split.train], 16, 3)
        plain, _ = model.forward_train(xtr, ptr)
        recycled, _ = model.forward_train(xtr, ptr, previous=plain)
        # The recycle projection is zero-initialised, so an untrained recycled
        # model must equal the baseline: the comparison starts nested.
        torch.testing.assert_close(plain, recycled)
        torch.nn.init.ones_(model.recycle.weight)
        moved, _ = model.forward_train(xtr, ptr, previous=plain)
        self.assertGreater((moved - plain).abs().sum().item(), 1e-3)
        # A rotation of the prediction leaves the Gram loss untouched but changes
        # the coordinate loss: that gauge freedom is what the pair arm removes.
        target = torch.randn(30, 3)
        rotation = torch.linalg.qr(torch.randn(3, 3))[0]
        gram = lambda z: z @ z.T
        torch.testing.assert_close(gram(plain @ rotation), gram(plain))
        self.assertGreater(
            abs(torch.nn.functional.mse_loss(plain @ rotation, target).item()
                - torch.nn.functional.mse_loss(plain, target).item()), 1e-6)

    def test_inductive_forward_and_fusion_gradient(self):
        split = make_synthetic(n_train=30, n_test=12)
        train, test = build_graphs(split, 5)
        ptr = [scipy_to_torch(p, torch.device("cpu")) for p in train]
        pte = [scipy_to_torch(p, torch.device("cpu")) for p in test]
        xtr = [torch.tensor(x) for x in split.train]
        xte = [torch.tensor(x) for x in split.test]
        model = MultiViewSAGE([x.shape[1] for x in split.train], 16, 3,
                              message_passing=True, learn_fusion=True)
        ztr, _ = model.forward_train(xtr, ptr)
        zte = model.forward_test(xtr, xte, ptr, pte)
        self.assertEqual(tuple(ztr.shape), (30, 3))
        self.assertEqual(tuple(zte.shape), (12, 3))
        ztr.square().mean().backward()
        self.assertIsNotNone(model.fusion_logits.grad)
        self.assertTrue(torch.isfinite(model.fusion_logits.grad).all())


if __name__ == "__main__":
    unittest.main()
