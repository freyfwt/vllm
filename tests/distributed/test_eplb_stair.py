# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import math
import unittest
from dataclasses import FrozenInstanceError, asdict
from statistics import NormalDist

import numpy as np

from vllm.distributed.eplb.policy.stair import (
    StairConfig,
    StairPlan,
    StairPlanner,
)


class TestStairPlanner(unittest.TestCase):
    def test_full_window_rejects_mean_and_p95_regressions(self):
        alternating = np.tile([[10, 1, 8, 1], [1, 10, 2, 7]], (64, 1))
        constant = np.tile([11, 11, 10, 8], (128, 1))
        tail_regression = constant.copy()
        # These bursts improve the binned candidate mean but regress its p95.
        tail_regression[:8] = alternating[:8] * 2
        samples = np.stack((alternating, tail_regression, constant), axis=1)
        current = np.array([[[0, 1], [2, 3]]] * 3)

        plan = StairPlanner.plan_rebalance(
            samples,
            current,
            np.full(3, np.nan),
            np.zeros(2, dtype=np.int64),
            StairConfig(load_window_bins=64),
        )

        np.testing.assert_array_equal(plan.rank_expert_ids[:2], current[:2])
        np.testing.assert_array_equal(plan.source_rank_ids[:2], [[[0, 0], [1, 1]]] * 2)
        np.testing.assert_array_equal(plan.source_slot_ids[:2], [[[0, 1], [0, 1]]] * 2)
        self.assertTrue(np.isnan(plan.predicted_mean_ratios[:2]).all())
        np.testing.assert_array_equal(plan.rank_expert_ids[2], [[0, 2], [1, 3]])
        np.testing.assert_allclose(
            plan.predicted_mean_ratios[2], 1.05, rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            plan.imbalance_ratios,
            [
                [1.10, 1.10, 1.10, 1.10],
                [1.10, 1.10, 1.10, 1.10],
                [1.10, 1.10, 1.05, 1.05],
            ],
            rtol=0,
            atol=1e-15,
        )

    def test_prebinned_sums_use_frequency_weighted_means(self):
        plan = StairPlanner.plan_rebalance(
            np.array([[[2.0, 4.0]], [[9.0, 3.0]]]),
            np.array([[[0], [1]]]),
            np.array([np.nan]),
            np.array([0, 0]),
            StairConfig(),
            sample_counts=np.array([2, 3]),
        )

        np.testing.assert_allclose(
            plan.imbalance_ratios,
            [[43 / 30, 1.5, 43 / 30, 1.5]],
            rtol=0,
            atol=1e-15,
        )

    def test_compression_preserves_all_steps_as_weighted_bins(self):
        samples = np.arange(20).reshape(5, 2, 2)

        compressed, weights = StairPlanner.compress_load_window(samples, 2)

        np.testing.assert_array_equal(weights, [2, 3])
        np.testing.assert_allclose(compressed[0], samples[:2].mean(axis=0))
        np.testing.assert_allclose(compressed[1], samples[2:].mean(axis=0))
        np.testing.assert_allclose(
            np.average(compressed, axis=0, weights=weights), samples.mean(axis=0)
        )

    def test_weighted_moments_use_covariance(self):
        samples = np.array([[1.0, 4.0], [3.0, 2.0]])
        weights = np.array([2, 1])
        expanded = np.repeat(samples, weights, axis=0)

        mean, variance, covariance = StairPlanner.weighted_moments(samples, weights)

        np.testing.assert_allclose(mean, expanded.mean(axis=0))
        np.testing.assert_allclose(variance, expanded.var(axis=0, ddof=1))
        np.testing.assert_allclose(covariance, np.cov(expanded, rowvar=False))
        self.assertFalse(np.shares_memory(variance, covariance))

    def test_single_sample_has_zero_covariance(self):
        mean, variance, covariance = StairPlanner.weighted_moments(
            np.array([[2.0, 3.0]]), np.array([1])
        )

        np.testing.assert_array_equal(mean, [2.0, 3.0])
        np.testing.assert_array_equal(variance, np.zeros(2))
        np.testing.assert_array_equal(covariance, np.zeros((2, 2)))

    def test_score_uses_mean_and_weighted_nearest_rank_p95(self):
        samples = np.array([[8.0, 0.0], [4.0, 4.0]])
        weights = np.array([1, 19])

        imbalance = StairPlanner.placement_imbalance(
            samples, weights, np.array([[0], [1]])
        )

        self.assertEqual(imbalance.mean_ratio, 1.05)
        self.assertEqual(imbalance.p95_ratio, 1.0)

    def test_replica_counts_reject_invalid_placements(self):
        for placement in (np.array([[0, 0], [1, 2]]), np.array([[0], [1]])):
            with self.subTest(placement=placement), self.assertRaises(ValueError):
                StairPlanner.placement_replica_counts(placement, 3)

    def test_expert_risk_uses_mean_and_variance(self):
        risk = StairPlanner.expert_risk(np.array([1.0, 2.0]), np.array([4.0, 0.0]), 0.5)

        np.testing.assert_array_equal(risk, [2.0, 2.0])

    def test_layer_gate_skips_zero_load(self):
        result = StairPlanner.gated_layer_imbalance(
            np.zeros((1, 2)),
            np.ones(1, dtype=np.int64),
            np.array([[0], [1]]),
            None,
            StairConfig(),
        )

        self.assertIsNone(result)

    def test_layer_gate_accepts_first_nonzero_window(self):
        for anchor in (None, np.nan):
            with self.subTest(anchor=anchor):
                imbalance = StairPlanner.gated_layer_imbalance(
                    np.array([[11.0, 9.0]]),
                    np.ones(1, dtype=np.int64),
                    np.array([[0], [1]]),
                    anchor,
                    StairConfig(),
                )

                self.assertIsNotNone(imbalance)
                self.assertEqual(imbalance.mean_ratio, 1.1)

    def test_layer_gate_accepts_relative_deterioration(self):
        imbalance = StairPlanner.gated_layer_imbalance(
            np.array([[3.0, 2.0]]),
            np.ones(1, dtype=np.int64),
            np.array([[0], [1]]),
            1.1,
            StairConfig(absolute_balance_threshold=0.5),
        )

        self.assertIsNotNone(imbalance)
        self.assertEqual(imbalance.mean_ratio, 1.2)

    def test_layer_gate_accepts_absolute_imbalance(self):
        imbalance = StairPlanner.gated_layer_imbalance(
            np.array([[3.0, 2.0]]),
            np.ones(1, dtype=np.int64),
            np.array([[0], [1]]),
            1.3,
            StairConfig(),
        )

        self.assertIsNotNone(imbalance)
        self.assertEqual(imbalance.mean_ratio, 1.2)

    def test_layer_gate_rejects_stable_balanced_layer(self):
        result = StairPlanner.gated_layer_imbalance(
            np.array([[11.0, 9.0]]),
            np.ones(1, dtype=np.int64),
            np.array([[0], [1]]),
            1.1,
            StairConfig(),
        )

        self.assertIsNone(result)

    def test_replica_search_is_bounded_and_deterministic(self):
        kwargs = dict(
            num_stages=3,
            budget_radius=2,
            beam_size=4,
            candidate_score=lambda value: float(np.square(value - 2).sum()),
        )

        first = StairPlanner.replica_candidates(
            np.array([8.0, 4.0, 2.0]), 6, 3, **kwargs
        )
        second = StairPlanner.replica_candidates(
            np.array([8.0, 4.0, 2.0]), 6, 3, **kwargs
        )

        self.assertTrue(first)
        self.assertLessEqual(len(first), 4)
        self.assertEqual(
            [item.tolist() for item in first], [item.tolist() for item in second]
        )
        self.assertEqual(len({tuple(item) for item in first}), len(first))
        self.assertTrue(
            all(item.sum() == 6 and np.all((item >= 1) & (item <= 3)) for item in first)
        )

    def test_replica_search_supports_zero_redundancy(self):
        candidates = StairPlanner.replica_candidates(
            np.array([8.0, 4.0, 2.0]),
            3,
            3,
            num_stages=3,
            budget_radius=2,
            beam_size=4,
            candidate_score=lambda value: float(value.sum()),
        )

        self.assertEqual(len(candidates), 1)
        np.testing.assert_array_equal(candidates[0], [1, 1, 1])

    def test_replica_search_caps_each_expert_at_one_copy_per_rank(self):
        candidates = StairPlanner.replica_candidates(
            np.array([8.0, 3.0]),
            6,
            3,
            num_stages=1,
            budget_radius=0,
            beam_size=1,
            candidate_score=lambda value: float(value.max()),
        )

        np.testing.assert_array_equal(candidates[0], [3, 3])

    def test_replica_search_covers_small_valid_topologies(self):
        for num_experts in range(1, 6):
            for num_ranks in range(1, 4):
                for total_slots in range(
                    num_ranks, num_experts * num_ranks + 1, num_ranks
                ):
                    if total_slots < num_experts:
                        continue
                    with self.subTest(
                        num_experts=num_experts,
                        num_ranks=num_ranks,
                        total_slots=total_slots,
                    ):
                        candidates = StairPlanner.replica_candidates(
                            np.arange(num_experts, 0, -1),
                            total_slots,
                            num_ranks,
                            num_stages=4,
                            budget_radius=2,
                            beam_size=8,
                            candidate_score=lambda value: float(np.square(value).sum()),
                        )
                        self.assertTrue(candidates)
                        self.assertTrue(
                            all(
                                candidate.sum() == total_slots
                                and np.all((candidate >= 1) & (candidate <= num_ranks))
                                for candidate in candidates
                            )
                        )

    def test_replica_search_scores_only_final_candidates(self):
        calls: list[tuple[object, ...]] = []

        def record_score(value: np.ndarray) -> float:
            calls.append(tuple(value))
            return float(np.square(value).sum())

        candidates = StairPlanner.replica_candidates(
            np.arange(8.0, 0.0, -1.0),
            16,
            4,
            num_stages=4,
            budget_radius=4,
            beam_size=8,
            candidate_score=record_score,
        )

        self.assertEqual(len(calls), len(candidates))
        self.assertLessEqual(len(calls), 8)

    def test_replica_search_rejects_invalid_topology_and_controls(self):
        kwargs = dict(
            num_stages=2,
            budget_radius=1,
            beam_size=4,
            candidate_score=lambda value: float(value.sum()),
        )
        with self.assertRaises(ValueError):
            StairPlanner.replica_candidates(np.ones(3), 4, 3, **kwargs)
        with self.assertRaises(ValueError):
            StairPlanner.replica_candidates(
                np.ones(3), 6, 3, **(kwargs | {"budget_radius": 1.0})
            )

    def test_incremental_replica_candidates_start_from_current_counts(self):
        current = np.array([[0, 1], [0, 2]])
        candidates = StairPlanner.incremental_replica_candidates(
            np.array([1.0, 8.0, 2.0]),
            current,
            num_ranks=2,
            max_replica_changes=1,
            num_stages=3,
            budget_radius=2,
            beam_size=8,
        )

        current_counts = np.array([2, 1, 1])
        self.assertTrue(
            any(np.array_equal(candidate, current_counts) for candidate in candidates)
        )
        self.assertTrue(
            all(
                np.abs(candidate - current_counts).sum() <= 2
                for candidate in candidates
            )
        )

    def test_incremental_replica_candidates_keep_zero_redundancy_counts(self):
        current = np.array([[0, 1], [2, 3]])
        candidates = StairPlanner.incremental_replica_candidates(
            np.array([8.0, 4.0, 2.0, 1.0]),
            current,
            num_ranks=2,
            max_replica_changes=2,
            num_stages=4,
            budget_radius=2,
            beam_size=8,
        )

        self.assertEqual(len(candidates), 1)
        np.testing.assert_array_equal(candidates[0], np.ones(4, dtype=np.int64))

    def test_lpt_placement_co_locates_negatively_correlated_experts(self):
        means = np.zeros(4)
        variances = np.ones(4)
        covariance = np.eye(4)
        covariance[0, 1] = covariance[1, 0] = -0.9

        placement = StairPlanner.lpt_placement(
            means,
            variances,
            covariance,
            np.ones(4, dtype=np.int64),
            num_ranks=2,
            z_score=1.0,
            current_rank_expert_ids=np.array([[0, 1], [2, 3]]),
            rank_node_ids=np.zeros(2, dtype=np.int64),
            rank_transfer_limit=1,
            cross_node_transfer_limit=1,
            backtrack_limit=0,
        )

        np.testing.assert_array_equal(placement.rank_expert_ids, [[0, 1], [2, 3]])

    def test_lpt_placement_is_deterministic_and_preserves_replica_constraints(self):
        kwargs = dict(
            expert_means=np.array([8.0, 3.0, 1.0]),
            expert_variances=np.array([2.0, 1.0, 0.5]),
            expert_covariance=np.diag([2.0, 1.0, 0.5]),
            replica_counts=np.array([2, 1, 1]),
            num_ranks=2,
            z_score=0.5,
            current_rank_expert_ids=np.array([[0, 1], [0, 2]]),
            rank_node_ids=np.zeros(2, dtype=np.int64),
            rank_transfer_limit=1,
            cross_node_transfer_limit=1,
            backtrack_limit=0,
        )

        first = StairPlanner.lpt_placement(**kwargs)
        second = StairPlanner.lpt_placement(**kwargs)

        np.testing.assert_array_equal(first.rank_expert_ids, second.rank_expert_ids)
        np.testing.assert_array_equal(first.source_rank_ids, second.source_rank_ids)
        np.testing.assert_array_equal(first.source_slot_ids, second.source_slot_ids)
        self.assertTrue(
            all(len(set(rank)) == len(rank) for rank in first.rank_expert_ids.tolist())
        )
        np.testing.assert_array_equal(
            StairPlanner.placement_replica_counts(first.rank_expert_ids, 3), [2, 1, 1]
        )

    def test_lpt_placement_rejects_invalid_covariance_shape(self):
        with self.assertRaises(ValueError):
            StairPlanner.lpt_placement(
                np.ones(2),
                np.ones(2),
                np.ones((2, 3)),
                np.ones(2, dtype=np.int64),
                2,
                0.5,
                current_rank_expert_ids=np.array([[0], [1]]),
                rank_node_ids=np.zeros(2, dtype=np.int64),
                rank_transfer_limit=1,
                cross_node_transfer_limit=1,
                backtrack_limit=0,
            )

    def test_lpt_placement_rejects_uneven_rank_capacity(self):
        with self.assertRaises(ValueError):
            StairPlanner.lpt_placement(
                np.ones(2),
                np.ones(2),
                np.eye(2),
                np.array([1, 2]),
                2,
                0.5,
                current_rank_expert_ids=np.array([[0], [1]]),
                rank_node_ids=np.zeros(2, dtype=np.int64),
                rank_transfer_limit=1,
                cross_node_transfer_limit=1,
                backtrack_limit=0,
            )

    def test_lpt_placement_rejects_invalid_rank_nodes(self):
        with self.assertRaisesRegex(ValueError, "rank_node_ids"):
            StairPlanner.lpt_placement(
                np.ones(2),
                np.ones(2),
                np.eye(2),
                np.ones(2, dtype=np.int64),
                2,
                0.5,
                current_rank_expert_ids=np.array([[0], [1]]),
                rank_node_ids=np.array([0]),
                rank_transfer_limit=1,
                cross_node_transfer_limit=1,
                backtrack_limit=0,
            )

    def test_lpt_placement_aligns_slots_with_topology_aware_sources(self):
        current = np.array([[2, 0], [1, 0], [3, 1]])
        placement = StairPlanner.lpt_placement(
            np.zeros(4),
            np.zeros(4),
            np.zeros((4, 4)),
            np.array([2, 2, 1, 1]),
            num_ranks=3,
            z_score=0.0,
            current_rank_expert_ids=current,
            rank_node_ids=np.array([0, 1, 0]),
            rank_transfer_limit=1,
            cross_node_transfer_limit=1,
            backtrack_limit=0,
        )

        self.assertIsNotNone(placement)
        np.testing.assert_array_equal(
            placement.rank_expert_ids, [[1, 0], [1, 0], [3, 2]]
        )
        np.testing.assert_array_equal(
            placement.source_rank_ids, [[2, 0], [1, 1], [2, 0]]
        )
        np.testing.assert_array_equal(
            placement.source_slot_ids, [[1, 1], [0, 1], [0, 0]]
        )

    def test_lpt_placement_backtracks_from_greedy_dead_end(self):
        means = np.array([100.0, 6.0, 5.0, 4.0, 3.0, 2.0, 3.0])
        current = np.array([[0, 1, 6], [2, 3, 6], [4, 5, 6]])
        kwargs = dict(
            expert_means=means,
            expert_variances=np.zeros(7),
            expert_covariance=np.zeros((7, 7)),
            replica_counts=np.array([1, 1, 1, 1, 1, 1, 3]),
            num_ranks=3,
            z_score=0.0,
            current_rank_expert_ids=current,
            rank_node_ids=np.zeros(3, dtype=np.int64),
            rank_transfer_limit=1,
            cross_node_transfer_limit=1,
        )

        # Greedy choices fill one rank too early, leaving no three distinct
        # ranks for the final expert even though a valid placement exists.
        self.assertIsNone(StairPlanner.lpt_placement(**kwargs, backtrack_limit=1))
        placement = StairPlanner.lpt_placement(**kwargs, backtrack_limit=2)

        self.assertIsNotNone(placement)
        for dst_rank, target_experts in enumerate(placement.rank_expert_ids):
            for slot, expert in enumerate(target_experts):
                self.assertIn(
                    expert, current[placement.source_rank_ids[dst_rank, slot]]
                )
            for src_rank in range(3):
                if src_rank != dst_rank:
                    self.assertLessEqual(
                        np.sum(placement.source_rank_ids[dst_rank] == src_rank), 1
                    )

    def test_lpt_placement_accepts_valid_covariance_with_strong_cancellation(self):
        deviations = np.array([-0.04, -0.01, 0.03, -0.15, 0.20, 43.74, 0.18, -43.95])
        center = np.full(8, 44.95)
        samples = np.stack([center + deviations, center - deviations])
        means, variances, covariance = StairPlanner.weighted_moments(
            samples, np.ones(2, dtype=np.int64)
        )

        placement = StairPlanner.lpt_placement(
            means,
            variances,
            covariance,
            np.ones(8, dtype=np.int64),
            num_ranks=1,
            z_score=1.0,
            current_rank_expert_ids=np.arange(8).reshape(1, 8),
            rank_node_ids=np.zeros(1, dtype=np.int64),
            rank_transfer_limit=1,
            cross_node_transfer_limit=1,
            backtrack_limit=0,
        )

        self.assertIsNotNone(placement)

    def test_lpt_placement_rejects_small_indefinite_covariance(self):
        with self.assertRaises(ValueError):
            StairPlanner.lpt_placement(
                np.zeros(2),
                np.full(2, 1e-30),
                np.array([[1e-30, -1e-15], [-1e-15, 1e-30]]),
                np.ones(2, dtype=np.int64),
                num_ranks=1,
                z_score=1.0,
                current_rank_expert_ids=np.array([[0, 1]]),
                rank_node_ids=np.zeros(1, dtype=np.int64),
                rank_transfer_limit=1,
                cross_node_transfer_limit=1,
                backtrack_limit=0,
            )

    def test_plan_layer_accepts_mean_improvement(self):
        samples = np.array([[8.0, 7.0, 6.0, 5.0]])
        current = np.array([[0, 1], [2, 3]])

        plan = StairPlanner.plan_layer(
            samples,
            np.ones(1, dtype=np.int64),
            current,
            np.array([0, 1]),
            StairConfig(),
        )

        self.assertIsNotNone(plan)
        np.testing.assert_array_equal(plan.placement.rank_expert_ids, [[0, 3], [2, 1]])
        self.assertEqual(plan.predicted_imbalance.mean_ratio, 1.0)

    def test_plan_layer_can_disable_cross_node_improvement(self):
        plan = StairPlanner.plan_layer(
            np.array([[8.0, 7.0, 6.0, 5.0]]),
            np.ones(1, dtype=np.int64),
            np.array([[0, 1], [2, 3]]),
            np.array([0, 1]),
            StairConfig(cross_node_transfer_limit=0),
        )

        self.assertIsNone(plan)

    def test_plan_layer_skips_noop(self):
        self.assertIsNone(
            StairPlanner.plan_layer(
                np.array([[8.0, 7.0, 6.0, 5.0]]),
                np.ones(1, dtype=np.int64),
                np.array([[0, 3], [2, 1]]),
                np.array([0, 1]),
                StairConfig(),
            )
        )

    def test_plan_layer_breaks_mean_tie_by_migration_count(self):
        plan = StairPlanner.plan_layer(
            np.ones((1, 3)),
            np.ones(1, dtype=np.int64),
            np.array([[0, 2], [0, 1]]),
            np.array([0, 1]),
            StairConfig(),
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.predicted_imbalance.mean_ratio, 1.0)
        np.testing.assert_array_equal(plan.placement.rank_expert_ids, [[0, 2], [2, 1]])
        self.assertEqual(
            np.sum(plan.placement.source_rank_ids != np.arange(2)[:, None]), 1
        )

    def test_plan_layer_rejects_mean_regression(self):
        samples = np.array(
            [
                [21, 28, 26, 15],
                [28, 29, 29, 2],
                [13, 18, 8, 11],
                [18, 24, 17, 5],
                [20, 26, 6, 16],
            ]
        )

        plan = StairPlanner.plan_layer(
            samples,
            np.array([2, 5, 1, 3, 5]),
            np.array([[0, 3], [1, 2]]),
            np.array([0, 1]),
            StairConfig(),
        )

        self.assertIsNone(plan)

    def test_plan_rebalance_preserves_filtered_layers(self):
        samples = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [8.0, 7.0, 6.0, 5.0],
                ]
            ]
        )
        current = np.array(
            [
                [[0, 1], [2, 3]],
                [[0, 3], [2, 1]],
                [[0, 1], [2, 3]],
            ]
        )

        plan = StairPlanner.plan_rebalance(
            samples,
            current,
            np.array([np.nan, 1.0, np.nan]),
            np.array([0, 1]),
            StairConfig(),
        )

        np.testing.assert_array_equal(plan.rank_expert_ids[:2], current[:2])
        np.testing.assert_array_equal(plan.source_rank_ids[:2], [[[0, 0], [1, 1]]] * 2)
        np.testing.assert_array_equal(plan.source_slot_ids[:2], [[[0, 1], [0, 1]]] * 2)
        np.testing.assert_array_equal(plan.rank_expert_ids[2], [[0, 3], [2, 1]])
        np.testing.assert_array_equal(
            np.isnan(plan.predicted_mean_ratios), [True, True, False]
        )
        self.assertEqual(plan.predicted_mean_ratios[2], 1.0)

    def test_validate_plan_accepts_explicit_sources(self):
        current = np.array([[[0, 1], [2, 3]]])
        plan = StairPlan(
            rank_expert_ids=np.array([[[0, 2], [1, 3]]]),
            source_rank_ids=np.array([[[0, 1], [0, 1]]]),
            source_slot_ids=np.array([[[0, 0], [1, 1]]]),
            predicted_mean_ratios=np.array([1.0]),
        )

        StairPlanner.validate_plan(current, plan, 4, np.zeros(2, dtype=np.int64), 1, 1)

    def test_validate_plan_accepts_zero_redundancy_three_rank_cycle(self):
        current = np.array([[[0], [1], [2]]])
        plan = StairPlan(
            rank_expert_ids=np.array([[[1], [2], [0]]]),
            source_rank_ids=np.array([[[1], [2], [0]]]),
            source_slot_ids=np.zeros_like(current),
            predicted_mean_ratios=np.array([1.0]),
        )

        StairPlanner.validate_plan(current, plan, 3, np.zeros(3, dtype=np.int64), 1, 1)

    def test_validate_plan_rejects_false_source_ownership(self):
        current = np.array([[[0, 1], [2, 3]]])
        plan = StairPlan(
            rank_expert_ids=np.array([[[0, 2], [1, 3]]]),
            source_rank_ids=np.array([[[0, 1], [0, 1]]]),
            source_slot_ids=np.array([[[0, 1], [1, 1]]]),
            predicted_mean_ratios=np.array([1.0]),
        )

        with self.assertRaisesRegex(ValueError, "does not own"):
            StairPlanner.validate_plan(
                current, plan, 4, np.zeros(2, dtype=np.int64), 1, 1
            )

    def test_validate_plan_accepts_unlimited_transfer_budgets(self):
        current = np.array([[[0, 1], [2, 3], [4, 5]]])
        plan = StairPlan(
            rank_expert_ids=np.array([[[2, 3], [0, 1], [4, 5]]]),
            source_rank_ids=np.array([[[1, 1], [0, 0], [2, 2]]]),
            source_slot_ids=np.array([[[0, 1], [0, 1], [0, 1]]]),
            predicted_mean_ratios=np.array([1.0]),
        )

        StairPlanner.validate_plan(current, plan, 6, np.arange(3), -1, -1)

    def test_validate_plan_rejects_invalid_controls(self):
        current = np.array([[[0, 1], [2, 3], [4, 5]]])
        plan = StairPlan(
            rank_expert_ids=np.array([[[2, 3], [0, 1], [4, 5]]]),
            source_rank_ids=np.array([[[1, 1], [0, 0], [2, 2]]]),
            source_slot_ids=np.array([[[0, 1], [0, 1], [0, 1]]]),
            predicted_mean_ratios=np.array([1.0]),
        )

        invalid_controls = (
            (6.0, 1, 1),
            (6, 0, 1),
            (6, 1.5, 1),
            (6, 1, -2),
            (6, 1, True),
        )
        for num_experts, rank_limit, node_limit in invalid_controls:
            with (
                self.subTest(controls=(num_experts, rank_limit, node_limit)),
                self.assertRaises(ValueError),
            ):
                StairPlanner.validate_plan(
                    current,
                    plan,
                    num_experts,
                    np.zeros(3, dtype=np.int64),
                    rank_limit,
                    node_limit,
                )

    def test_validate_plan_rejects_prediction_for_unchanged_layer(self):
        current = np.array([[[0], [1]]])
        plan = StairPlan(
            rank_expert_ids=current.copy(),
            source_rank_ids=np.array([[[0], [1]]]),
            source_slot_ids=np.zeros_like(current),
            predicted_mean_ratios=np.array([1.0]),
        )

        with self.assertRaisesRegex(ValueError, "finite for changed layers"):
            StairPlanner.validate_plan(
                current, plan, 2, np.zeros(2, dtype=np.int64), 1, 1
            )

    def test_validate_plan_rejects_non_float_predictions(self):
        current = np.array([[[0], [1]]])
        plan = StairPlan(
            rank_expert_ids=current.copy(),
            source_rank_ids=np.array([[[0], [1]]]),
            source_slot_ids=np.zeros_like(current),
            predicted_mean_ratios=np.array([True]),
        )

        with self.assertRaisesRegex(ValueError, "floating-point"):
            StairPlanner.validate_plan(
                current, plan, 2, np.zeros(2, dtype=np.int64), 1, 1
            )

    def test_statistics_reject_invalid_inputs(self):
        invalid_samples = np.array([[[1.0, -1.0]]])
        with self.assertRaises(ValueError):
            StairPlanner.compress_load_window(invalid_samples, 2)
        with self.assertRaises(ValueError):
            StairPlanner.weighted_moments(np.ones((2, 2)), np.array([1.0, 1.0]))
        with self.assertRaises(ValueError):
            StairPlanner.placement_imbalance(
                np.ones((1, 2)), np.array([0]), np.array([[0], [1]])
            )


class TestStairConfig(unittest.TestCase):
    def test_stair_config_defaults_and_overrides(self):
        defaults = StairConfig()
        config = StairConfig(rank_transfer_limit=2, load_risk_quantile=0.9)

        self.assertEqual(
            asdict(defaults),
            {
                "load_window_bins": 64,
                "load_risk_quantile": 0.75,
                "relative_balance_threshold": 0.95,
                "absolute_balance_threshold": 0.90,
                "rank_transfer_limit": -1,
                "cross_node_transfer_limit": -1,
                "replica_search_num_stages": 4,
                "replica_search_radius": 8,
                "replica_search_beam_size": 64,
                "placement_search_backtrack_limit": 32,
            },
        )
        self.assertEqual(config.rank_transfer_limit, 2)
        self.assertEqual(config.z_score, NormalDist().inv_cdf(0.9))

    def test_stair_config_default_factory_and_frozen_contract(self):
        first = StairConfig()
        second = StairConfig()

        self.assertIsNot(first, second)
        with self.assertRaises(FrozenInstanceError):
            first.load_window_bins = 2

    def test_stair_config_accepts_boundaries(self):
        for value in (
            {"load_window_bins": 2},
            {"load_window_bins": 256},
            {"load_risk_quantile": 0.500001},
            {"load_risk_quantile": 0.999999},
            {"relative_balance_threshold": 0.000001},
            {"relative_balance_threshold": 1},
            {"absolute_balance_threshold": 0.000001},
            {"absolute_balance_threshold": 1},
            {"rank_transfer_limit": 1},
            {"rank_transfer_limit": -1},
            {"cross_node_transfer_limit": 0},
            {"cross_node_transfer_limit": -1},
            {"replica_search_num_stages": 1},
            {"replica_search_num_stages": 8},
            {"replica_search_radius": 0},
            {"replica_search_radius": 32},
            {"replica_search_beam_size": 1},
            {"replica_search_beam_size": 128},
            {"placement_search_backtrack_limit": 0},
            {"placement_search_backtrack_limit": 64},
        ):
            with self.subTest(value=value):
                StairConfig(**value)

    def test_stair_config_rejects_invalid_values(self):
        for value in (
            {"load_window_bins": 1},
            {"load_window_bins": 257},
            {"load_risk_quantile": 0.5},
            {"load_risk_quantile": 1},
            {"relative_balance_threshold": 0},
            {"relative_balance_threshold": 1.001},
            {"absolute_balance_threshold": 0},
            {"absolute_balance_threshold": 1.001},
            {"rank_transfer_limit": 0},
            {"cross_node_transfer_limit": -2},
            {"replica_search_num_stages": 0},
            {"replica_search_num_stages": 9},
            {"replica_search_radius": -1},
            {"replica_search_radius": 33},
            {"replica_search_beam_size": 0},
            {"replica_search_beam_size": 129},
            {"placement_search_backtrack_limit": -1},
            {"placement_search_backtrack_limit": 65},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                StairConfig(**value)

    def test_stair_config_rejects_boolean_and_non_finite_numbers(self):
        names = asdict(StairConfig())
        for name in names:
            with (
                self.subTest(name=name, value=True),
                self.assertRaisesRegex(ValueError, "must not be booleans"),
            ):
                StairConfig(**{name: True})
            for value in (math.nan, math.inf, -math.inf):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    StairConfig(**{name: value})

    def test_stair_config_rejects_unknown_option(self):
        with self.assertRaises(ValueError):
            StairConfig(**{"unknown_option": 0})

    def test_stair_config_rejects_internal_policy_controls(self):
        for name in ("z_score", "use_covariance", "hysteresis_enabled"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                StairConfig(**{name: 0})
