"""Tests for CSV-only Table 4 pairing, crossed resampling, and training-seed means."""
import unittest
from itertools import product
import numpy as np
import pandas as pd
from recalculate_coverage import MODELS, SHIFTS, COVERAGES, validate_runs, paired_contrasts, bootstrap_draws, summarise, publication_tables

def synthetic_runs():
    rows = []
    for (model, ts, ps, shift, coverage) in product(MODELS, range(3), range(5), SHIFTS, COVERAGES):
        alpha = 1.0 if shift == 'no shift' else 0.25
        sn = 10.0 * (ps + 1) * alpha
        penalty = 0.1 if model.endswith('_c') or shift == 'no shift' else 0.3
        ratio = 1.0 + (penalty if coverage == 'outside' else 0.0)
        rows.append(dict(dataset='cta', model=model, train_seed=ts, placement_seed=ps, shift=shift, coverage=coverage, MAE=ratio * sn, snaive7=sn, common_mode=0.8 * sn, n_target=21, n_observed=22, realised_shift=1.0))
    return pd.DataFrame(rows)

class CoverageTests(unittest.TestCase):

    def test_training_seed_export_averages_five_per_run_ratios(self):
        (runs, contrasts, summaries) = ([], [], [])
        for dataset in ('cta', 'mta'):
            d = synthetic_runs().assign(dataset=dataset)
            if dataset == 'mta':
                for (p, excess) in enumerate([1, 2, 4, 8, 16]):
                    mask = (d.model == 'spin') & (d.train_seed == 0) & (d.placement_seed == p) & (d['shift'] == 'regional shift') & (d.coverage == 'outside')
                    d.loc[mask, 'MAE'] = d.loc[mask, 'snaive7'] * (1.1 + excess)
            d['MAE_over_snaive7'] = d.MAE / d.snaive7
            c = paired_contrasts(d).assign(dataset=dataset)
            s = summarise(c, 100, 0).assign(dataset=dataset, score_support_status='test')
            runs.append(d)
            contrasts.append(c)
            summaries.append(s)
        exports = publication_tables(pd.concat(runs), pd.concat(contrasts), pd.concat(summaries), 'test')
        seeds = exports['training_seed_means.csv']
        self.assertEqual(len(seeds), 12)
        self.assertTrue(seeds.n_placements.eq(5).all())
        row = seeds[(seeds.dataset == 'Subway') & (seeds.backbone == 'SPIN-s') & (seeds.train_seed == 0)].iloc[0]
        self.assertAlmostEqual(row.D_uncentred, 6.2)
        self.assertAlmostEqual(row.D_KRIN, 0.0)
        self.assertAlmostEqual(row.R, 6.2)
        mae = exports['training_seed_mae.csv']
        self.assertEqual(len(mae), 96)
        cell = mae[(mae.dataset == 'Subway') & (mae.model == 'spin') & (mae.train_seed == 0) & (mae['shift'] == 'regional shift') & (mae.coverage == 'outside')].iloc[0]
        self.assertAlmostEqual(cell.MAE_over_snaive7, 7.3)
        self.assertNotAlmostEqual(cell.MAE_over_snaive7, cell.MAE / cell.snaive7)

    def test_incomplete_duplicate_invalid_and_unmatched_inputs_fail(self):
        d = synthetic_runs()
        validate_runs(d, 'cta')
        broken = [d.iloc[:-1], pd.concat([d, d.iloc[:1]])]
        for (col, value) in (('snaive7', 0), ('MAE', np.nan), ('snaive7', 99), ('dataset', 'mta')):
            edited = d.copy()
            edited.loc[0, col] = value
            broken.append(edited)
        for edited in broken:
            with self.subTest(case=len(edited), value=edited.iloc[0].to_dict()):
                with self.assertRaises(ValueError):
                    validate_runs(edited, 'cta')

    def test_normalisation_removes_pure_scale_contraction(self):
        d = synthetic_runs()
        d['MAE_over_snaive7'] = d.MAE / d.snaive7
        contrasts = paired_contrasts(validate_runs(d.sample(frac=1, random_state=3), 'cta'))
        extra = contrasts[contrasts.contrast == 'shift_excess_penalty']
        centred = extra[extra.model.str.endswith('_c')]
        np.testing.assert_allclose(centred[centred.metric == 'MAE_over_snaive7'].value, 0, atol=1e-14)
        self.assertTrue((centred[centred.metric == 'MAE'].value < 0).all())
        plain = extra[(extra.model == 'spin') & (extra.metric == 'MAE_over_snaive7')]
        np.testing.assert_allclose(plain.value, 0.2)
        reduction = contrasts[(contrasts.contrast == 'krin_reduction') & (contrasts.metric == 'MAE_over_snaive7')]
        np.testing.assert_allclose(reduction.value, 0.2)

    def test_crossed_draws_share_placement_across_training_seeds(self):
        rows = np.array([[0, 0, 0], [0, 1, 2], [2, 2, 1]])
        columns = np.array([[0, 0, 0, 0, 0], [4, 3, 2, 1, 0], [1, 1, 1, 1, 2]])
        matrix = np.arange(15).reshape(3, 5)
        expected = [matrix[np.ix_(r, c)].mean() for (r, c) in zip(rows, columns)]
        np.testing.assert_allclose(bootstrap_draws(matrix, rows, columns), expected)
        placements_only = np.tile(np.arange(5), (3, 1))
        np.testing.assert_allclose(bootstrap_draws(placements_only, rows, columns), np.arange(5)[columns].mean(axis=1))
        training_only = np.repeat(np.arange(3)[:, None], 5, axis=1)
        np.testing.assert_allclose(bootstrap_draws(training_only, rows, columns), np.arange(3)[rows].mean(axis=1))
if __name__ == '__main__':
    unittest.main()
