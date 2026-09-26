# Coverage results by training seed

Input source: **matched**. Each entry averages placement seeds 0–4
within the indicated training seed. D and R are dimensionless; raw-MAE
contrasts are included separately in the CSV. D_KRIN is the residual excess
after KRIN, and R = D_uncentred - D_KRIN.

| Dataset | Backbone | Training seed | D, uncentred | D, KRIN | R |
|---|---|---:|---:|---:|---:|
| Chicago | SPIN-s | 0 | 1.207825409 | 0.000131512 | 1.207693897 |
| Chicago | SPIN-s | 1 | 0.549139578 | -0.000023331 | 0.549162910 |
| Chicago | SPIN-s | 2 | 1.417822661 | 0.000246586 | 1.417576075 |
| Chicago | IGNNK-R | 0 | 0.396298375 | -0.000032297 | 0.396330672 |
| Chicago | IGNNK-R | 1 | 0.457856634 | -0.000155850 | 0.458012484 |
| Chicago | IGNNK-R | 2 | 0.383512271 | -0.000099940 | 0.383612211 |
| Subway | SPIN-s | 0 | 1.149977674 | 0.000051900 | 1.149925774 |
| Subway | SPIN-s | 1 | 1.481966362 | -0.000252571 | 1.482218933 |
| Subway | SPIN-s | 2 | 1.505853011 | 0.000134200 | 1.505718811 |
| Subway | IGNNK-R | 0 | 0.235682806 | -0.000072469 | 0.235755275 |
| Subway | IGNNK-R | 1 | 0.510026775 | -0.000099049 | 0.510125824 |
| Subway | IGNNK-R | 2 | 0.479588550 | 0.000162918 | 0.479425631 |

`training_seed_mae.csv` also gives each fitted model's original-scale
MAE averaged over five placements, separately for each shift and coverage
condition. The mean of per-run ratios is retained; it is not replaced by a
ratio of mean errors. Uncertainty over the crossed factors is in `table4.csv`.
