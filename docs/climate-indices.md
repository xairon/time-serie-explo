# Standardized climate indices

SPI, STI and SPEI as this application serves them: what they mean, how they are validated,
and what not to misread.

The indices themselves are **not computed here**. They are read from
`gold.fct_era5_indices_grid`, produced by the `hubeau_data_integration` warehouse. The method
— gamma fit for SPI, z-score for STI, generalized-logistic fit for SPEI, and Hargreaves
reference PET — is documented there, in `docs/ERA5.md`.

| Index | Variable | Distribution |
|-------|----------|--------------|
| **SPI** | Precipitation | Gamma |
| **STI** | Temperature | Normal (z-score) |
| **SPEI** | Precipitation − PET | Generalized logistic |

All three use the same 1991–2020 reference period and the 7-class McKee/WMO scale.

## Validation

A standardized index must follow ~N(0,1) **over its own reference period**. Measured on
`gold.fct_era5_indices_grid` — 41,960,400 rows, 11,496 cells:

| Index | Windows | Mean | Std dev | Median | Saturation \|z\| ≥ 3.08 |
|-------|---------|------|---------|--------|-------------------------|
| SPI | 1/3/6/12 | −0.008 → +0.002 | 0.985 – 1.031 | 0.00 – 0.04 | 0.05 – 0.51 % |
| STI | 1/3/6/12 | 0.000 | 0.983 | −0.09 – −0.02 | 0.00 – 0.04 % |
| SPEI | 1/3/6/12 | +0.004 → +0.006 | 0.999 – 1.013 | 0.01 – 0.03 | 0.012 – 0.035 % |

The acceptance criterion, fixed *before* measuring: mean ≈ 0 (±0.05), standard deviation ≈ 1
(±0.05), saturation of the same order as SPI (< 1 %). All three indices meet it on all four
windows.

SPEI coverage is uniform at **100 %** across the four windows (~4.14 M rows each), against
2.78–3.65 M under the earlier log-logistic fit. The calibration is therefore not just
preserved but measured over the whole domain.

### Control query — re-run it whenever the upstream marts change

```sql
SELECT fenetre, avg(spei), stddev_samp(spei),
       percentile_cont(0.5) WITHIN GROUP (ORDER BY spei),
       100.0*count(*) FILTER (WHERE abs(spei)>=3.08)/count(spei) AS sat_pct
FROM gold.fct_era5_indices_grid
WHERE month>='1991-01-01' AND month<'2021-01-01' AND spei IS NOT NULL
GROUP BY fenetre ORDER BY fenetre;
```

## Two things not to misread

**The NORMAL class covers 54.8 %, not the theoretical 59.9 %,** even though the standard
deviation is 1.008. The distribution keeps slightly heavier shoulders than a Gaussian. This
has no operational consequence — the McKee thresholds are the same ones SPI and STI use — and
it must **not** be read as a drought anomaly. It is a property of the GLO → normal transform
near the class boundaries.

**A decreasing decade mean is climate, not drift.** The window-1 SPEI mean decreases
monotonically: +0.038 (1990s), +0.007 (2000s), −0.016 (2010s), −0.117 (2020s). This was
predicted before being measured, and confirms a drying signal rather than a fitting bias.
Likewise, a severe recent month reflects its inputs: June 2026 is the driest month of the
sample at 49 mm of precipitation and comes out at a window-3 SPEI median of −1.94 —
meteorological, not artefactual.

## Cross-repository contract

The IPS/SSFI classification maths exists in two places and must not drift:

| Repository | File |
|------------|------|
| `time-serie-explo` (here) | `dashboard/utils/reference.py::value_to_zscore`, guarded by `tests/test_drought_classification_contract.py` |
| `hubeau_data_integration` | `src/hubeau_pipeline/ml/indices.py::compute_reference_grid`, guarded by `tests/test_indices.py::GOLDEN_Z_TO_CLASS` |

Same grid, same clips, same thresholds. The two golden tables are identical on purpose —
change one, change the other, and run both test suites.
