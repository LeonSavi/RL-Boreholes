# Option C reclassification report (REPORT MODE -- nothing written yet)

Total coarse-class (well, depth, fm) rows considered: **1,019,823**
Would be kept and renamed: **807,171** (79.1 %)
Would be **dropped**: **212,652** (20.9 %)

v7 rule: per-formation adaptive depth window only. If the row has fewer than 2 wireline channels, or no candidate template can be built within the window for this row's formation, the row is dropped. No off-formation or geology fallback.

## Per-class outcome (with reason)

| coarse class | reason | new label | rows | % of class |
|---|---|---|---:|---:|
| `clay` | protected | `clay` | 608,996 | 100.0 % |
| `claystone` | drop | **DROP** | 190,099 | 52.8 % |
| `claystone` | data_formation | `claystone_cool` | 119,262 | 33.2 % |
| `claystone` | data_formation | `claystone_hot` | 50,389 | 14.0 % |
| `other` | drop | **DROP** | 21,362 | 43.1 % |
| `other` | data_formation | `halite_pure` | 10,270 | 20.7 % |
| `other` | data_formation | `anhydrite` | 9,911 | 20.0 % |
| `other` | data_formation | `dolomite` | 5,734 | 11.6 % |
| `other` | data_formation | `claystone_hot` | 2,324 | 4.7 % |
| `sandstone` | drop | **DROP** | 1,191 | 80.7 % |
| `sandstone` | data_formation | `sandstone_shaly` | 208 | 14.1 % |
| `sandstone` | data_formation | `sandstone_clean` | 77 | 5.2 % |

## Reason summary

| reason | rows | description |
|---|---:|---|
| `protected` | 608,996 | `clay` -- left unchanged |
| `drop` | 212,652 | no usable per-formation template (or <2 wireline channels) |
| `data_formation` | 198,175 | matched a per-(formation, fine class) template |

## Drop breakdown by (coarse class, formation)

| coarse class | formation | rows dropped |
|---|---|---:|
| `claystone` | RN | 79,473 |
| `claystone` | KN | 64,371 |
| `other` | ZE | 21,362 |
| `claystone` | SL | 13,034 |
| `claystone` | SG | 10,734 |
| `claystone` | DC | 8,135 |
| `claystone` | AT | 6,932 |
| `claystone` | SK | 4,582 |
| `claystone` | ZE | 2,838 |
| `sandstone` | RB | 664 |
| `sandstone` | RO | 527 |

## Next step

If these numbers look right, re-run with `--apply` to write the reclassified `samples.parquet`. The old parquet is backed up to `samples.parquet.bak_pre_optionC` before the overwrite.

Per-row decisions: `plots/analysis/reclassify_decisions.csv` (1,019,823 rows).