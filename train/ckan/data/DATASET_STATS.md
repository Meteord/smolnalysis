# CKAN Retrieval Dataset Stats

Generated from the München CKAN inventory at `https://opendata.muenchen.de/`.

## Curated Files

| File | Rows | Size |
| --- | ---: | ---: |
| `dataset_inventory.jsonl` | 336 | 460 KB |
| `harvested_inventory_scenarios.jsonl` | 1,321 | 2.8 MB |
| `scenarios_train.jsonl` | 1,000 | 2.1 MB |
| `scenarios_eval_golden.jsonl` | 60 | 111 KB |
| `generated/valid_train_1000_repaired.jsonl` | 1,000 | 1.6 MB |
| `generated/valid_eval_golden_60_repaired.jsonl` | 60 | 96 KB |
| `generated/report_train_1000_repaired.jsonl` | 1,000 | 418 KB |
| `generated/report_eval_golden_60_repaired.jsonl` | 60 | 25 KB |

## Inventory Coverage

- Dataset inventory rows: 336
- Inventory-grounded scenarios: 1,321
- Scenario package ids: 336
- Train scenario package ids: 322
- Golden eval scenario package ids: 14
- Train/eval package overlap: 0
- Generated train package ids: 319
- Generated eval package ids: 14
- Generated train/eval package overlap: 0

Top groups in the inventory:

| Group | Datasets |
| --- | ---: |
| `tran` | 64 |
| `soci` | 49 |
| `gove` | 35 |
| `educ` | 30 |
| `econ` | 14 |
| `tech` | 8 |
| `heal` | 2 |
| `envi` | 1 |
| `just` | 1 |

Top organizations in the inventory:

| Organization | Datasets |
| --- | ---: |
| `mobilitaetsreferat` | 113 |
| `statistisches-amt` | 102 |
| `kreisverwaltungsreferat` | 16 |
| `geodatenservice-muenchen` | 15 |
| `referat-fuer-klima-und-umweltschutz` | 12 |
| `it-referat` | 10 |
| `direktorium-der-landeshauptstadt-muenchen` | 8 |
| `baureferat` | 7 |
| `referat-fuer-stadtplanung-und-bauordnung` | 5 |
| `sozialreferat` | 5 |

## Scenario Action Mix

Inventory-grounded scenario pool:

| Action | Count |
| --- | ---: |
| `package_search` | 336 |
| `package_show` | 336 |
| `select_resource` | 342 |
| `finish` | 171 |
| `reject_result` | 136 |

Train scenarios:

| Action | Count |
| --- | ---: |
| `package_search` | 236 |
| `package_show` | 236 |
| `select_resource` | 235 |
| `finish` | 162 |
| `reject_result` | 131 |

Golden eval scenarios:

| Action | Count |
| --- | ---: |
| `package_search` | 14 |
| `package_show` | 14 |
| `select_resource` | 18 |
| `finish` | 9 |
| `reject_result` | 5 |

## Generated SFT Action Mix

Train examples, after repair:

| Action | Count |
| --- | ---: |
| `package_search` | 236 |
| `package_show` | 236 |
| `select_resource` | 233 |
| `finish` | 162 |
| `reject_result` | 133 |

Golden eval examples, after repair:

| Action | Count |
| --- | ---: |
| `package_search` | 14 |
| `package_show` | 14 |
| `select_resource` | 18 |
| `finish` | 9 |
| `reject_result` | 5 |

## Validation

- Train examples: 1,000 valid, 0 rejected after repair.
- Golden eval examples: 60 valid, 0 rejected after repair.
- Repairs only trimmed overlong `thought` fields.
- Natural-language request leak checks were performed during scenario generation: group and organization constraints are kept in retrieval state, not in the user request.
