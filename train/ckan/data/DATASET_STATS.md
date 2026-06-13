# CKAN Retrieval Dataset Stats

Generated from the Munich CKAN inventory at `https://opendata.muenchen.de/`.

## Current Multi-Tool Files

| File | Rows | Size |
| --- | ---: | ---: |
| `dataset_inventory.jsonl` | 336 | 449.3 KB |
| `harvested_inventory_scenarios.jsonl` | 2,329 | 4.8 MB |
| `scenarios_train.jsonl` | 1,600 | 3.3 MB |
| `scenarios_eval_golden.jsonl` | 160 | 328.3 KB |
| `scenarios_200.jsonl` | 200 | 435.1 KB |
| `generated/valid_examples_multitool_train_1600_repaired.jsonl` | 1,600 | 2.7 MB |
| `generated/valid_examples_multitool_eval_160.jsonl` | 160 | 271.6 KB |
| `generated/valid_examples_multitool_200_repaired.jsonl` | 200 | 349.2 KB |

Older generated files such as `valid_train_1000_repaired.jsonl` and `valid_eval_golden_60_repaired.jsonl` use the legacy `reject_result` contract and should not be used for the next CKAN LoRA run.

## Inventory Coverage

- Dataset inventory rows: 336
- Inventory-grounded scenarios: 2,329
- Scenario package ids: 336
- Train scenarios: 1,600
- Golden eval scenarios: 160
- Train/eval package overlap: 0

## Scenario Action Mix

Inventory-grounded scenario pool:

| Action | Count |
| --- | ---: |
| `finish` | 171 |
| `group_list` | 336 |
| `organization_list` | 336 |
| `package_search` | 472 |
| `package_show` | 336 |
| `select_resource` | 342 |
| `tag_search` | 336 |

Train scenarios:

| Action | Count |
| --- | ---: |
| `finish` | 159 |
| `group_list` | 241 |
| `organization_list` | 240 |
| `package_search` | 240 |
| `package_show` | 240 |
| `select_resource` | 240 |
| `tag_search` | 240 |

Golden eval scenarios:

| Action | Count |
| --- | ---: |
| `finish` | 12 |
| `group_list` | 23 |
| `organization_list` | 23 |
| `package_search` | 32 |
| `package_show` | 23 |
| `select_resource` | 24 |
| `tag_search` | 23 |

## Generated SFT Action Mix

Train examples after repair:

| Action | Count |
| --- | ---: |
| `finish` | 159 |
| `group_list` | 241 |
| `organization_list` | 240 |
| `package_search` | 240 |
| `package_show` | 240 |
| `select_resource` | 240 |
| `tag_search` | 240 |

Golden eval examples:

| Action | Count |
| --- | ---: |
| `finish` | 12 |
| `group_list` | 23 |
| `organization_list` | 23 |
| `package_search` | 32 |
| `package_show` | 23 |
| `select_resource` | 24 |
| `tag_search` | 23 |

Balanced smoke examples after repair:

| Action | Count |
| --- | ---: |
| `finish` | 29 |
| `group_list` | 29 |
| `organization_list` | 29 |
| `package_search` | 29 |
| `package_show` | 28 |
| `select_resource` | 28 |
| `tag_search` | 28 |

## Validation

- 20-example smoke run: 20 valid, 0 rejected.
- 200-example smoke run: 200 valid, 0 rejected after repair.
- 1,600-example train run: 1,600 valid, 0 rejected after repair.
- 160-example eval run: 160 valid, 0 rejected.
- Repair trims overlong `thought` fields and strips markdown JSON fences while keeping final assistant output strict JSON.
