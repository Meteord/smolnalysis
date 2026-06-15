# Adapter Router MLP

This folder trains a small classifier that receives tokenizer `input_ids` and predicts which MiniCPM adapter role to use.

Labels:

- `general_agent`: no adapter, for smalltalk and generic chat.
- `ckan_retrieval`: CKAN retrieval adapter.
- `openui_translator`: OpenUI-Lang adapter.

Generate balanced data:

```bash
python train/router/generate_router_data.py --per-label 240
```

Smoke-test tokenization:

```bash
python train/router/train_router_mlp.py --dry-run
```

Train:

```bash
python train/router/train_router_mlp.py --epochs 20
```

The output directory contains:

- `router_mlp.pt`: PyTorch state dict.
- `config.json`: architecture and label mapping.
- `metrics.json`: eval/test accuracy and confusion matrix.

Load for inference:

```python
from train.router.router_mlp import load_router_mlp

router, config = load_router_mlp("train/router/outputs/router-mlp")
output = router(input_ids=input_ids, attention_mask=attention_mask)
adapter = config.labels[int(output["logits"].argmax(dim=-1).item())]
```
