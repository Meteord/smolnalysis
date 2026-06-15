# Adapter Router

This folder trains a small classifier that receives tokenizer `input_ids` and predicts which MiniCPM adapter role to use.
The current router uses a frozen pretrained encoder and trains only a lightweight MLP head over the final hidden state.
The JSON training data format is unchanged.

Labels:

- `general_agent`: no adapter, for smalltalk and generic chat.
- `ckan_retrieval`: retrieval/tool-result adapter for the initial user request.
- `openui_translator`: OpenUI-Lang adapter after retrieval, when the prompt includes the user question plus `Tool result`.

The dataset is stage-oriented. A request such as "show this as a chart" still belongs to
`ckan_retrieval` until a structured tool result exists. Once the backend has the adapter
input containing the original user message and `Tool result`, the router should choose
`openui_translator`.

Generate balanced data:

```bash
python train/router/generate_router_data.py --per-label 1200
```

Smoke-test tokenization:

```bash
python train/router/train_router_mlp.py --dry-run
```

Train:

```bash
python train/router/train_router_mlp.py --epochs 20
```

By default, `--model-name` is used for both tokenization and the frozen encoder. To keep the tokenizer fixed while
using another compatible encoder, pass `--encoder-model-name`.

The output directory contains:

- `router_mlp.pt`: PyTorch state dict for the classifier head only.
- `config.json`: architecture and label mapping.
- `metrics.json`: eval/test accuracy and confusion matrix.

Upload the router artifacts to Hugging Face Hub:

```bash
HF_TOKEN=... python train/router/upload_router_to_hf.py \
  --router-dir train/router/outputs/router-mlp \
  --repo-id build-small-hackathon/smolnalysis-adapter-router
```

The app runtime uses `build-small-hackathon/smolnalysis-adapter-router` by default when local router artifacts are not present.
Set `SMOLNALYSIS_ROUTER_REPO_ID` only when using a different router repo.

Load for inference:

```python
from train.router.router_mlp import load_router_mlp

router, config = load_router_mlp("train/router/outputs/router-mlp")
output = router(input_ids=input_ids, attention_mask=attention_mask)
adapter = config.labels[int(output["logits"].argmax(dim=-1).item())]
```

The current `SmolnalysisMoE` wrapper uses the router for `adapter="auto"`:

1. Route the latest user request. If it predicts `ckan_retrieval`, run the retrieval adapter.
2. Pass the OpenUI adapter its normal input shape: original user message plus `Tool result`.

This avoids asking the router to infer OpenUI translation before the tool result exists.
