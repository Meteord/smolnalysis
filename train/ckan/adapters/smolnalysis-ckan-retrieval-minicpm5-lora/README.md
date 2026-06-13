---
base_model: openbmb/MiniCPM5-1B
library_name: peft
model_name: smolnalysis-ckan-retrieval-minicpm5-lora
tags:
- base_model:adapter:openbmb/MiniCPM5-1B
- lora
- peft
- sft
- transformers
- trl
- tool-calling
- ckan
- open-data
license: mit
pipeline_tag: text-generation
---

# smolnalysis-ckan-retrieval-minicpm5-lora

This is a PEFT LoRA adapter for [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B), trained for the `smolnalysis` CKAN retrieval role.

The adapter is not a general chat model. It is a small policy model that proposes the next validated CKAN retrieval action in a Python-controlled agent loop.

## Task

Given a user request, CKAN endpoint, previous tool observations, and compact catalog context, the model emits exactly one JSON action:

```json
{"thought":"short decision summary","action":"tag_search","args":{"query":"Fahrrad","rows":10},"confidence":0.95}
```

Supported actions:

- `tag_search`
- `group_list`
- `organization_list`
- `package_search`
- `package_show`
- `select_resource`
- `finish`
- `ask_clarification`

Python remains responsible for validating actions, checking observed package/resource ids, executing CKAN API calls, retrying malformed output, and stopping the loop.

## Quick Start

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "openbmb/MiniCPM5-1B"
adapter = "build-small-hackathon/smolnalysis-ckan-retrieval-minicpm5-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter).eval()

messages = [
    {
        "role": "system",
        "content": (
            "You are the CKAN retrieval policy for smolnalysis. Emit strict JSON only. "
            "Output exactly one compact JSON object with keys: thought, action, args, confidence."
        ),
    },
    {
        "role": "user",
        "content": """Request: Zeig mir nützliche Daten zu Fahrrad.
Endpoint: https://opendata.muenchen.de/
Current state: No catalog tools have been run yet.
Observed packages: []
Observed resources: []
Observed tags: []
Observed groups: []
Observed organizations: []
Tool errors or empty results: []
Enough evidence: False
Desired next action type: tag_search
Required args schema: {"query":"search terms","rows":10}""",
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=384,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## smolnalysis Space Configuration

```text
SMOLNALYSIS_MINICPM_BACKEND=transformers
SMOLNALYSIS_MINICPM_TRANSFORMERS_MODEL_ID=openbmb/MiniCPM5-1B
SMOLNALYSIS_MINICPM_CKAN_RETRIEVAL_ADAPTER_REPO_ID=build-small-hackathon/smolnalysis-ckan-retrieval-minicpm5-lora
SMOLNALYSIS_MINICPM_CKAN_RETRIEVAL_TEMPERATURE=0
SMOLNALYSIS_MINICPM_MAX_NEW_TOKENS=384
```

## Training Procedure

This adapter was trained with supervised fine-tuning using TRL + PEFT LoRA.

- Base model: `openbmb/MiniCPM5-1B`
- Training data: Munich Open Data CKAN inventory scenarios
- Loss: assistant-only loss
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

The training script used a MiniCPM-compatible training-only chat template with `{% generation %}` markers so that only assistant JSON actions contribute to loss.

## Evaluation

Generation evaluation after the compact-target training run:

```json
{
  "total": 160,
  "json_parse_rate": 1.0,
  "valid_action_rate": 1.0,
  "exact_action_match_rate": 1.0,
  "exact_action_matches": 160,
  "issue_counts": {}
}
```

Hand challenge set:

```json
{
  "total": 8,
  "json_parse_rate": 1.0,
  "valid_action_rate": 1.0,
  "exact_action_match_rate": 1.0,
  "exact_action_matches": 8,
  "issue_counts": {}
}
```

## Limitations

- The adapter is trained for CKAN retrieval tool calling, not general conversation.
- It should be used behind strict JSON parsing and action validation.
- It may not generalize to every CKAN portal without additional traces from that portal.
- It does not execute tools or verify dataset suitability by itself.

## Framework Versions

- PEFT 0.19.1
- TRL 1.5.1
- Transformers 5.11.0
- PyTorch 2.12.0
- Datasets 5.0.0
- Tokenizers 0.22.2

## Citation

```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```
