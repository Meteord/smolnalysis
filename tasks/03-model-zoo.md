# Model Zoo with CKAN and OpenUI Integration

- Parent list: [task_list.md](task_list.md)
- Related vision: [vision.md](vision.md)

## Status

- Progress: In progress
- Owner:
- Target date:

## Checklist

- [x] Design model zoo structure
- [ ] Integrate CKAN dataset querying MiniCPM GGUF/LoRA
- [ ] Integrate OpenUI-specialized MiniCPM GGUF/LoRA
- [ ] Implement router for local llama.cpp role selection
- [ ] Test end-to-end query routing

## Notes

- Direction: MiniCPM-only. Do not carry the Gemma runtime into the deployed model zoo.
- Runtime target: llama.cpp in the Hugging Face Gradio Space on ZeroGPU where compatible, with CPU GGUF fallback in the same Space.
- Artifact target: GGUF base model plus GGUF LoRA adapters, or pre-merged role-specific GGUF models if adapter routing is simpler and more reliable.
- Initial roles:
  - `ckan_retrieval` -> `smolnalysis-ckan-retrieval-minicpm5-lora`
  - `openui_translator` -> `smolnalysis-openui-translator-minicpm5-lora`
  - optional later `data_analysis` LoRA if tool/Python analysis becomes model-owned.
