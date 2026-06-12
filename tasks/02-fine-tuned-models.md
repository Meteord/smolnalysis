# Fine-Tuned Models (data analysis and OpenUI translation)

- Parent list: [task_list.md](task_list.md)
- Related vision: [vision.md](vision.md)

## Status

- Progress: In progress
- Owner:
- Target date:

## Checklist

- [x] Define training objectives for both models
- [ ] Prepare and clean training dataset
- [ ] Train MiniCPM data analysis LoRA if needed
- [ ] Train MiniCPM OpenUI translation LoRA
- [ ] Evaluate quality and document metrics

## Notes

- Model family target: MiniCPM, currently `openbmb/MiniCPM5-1B`.
- Deployment target: llama.cpp in the Hugging Face Gradio Space on ZeroGPU where compatible, with CPU GGUF fallback in the same Space.
- Export target: GGUF base model and llama.cpp-compatible LoRA adapters, or pre-merged role-specific GGUF models.
- Do not add new Gemma fine-tuning work for the deployed path.
- Keep adapters separate by role during training, even if deployment later uses merged GGUF artifacts.
