# Gradio Space with Live Demo

- Parent list: [task_list.md](task_list.md)
- Related vision: [vision.md](vision.md)

## Status

- Progress: In progress
- Owner:
- Target date:

## Checklist

- [x] Prepare deployment-ready Gradio app
- [x] Configure Space environment and dependencies
- [ ] Add in-Space llama.cpp runtime configuration for ZeroGPU
- [ ] Verify `llama-cpp-python` CUDA offload compatibility with ZeroGPU
- [ ] Serve only MiniCPM-family GGUF models, no Gemma runtime
- [ ] Convert selected MiniCPM LoRA adapters to llama.cpp-compatible GGUF adapters
- [ ] Route chat requests to local llama.cpp, with CPU GGUF fallback if ZeroGPU is incompatible
- [ ] Deploy and run live demo
- [ ] Verify stability and response quality
- [ ] Share public link

## Notes

- Primary target should mirror `build-small-hackathon/CodeFlow`: Hugging Face runs the Gradio Server/custom frontend and loads a GGUF model through `llama-cpp-python`.
- Desired hardware target: Hugging Face ZeroGPU. Modal should not be used for the deployed path.
- Keep the Space metadata as `sdk: gradio`; ZeroGPU is selected in the Space hardware settings.
- ZeroGPU details from HF docs:
  - Import `spaces`.
  - Decorate GPU-dependent generation functions with `@spaces.GPU`.
  - Use `duration=...` for calls that may exceed the default runtime window.
  - ZeroGPU is primarily compatible with PyTorch-based Gradio Spaces, so llama.cpp CUDA compatibility must be verified.
- The long-term model stack should be MiniCPM-only:
  - Base model: `openbmb/MiniCPM5-1B`, converted/quantized to GGUF unless later benchmarks pick a different MiniCPM checkpoint.
  - LoRA adapters:
    - `smolnalysis-ckan-retrieval-minicpm5-lora`
    - `smolnalysis-openui-translator-minicpm5-lora`
    - future data-analysis LoRA adapter if the analysis step moves from Python/tools into model inference.
- In-Space llama.cpp runtime shape:
  - `llama-cpp-python` installed from a CPU wheel first, or a CUDA-enabled wheel/build if ZeroGPU compatibility is confirmed
  - `huggingface_hub` downloads the MiniCPM GGUF on first run
  - `MODEL_REPO_ID`, `MODEL_FILENAME`, `MODEL_PATH`, `N_CTX`, `N_THREADS`, `MAX_TOKENS` environment variables control runtime behavior.
- If llama.cpp cannot dynamically select between simultaneously loaded LoRAs per request for this workflow, prefer pre-merged role-specific GGUF models for the in-Space runtime.
