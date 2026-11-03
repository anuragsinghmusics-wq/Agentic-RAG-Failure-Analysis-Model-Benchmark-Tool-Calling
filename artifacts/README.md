# Heavyweight artifacts

The attached Kaggle output bundle contains the trained `3B-ft` PEFT adapter under `adapters/3B-ft/`. Its `adapter_model.safetensors` file is approximately 120 MB, which is above GitHub's standard 100 MB per-file limit. It is therefore intentionally not committed to the repository.

To publish or consume the adapter, use a Hugging Face model repository or Git LFS and keep the base model reference from `adapter_config.json`:

```text
Qwen/Qwen2.5-3B-Instruct
```

The notebook can regenerate the adapter from the filtered teacher traces. The committed report and evaluation traces do not require the checkpoint to inspect the published results.
