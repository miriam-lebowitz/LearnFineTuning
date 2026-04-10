# LearnFineTuning

Hands-on Jupyter notebooks for learning **supervised fine-tuning** of a small instruction model (**Microsoft Phi-3 Mini**) using **QLoRA** (4-bit base weights + trainable LoRA adapters). The `modules/` track is intentionally incomplete: you fill in `???` placeholders and dataset choices so the steps stick.

## What you will do

1. **Environment** — Install Transformers, Datasets, PEFT, TRL, Accelerate, and bitsandbytes; confirm GPU when available.
2. **Data** — Load a dataset from the [Hugging Face Hub](https://huggingface.co/datasets), map rows into Phi-3’s **chat** format, and build a text column for training.
3. **LoRA** — Load Phi-3 in **4-bit**, attach **LoRA** with `peft`, and inspect how many parameters are trainable.
4. **Training** — Run **`SFTTrainer`** (TRL) with `TrainingArguments`: batch size, LR, precision (`bf16` / `fp16`), checkpoints.
5. **Evaluation & export** — Try generations with the chat template, optionally score with `evaluate`, **merge** adapters, **save** locally, and optionally **push** to the Hub.

## Repository layout

| Path | Purpose |
|------|--------|
| `modules/` | **Lesson notebooks** — work through these in order (`environment` → `data` → `lora` → `training` → `eval`). |
| `answers/` | **Reference notebooks** — filled-in versions you can compare against when you are stuck. |

## How to run the notebooks

- **Single session (recommended):** Open the notebooks in order in one Jupyter or VS Code session and use the **same Python kernel** so `model`, `tokenizer`, and `dataset` stay in memory between phases.
- **Google Colab:** Upload or clone the repo, enable a **GPU** runtime for training, and run `modules/environment.ipynb` first (install cells are written for `!pip` where needed).
- **Local:** Use Python 3.10+ and a virtual environment. A **CUDA GPU** with enough VRAM for Phi-3 Mini in 4-bit is strongly recommended; CPU runs are possible but very slow for training.

## Stack (high level)

- **[Transformers](https://huggingface.co/docs/transformers)** — Model and tokenizer loading, chat templates.
- **[Datasets](https://huggingface.co/docs/datasets)** — Hub datasets and `.map()` preprocessing.
- **[PEFT](https://huggingface.co/docs/peft)** — LoRA configuration and `get_peft_model`.
- **[TRL](https://huggingface.co/docs/trl)** — `SFTTrainer` for supervised fine-tuning.
- **[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)** — 4-bit quantization for QLoRA.

## Hugging Face Hub (optional but useful)

- **Gated models / uploads:** [Create an account](https://huggingface.co/join) and run `huggingface-cli login` (or use a notebook token) so downloads and `push_to_hub` work when you need them.
- **Dataset access:** Some datasets require accepting terms on the dataset page before `load_dataset` succeeds.

## License

See [LICENSE](LICENSE) in this repository.
