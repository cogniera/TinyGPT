# TinyGPT

TinyGPT is a small GPT-style language model trained on the TinyStories dataset using JAX and Flax NNX.
The project is implemented in the notebook `miniGPT.ipynb` and covers the full pipeline:

- model architecture
- data processing
- training with checkpointing
- text generation (inference)

## Project Overview

This notebook builds a decoder-only transformer from scratch with:

- token + positional embeddings
- causal self-attention
- stacked transformer blocks (pre-layer norm)
- linear language modeling head

The model is trained autoregressively (next-token prediction) on TinyStories text split by `<|endoftext|>`.

## Model Configuration (Current Notebook)

- Sequence length: `128`
- Embedding dimension: `256`
- Attention heads: `8`
- Feed-forward dimension: `1024`
- Transformer blocks: `6`
- Tokenizer: `tiktoken` GPT-2 encoding
- Epochs: `3`
- Batch size: `32`
- Max stories loaded: `100000`

## Tech Stack

- Python
- JAX
- Flax NNX
- Optax
- Orbax (checkpoints)
- tiktoken
- grain (data loading)

## Notebook Structure

The notebook is organized into four parts:

1. Model Architecture
2. Data Loading
3. Training
4. Inference

### 1) Model Architecture

- `TokenEmbedding`: combines token and learned positional embeddings
- `causal_attention_mask`: creates a lower-triangular mask for autoregressive attention
- `TransformerBlock`: pre-LN attention + MLP with residuals
- `miniGPTModel`: full GPT-style stack and vocabulary projection

### 2) Data Loading

- Reads TinyStories training text file
- Splits stories using `<|endoftext|>` delimiter
- Tokenizes with GPT-2 tokenizer
- Truncates/pads to fixed length (`maxlen`)
- Uses `grain` `DataLoader` with batching

### 3) Training

- Loss: cross-entropy with integer labels (`optax.softmax_cross_entropy_with_integer_labels`)
- LR schedule: warmup + cosine decay
- Optimizer: AdamW (`optax.adamw`)
- JIT-compiled train step with `@nnx.jit`
- Checkpoints saved every `100` steps and at each epoch end via Orbax

### 4) Inference

- Restores a saved checkpoint
- Generates text autoregressively with temperature sampling
- Stops when `<|endoftext|>` is generated or max tokens reached

## Setup

Install dependencies:

```bash
pip install jax flax optax orbax-checkpoint tiktoken grain
```

If you are running in Colab (as in the notebook), mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then update paths in the notebook to point to:

- TinyStories training text file
- checkpoint output directory

## How To Run

Open and run `tinyStoriesGPT.ipynb` top-to-bottom:

1. Install/import dependencies
2. Configure dataset and checkpoint paths
3. Build model
4. Load data
5. Train and save checkpoints
6. Restore checkpoint
7. Generate stories from prompts

## Dataset Format

The loader expects a plain text file containing stories separated by `<|endoftext|>`.

Example:

```text
Story one text...<|endoftext|>
Story two text...<|endoftext|>
```

## Checkpoints

Checkpoints are written to the configured `checkpoint_dir` with names such as:

- `epoch_1_step_100`
- `epoch_1_final`

To run inference, restore one checkpoint into the model state and call `generate(...)`.

## Notes

- The notebook currently uses Google Drive paths and Colab mounting.
- If you run locally, replace those paths with local filesystem paths.
- Hyperparameters are intentionally small to keep the model lightweight.

## Author

- Paarth Sharma
