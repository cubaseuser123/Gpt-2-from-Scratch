# GPT-2 From Scratch 🧠

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## What is this?
This is a ground-up, from-scratch implementation of a Generative Pre-trained Transformer (GPT) language model, built entirely in PyTorch. The goal is to recreate the architecture of models like GPT-2 to understand exactly how they work under the hood.

## Inspiration & Intentions
This project is heavily inspired by Andrej Karpathy's "Let's build GPT" tutorials. 

I decided to dive into this project because I wanted to move beyond just using high-level APIs and actually build a strong, first-principles foundation in Deep Learning. By implementing the data pipelines, tokenization, self-attention mechanisms, and transformer blocks myself, my intention is to deeply understand the engineering and mathematical concepts that power modern Large Language Models (LLMs).

## Repository Structure & Progress 🚧

We are progressively evolving the model across three focused notebooks:

### 1. `gpt2_from_scratch.ipynb` (Foundations & Pipeline)
- Setting up the environment and downloading the `tinyshakespeare` dataset.
- Building a custom character-level tokenizer (encoding strings to integers and decoding back).
- Preparing the dataset by converting it into PyTorch tensors with a training (90%) and validation (10%) split.
- Engineering the data loader `get_batch()` function to slice context windows (`block_size`) and stack them into parallel batches (`batch_size`) for GPU processing.
- Implementing an initial baseline **Bigram Language Model** using `torch.nn.Module` with an autoregressive generation loop and PyTorch `AdamW` training loop.

### 2. `gpt2_consolidated.ipynb` (Baseline & Attention Sandbox)
- Consolidating the core Bigram model and evaluation pipeline into a clean, reproducible workflow.
- Refactoring the architecture to extract a token embedding layer (`n_embed`), inject **position embeddings**, and route output through a linear language modeling head.
- Educational sandbox exploring the **mathematical mechanics of self-attention**: demonstrating how batched matrix multiplication with lower-triangular masking (`torch.tril`) and normalization (`softmax` along `dim=-1`) enables tokens to selectively aggregate context from preceding timesteps without leaking future information.

### 3. `gpt2_transformer.ipynb` ⚡ *(Active Workshop)*
- **Current Progress:** Transitioning from mathematical attention theory into a learnable, fully-fledged Transformer network!
- **Single-Head Self-Attention (`Head`)**: Fully integrated learnable key, query, and value projections with lower-triangular causal masking (`softmax` along `dim=-1`), driving validation loss down from baseline `4.20` to `2.40`.
- **Multi-Head Self-Attention (`MultiHeadAttention`)**: Running multiple attention heads in parallel and concatenating outputs across channel dimensions (`dim=-1`) to capture diverse contextual relationships simultaneously, lowering validation loss further to `2.27`.
- **FeedForward Network (`FeedForward`)**: Integrated multi-layer perceptron with `ReLU` non-linearity to provide deep per-token computation after self-attention aggregation.
- **Up Next:** Adding residual (skip) connections, `LayerNorm`, and composing these modules into complete, deep Transformer Blocks (`Block`).
