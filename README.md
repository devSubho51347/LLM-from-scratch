# LLM-from-scratch
We will write the custom code to build our very own LLM using PyTorch. We will go from base model to RLHF with PPO.

## Project Structure
The project is divided into 9 parts, each representing a major step in building the LLM:

- **part1/**: Core Transformer Architecture (positional embeddings, attention, multi-head, feed-forward, LayerNorm)
- **part2/**: Training a Tiny LLM (tokenization, dataset batching, training loop, sampling, evaluation)
- **part3/**: Modernizing the Architecture (RMSNorm, RoPE, SwiGLU, KV cache, sliding-window attention)
- **part4/**: Scaling Up (BPE tokenization, mixed precision, learning schedules, checkpointing, logging)
- **part5/**: Mixture-of-Experts (MoE theory, implementing MoE layers, hybrid architectures)
- **part6/**: Supervised Fine-Tuning (SFT) (instruction datasets, causal LM loss, curriculum learning, evaluation)
- **part7/**: Reward Modeling (preference datasets, reward architecture, loss functions, sanity checks)
- **part8/**: RLHF with PPO (policy with value head, PPO objective, training loop, stability tricks)
- **part9/**: RLHF with GRPO (group-relative baseline, advantage calculation, KL regularization)

Each part contains an `orchestrator.py` file that calls all substeps for that part.

- **utils/**: Shared utilities (checkpointing, common imports, etc.)
