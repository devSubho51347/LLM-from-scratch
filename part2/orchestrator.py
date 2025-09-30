"""
Part 2: Training a Tiny LLM
Orchestrator for tokenization, dataset handling, and basic training.
"""

import torch


def main():
    print("Starting Part 2: Training a Tiny LLM")

    # 2.1 Byte-level tokenization
    print("2.1 Byte-level tokenization")
    text = "Hello world!"
    tokenized = byte_level_tokenize(text)
    print(f"Tokenized: {tokenized}")

    # 2.2 Dataset batching & shifting
    print("\n2.2 Dataset batching & shifting")
    data = torch.arange(1, 21).reshape(4, 5)  # Dummy dataset
    batch_x, batch_y = create_batches(data, batch_size=2, context_len=3)
    print(f"Batch X: {batch_x}")
    print(f"Batch Y: {batch_y}")

    # 2.3 Cross-entropy loss & label shifting
    print("\n2.3 Cross-entropy loss & label shifting")
    vocab_size = 10
    loss_fn = torch.nn.CrossEntropyLoss()
    logits = torch.randn(2, 5, vocab_size)
    targets = torch.randint(0, vocab_size, (2, 5))
    loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
    print(f"Loss: {loss.item()}")

    # 2.4 Training loop
    print("\n2.4 Training loop")
    # Placeholder: would train a tiny model from Part 1
    print("Training loop started (placeholder)")

    # 2.5 Sampling
    print("\n2.5 Sampling with temperature, top-k, top-p")
    # Placeholder

    # 2.6 Evaluating on val set
    print("\n2.6 Evaluating loss on val set")
    # Placeholder

    print("\nPart 2 completed!")


def byte_level_tokenize(text: str) -> list[int]:
    """2.1: Simple byte-level tokenization."""
    return [ord(c) for c in text]


def create_batches(data: torch.Tensor, batch_size: int, context_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """2.2: Create next-token prediction batches."""
    B, T = batch_size, context_len
    idx = torch.randint(0, len(data) - T, (B,))
    x = torch.stack([data[i:i+T] for i in idx])
    y = torch.stack([data[i+1:i+1+T] for i in idx])
    return x, y


if __name__ == "__main__":
    main()
