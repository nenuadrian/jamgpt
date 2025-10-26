import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm
from train_gpt import GPT, GPTConfig
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune GPT model for chat")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./tokenizer_output/tokenizer.json",
        help="Path to trained tokenizer",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to chat dataset (JSONL format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./chat_gpt_output",
        help="Output directory for finetuned model",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for finetuning",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=2000,
        help="Maximum training iterations",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=200,
        help="Evaluation interval",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=500,
        help="Save checkpoint interval",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful AI assistant.",
        help="System prompt for chat model",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory for TensorBoard logs",
    )
    return parser.parse_args()


class ChatDataset(Dataset):
    """Dataset for chat/instruction finetuning.

    Expected JSONL format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    or
    {"instruction": "...", "input": "...", "output": "..."}
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer: Tokenizer,
        block_size: int,
        system_prompt: str = "You are a helpful AI assistant.",
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.system_prompt = system_prompt

        # Special tokens
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = tokenizer.encode(self.eos_token).ids[0]

        # Load dataset
        print(f"Loading chat dataset from {dataset_path}")
        self.examples = []

        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset not found at {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.examples.append(data)

        print(f"Loaded {len(self.examples)} chat examples")

    def format_chat_messages(self, example):
        """Format chat messages into a single string."""
        # Handle different formats
        if "messages" in example:
            messages = example["messages"]
        elif "instruction" in example:
            # Convert instruction format to messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": example.get("instruction", "")},
            ]
            if example.get("input"):
                messages[-1]["content"] += "\n\n" + example["input"]
            messages.append({"role": "assistant", "content": example.get("output", "")})
        else:
            raise ValueError(f"Unknown format for example: {example.keys()}")

        # Format as chat
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}{self.eos_token}\n\n"

        return formatted

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Format the conversation
        text = self.format_chat_messages(example)

        # Tokenize
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids

        # Truncate or pad to block_size + 1
        if len(tokens) > self.block_size + 1:
            tokens = tokens[: self.block_size + 1]
        else:
            # Pad with EOS token
            tokens = tokens + [self.eos_token_id] * (self.block_size + 1 - len(tokens))

        # Create input and target
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        return x, y


def load_pretrained_model(checkpoint_path: str, device: str):
    """Load pretrained GPT model from checkpoint."""
    print(f"Loading pretrained model from {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct config
    config_dict = checkpoint["config"]
    config = GPTConfig(**config_dict)

    # Create model
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    print(
        f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    return model, config


@torch.no_grad()
def evaluate(model, eval_loader, device, max_batches=50):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for i, (x, y) in enumerate(eval_loader):
        if i >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def train(args):
    """Finetune GPT model for chat."""

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # Load pretrained model
    model, config = load_pretrained_model(args.pretrained_model_path, args.device)

    # Update block size if needed
    if args.block_size != config.block_size:
        print(
            f"Warning: Dataset block_size ({args.block_size}) differs from model block_size ({config.block_size})"
        )
        print(f"Using model's block_size: {config.block_size}")
        args.block_size = config.block_size

    # Create TensorBoard writer
    log_dir = os.path.join(args.log_dir, f"chat_{args.output_dir.split('/')[-1]}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")
    print(f"Run: tensorboard --logdir {args.log_dir}")

    # Log hyperparameters
    writer.add_text("hyperparameters", json.dumps(vars(args), indent=2), 0)
    writer.add_text("model_config", json.dumps(config.__dict__, indent=2), 0)
    writer.add_text("system_prompt", args.system_prompt, 0)

    # Create dataset
    print("Loading chat dataset...")
    dataset = ChatDataset(
        args.dataset_path,
        tokenizer,
        args.block_size,
        args.system_prompt,
    )

    # Split into train and eval
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size

    if eval_size == 0:
        print("Warning: Dataset too small for train/eval split. Using 80/20 split.")
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size

    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # Log dataset info
    writer.add_scalar("Dataset/train_size", len(train_dataset), 0)
    writer.add_scalar("Dataset/eval_size", len(eval_dataset), 0)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Create optimizer with lower learning rate for finetuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Log learning rate
    writer.add_scalar("Hyperparameters/learning_rate", args.learning_rate, 0)

    # Training loop
    print("Starting finetuning...")
    os.makedirs(args.output_dir, exist_ok=True)

    model.train()
    iter_num = 0
    best_eval_loss = float("inf")
    train_loader_iter = iter(train_loader)
    accumulated_loss = 0.0
    running_loss = 0.0
    log_interval = 10

    progress_bar = tqdm(total=args.max_iters, desc="Training")

    while iter_num < args.max_iters:
        # Get batch
        try:
            x, y = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            x, y = next(train_loader_iter)

        x, y = x.to(args.device), y.to(args.device)

        # Forward pass
        _, loss = model(x, y)
        loss = loss / args.gradient_accumulation_steps
        accumulated_loss += loss.item()
        running_loss += loss.item() * args.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights after gradient accumulation
        if (iter_num + 1) % args.gradient_accumulation_steps == 0:
            # Log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            writer.add_scalar("Gradients/norm", total_norm, iter_num)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            progress_bar.set_postfix({"loss": accumulated_loss})

            # Log training loss
            writer.add_scalar("Loss/train_step", accumulated_loss, iter_num)
            accumulated_loss = 0.0

        # Log running loss
        if iter_num % log_interval == 0 and iter_num > 0:
            avg_loss = running_loss / log_interval
            writer.add_scalar("Loss/train_running", avg_loss, iter_num)
            running_loss = 0.0

        # Evaluate periodically
        if iter_num % args.eval_interval == 0:
            eval_loss = evaluate(model, eval_loader, args.device)
            print(f"\nStep {iter_num}: eval loss {eval_loss:.4f}")

            # Log to TensorBoard
            writer.add_scalar("Loss/eval", eval_loss, iter_num)
            writer.add_scalar("Loss/best_eval", best_eval_loss, iter_num)

            # Save best model
            if eval_loss < best_eval_loss:
                improvement = best_eval_loss - eval_loss
                best_eval_loss = eval_loss
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_eval_loss": best_eval_loss,
                    "config": config.__dict__,
                }
                torch.save(
                    checkpoint, os.path.join(args.output_dir, "best_chat_model.pt")
                )
                print(
                    f"Saved best model with eval loss {best_eval_loss:.4f} (improved by {improvement:.4f})"
                )
                writer.add_scalar("Loss/improvement", improvement, iter_num)

        # Save checkpoint periodically
        if iter_num % args.save_interval == 0 and iter_num > 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "config": config.__dict__,
            }
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, f"chat_checkpoint_{iter_num}.pt"),
            )

        iter_num += 1
        progress_bar.update(1)

    progress_bar.close()

    # Save final model
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "config": config.__dict__,
    }
    torch.save(checkpoint, os.path.join(args.output_dir, "final_chat_model.pt"))

    # Save config
    with open(os.path.join(args.output_dir, "chat_config.json"), "w") as f:
        json.dump(
            {
                "config": config.__dict__,
                "system_prompt": args.system_prompt,
                "block_size": args.block_size,
            },
            f,
            indent=2,
        )

    print(f"\nFinetuning complete! Model saved to {args.output_dir}")

    # Test generation
    print("\nTesting chat generation...")
    test_responses = test_chat(model, tokenizer, args.device, args.system_prompt)

    # Log test responses to TensorBoard
    for i, (prompt, response) in enumerate(test_responses):
        writer.add_text(f"test_generation/prompt_{i}", prompt, iter_num)
        writer.add_text(f"test_generation/response_{i}", response, iter_num)

    # Close writer
    writer.close()
    print(f"TensorBoard logs saved to {log_dir}")


def test_chat(model, tokenizer, device, system_prompt):
    """Test the chat model with sample prompts."""
    model.eval()

    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about coding.",
    ]

    responses = []

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"User: {prompt}")
        print(f"{'='*60}")

        # Format as chat
        chat_text = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"

        # Tokenize
        encoding = tokenizer.encode(chat_text)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=100,
                temperature=0.7,
                top_k=50,
            )

        # Decode
        output_text = tokenizer.decode(output_ids[0].tolist())

        # Extract assistant response
        if "Assistant:" in output_text:
            response = (
                output_text.split("Assistant:")[-1].split("<|endoftext|>")[0].strip()
            )
            print(f"Assistant: {response}")
        else:
            response = output_text
            print(f"Generated: {output_text}")

        responses.append((prompt, response))

    return responses


if __name__ == "__main__":
    args = parse_args()
    train(args)
