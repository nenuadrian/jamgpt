import json
import argparse
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Create chat dataset from HuggingFace")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tatsu-lab/alpaca",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./chat_data.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Maximum number of samples to use",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    return parser.parse_args()


def convert_to_messages(example, dataset_name):
    """Convert various formats to messages format."""
    
    # Alpaca format
    if "instruction" in example:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        if example.get("input"):
            messages[0]["content"] += "\n\n" + example["input"]
        return {"messages": messages}
    
    # ShareGPT format
    elif "conversations" in example:
        messages = []
        for turn in example["conversations"]:
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})
        return {"messages": messages}
    
    # OpenAI format (already in messages format)
    elif "messages" in example:
        return example
    
    # Simple prompt/response format
    elif "prompt" in example and "response" in example:
        return {
            "messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]},
            ]
        }
    
    else:
        raise ValueError(f"Unknown format: {example.keys()}")


def main():
    args = parse_args()
    
    print(f"Loading dataset: {args.dataset_name}")
    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying without split specification...")
        dataset = load_dataset(args.dataset_name)
        if isinstance(dataset, dict):
            dataset = dataset[args.split]
    
    # Limit samples
    if len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
    
    print(f"Converting {len(dataset)} examples...")
    
    # Convert and write
    with open(args.output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            try:
                converted = convert_to_messages(example, args.dataset_name)
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1} examples...")
            except Exception as e:
                print(f"Error converting example {i}: {e}")
                continue
    
    print(f"Dataset saved to {args.output_path}")
    
    # Show sample
    print("\nSample example:")
    with open(args.output_path, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
        print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
