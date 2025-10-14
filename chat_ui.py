from __future__ import annotations
import argparse
import torch
import gradio as gr
from pathlib import Path

from policy import PolicyWithValue
from rollout import RLHFTokenizer
from formatters import format_prompt_only


def load_model(ckpt_path: str, bpe_dir: str, device):
    """Load the GRPO-trained model and tokenizer."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})

    vocab_size = cfg.get("vocab_size", 32000)
    block_size = cfg.get("block_size", 256)
    n_layer = cfg.get("n_layer", 2)
    n_head = cfg.get("n_head", 2)
    n_embd = cfg.get("n_embd", 128)

    # Load tokenizer
    tok = RLHFTokenizer(block_size=block_size, bpe_dir=bpe_dir)

    # Load model
    model = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model.lm.load_state_dict(ckpt["model"])
    model.eval()

    return model, tok, cfg


def generate_response(
    model,
    tok,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    device=None,
):
    """Generate a response from the model."""
    # Format prompt
    formatted = format_prompt_only(prompt).replace("</s>", "")
    prompt_ids = tok.encode(formatted)

    # Generate
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode full output
    full_ids = output[0].tolist()
    full_text = tok.decode(full_ids)

    # Extract response (after the prompt)
    response = tok.decode(full_ids[len(prompt_ids) :])

    return response, full_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to GRPO checkpoint (e.g., runs/grpo-demo/model_last.pt)",
    )
    p.add_argument(
        "--bpe_dir",
        type=str,
        default=None,
        help="Path to BPE tokenizer directory",
    )
    p.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    p.add_argument("--port", type=int, default=7860, help="Port for Gradio server")
    p.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"Loading model from {args.ckpt}...")
    model, tok, cfg = load_model(args.ckpt, args.bpe_dir, device)
    print(f"Model loaded successfully on {device}")
    print(f"Config: {cfg}")

    def chat_fn(
        message: str,
        history: list,
        max_tokens: int,
        temperature: float,
        top_k: int,
    ):
        """Gradio chat function."""
        if not message.strip():
            return history, ""

        # Generate response
        response, full_text = generate_response(
            model,
            tok,
            message,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )

        # Update history
        history.append((message, response))
        return history, ""

    def clear_fn():
        """Clear chat history."""
        return [], ""

    # Build Gradio interface
    with gr.Blocks(title="GRPO Chat Interface") as demo:
        gr.Markdown("# 🤖 GRPO Model Chat Interface")
        gr.Markdown(
            f"Chat with your GRPO-trained model. Model: `{Path(args.ckpt).name}`"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_label=True,
                )

                with gr.Row():
                    msg_box = gr.Textbox(
                        label="Your message",
                        placeholder="Type your message here...",
                        lines=2,
                        scale=4,
                    )
                    submit_btn = gr.Button("Send", scale=1, variant="primary")

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### Generation Settings")

                max_tokens_slider = gr.Slider(
                    minimum=16,
                    maximum=512,
                    value=128,
                    step=16,
                    label="Max New Tokens",
                )

                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                )

                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-K",
                )

                gr.Markdown("### Model Info")
                gr.Markdown(f"- **Vocab Size**: {cfg.get('vocab_size', 'N/A')}")
                gr.Markdown(f"- **Context**: {cfg.get('block_size', 'N/A')}")
                gr.Markdown(f"- **Layers**: {cfg.get('n_layer', 'N/A')}")
                gr.Markdown(f"- **Heads**: {cfg.get('n_head', 'N/A')}")
                gr.Markdown(f"- **Embedding**: {cfg.get('n_embd', 'N/A')}")
                gr.Markdown(f"- **Device**: {device}")

        # Event handlers
        submit_btn.click(
            fn=chat_fn,
            inputs=[
                msg_box,
                chatbot,
                max_tokens_slider,
                temperature_slider,
                top_k_slider,
            ],
            outputs=[chatbot, msg_box],
        )

        msg_box.submit(
            fn=chat_fn,
            inputs=[
                msg_box,
                chatbot,
                max_tokens_slider,
                temperature_slider,
                top_k_slider,
            ],
            outputs=[chatbot, msg_box],
        )

        clear_btn.click(fn=clear_fn, outputs=[chatbot, msg_box])

        gr.Markdown("---")
        gr.Markdown(
            "💡 **Tip**: If your model was trained with thinking tags, "
            "it may use `<thinking>...</thinking>` in its responses!"
        )

    print(f"\nStarting Gradio interface on port {args.port}...")
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
