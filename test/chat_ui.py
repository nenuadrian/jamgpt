import os
import argparse
import json
import torch
import gradio as gr
from tokenizers import Tokenizer
from train_gpt import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Chat UI for GPT models")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./tokenizer_output/tokenizer.json",
        help="Path to trained tokenizer",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="chat",
        choices=["base", "chat"],
        help="Type of model (base or chat)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful AI assistant.",
        help="System prompt for chat model",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public sharing link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on",
    )
    return parser.parse_args()


def get_device():
    """Get the best available device for inference."""
    # Check if CUDA is available
    if torch.cuda.is_available():
        return "cuda"
    # Check if MPS is available (Apple Silicon)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple Metal Performance Shaders (MPS)")
        return "mps"

    print("Using CPU")
    return "cpu"


class ChatBot:
    """Chatbot wrapper for GPT models."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str,
        model_type: str = "chat",
        system_prompt: str = "You are a helpful AI assistant.",
    ):
        self.device = device
        self.model_type = model_type
        self.system_prompt = system_prompt

        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        config_dict = checkpoint["config"]
        config = GPTConfig(**config_dict)

        self.model = GPT(config)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device)
        self.model.eval()

        print(
            f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )

        # Special tokens
        self.eos_token = "<|endoftext|>"

    def format_prompt(self, message: str, history: list) -> str:
        """Format prompt based on model type."""
        if self.model_type == "chat":
            # Chat model format
            prompt = f"System: {self.system_prompt}\n\n"

            # Add conversation history
            for user_msg, bot_msg in history:
                prompt += f"User: {user_msg}\n\nAssistant: {bot_msg}\n\n"

            # Add current message
            prompt += f"User: {message}\n\nAssistant:"
        else:
            # Base model format (just continuation)
            prompt = ""
            for user_msg, bot_msg in history:
                prompt += f"{user_msg}\n{bot_msg}\n"
            prompt += message

        return prompt

    @torch.no_grad()
    def generate_response(
        self,
        message: str,
        history: list,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> str:
        """Generate response to user message."""
        # Format prompt
        prompt = self.format_prompt(message, history)

        # Tokenize
        encoding = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)

        # Truncate if too long
        max_length = self.model.config.block_size
        if input_ids.size(1) > max_length - max_new_tokens:
            input_ids = input_ids[:, -(max_length - max_new_tokens) :]

        # Generate
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        # Decode
        output_text = self.tokenizer.decode(output_ids[0].tolist())

        # Extract response
        if self.model_type == "chat":
            # Extract assistant response
            if "Assistant:" in output_text:
                response = (
                    output_text.split("Assistant:")[-1].split(self.eos_token)[0].strip()
                )
            else:
                response = output_text[len(prompt) :].split(self.eos_token)[0].strip()
        else:
            # For base model, just get the generated part
            response = output_text[len(prompt) :].split(self.eos_token)[0].strip()

        return response


def create_ui(chatbot: ChatBot):
    """Create Gradio UI."""

    with gr.Blocks(title="GPT Chat Interface") as demo:
        gr.Markdown(f"# 🤖 GPT Chat Interface")
        gr.Markdown(
            f"**Model Type:** {chatbot.model_type.upper()} | **Device:** {chatbot.device}"
        )

        if chatbot.model_type == "chat":
            gr.Markdown(f"**System Prompt:** {chatbot.system_prompt}")

        chatbot_ui = gr.Chatbot(
            label="Conversation",
            height=500,
            show_label=True,
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
                lines=2,
                scale=4,
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")

        with gr.Accordion("Generation Settings", open=False):
            max_tokens = gr.Slider(
                minimum=10,
                maximum=500,
                value=150,
                step=10,
                label="Max New Tokens",
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=200,
                value=50,
                step=1,
                label="Top K",
            )

        with gr.Row():
            clear_btn = gr.Button("Clear Chat")

        gr.Markdown("### Example Prompts")
        gr.Examples(
            examples=[
                ["What is machine learning?"],
                ["Explain neural networks in simple terms."],
                ["Write a Python function to calculate fibonacci numbers."],
                ["What are the benefits of exercise?"],
                ["Tell me a joke about programming."],
            ],
            inputs=msg,
        )

        def respond(message, chat_history, max_new_tokens, temp, top_k_val):
            """Handle user message and generate response."""
            if not message.strip():
                return "", chat_history

            try:
                # Generate response
                response = chatbot.generate_response(
                    message,
                    chat_history,
                    max_new_tokens=int(max_new_tokens),
                    temperature=temp,
                    top_k=int(top_k_val),
                )

                # Update chat history
                chat_history.append((message, response))

                return "", chat_history
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                chat_history.append((message, error_msg))
                return "", chat_history

        def clear_chat():
            """Clear chat history."""
            return []

        # Event handlers
        msg.submit(
            respond,
            inputs=[msg, chatbot_ui, max_tokens, temperature, top_k],
            outputs=[msg, chatbot_ui],
        )
        send_btn.click(
            respond,
            inputs=[msg, chatbot_ui, max_tokens, temperature, top_k],
            outputs=[msg, chatbot_ui],
        )
        clear_btn.click(clear_chat, outputs=[chatbot_ui])

        gr.Markdown(
            """
        ### Tips
        - Adjust **temperature** to control randomness (lower = more focused, higher = more creative)
        - Adjust **top_k** to control vocabulary diversity
        - Use **Clear Chat** to start a new conversation
        """
        )

    return demo


def main():
    args = parse_args()

    # Get best device
    device = get_device()

    # Initialize chatbot
    chatbot = ChatBot(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
        model_type=args.model_type,
        system_prompt=args.system_prompt,
    )

    # Create and launch UI
    demo = create_ui(chatbot)

    print(f"\nLaunching chat UI on port {args.port}...")
    print(f"Open your browser to: http://localhost:{args.port}")

    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
