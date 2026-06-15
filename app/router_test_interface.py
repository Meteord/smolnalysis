from __future__ import annotations

import argparse
from typing import Any

try:
    from .backend.smolnalysis_model_wrapper import SmolnalysisMoE
except ImportError:
    from backend.smolnalysis_model_wrapper import SmolnalysisMoE


def inspect_prompt(
    wrapper_model: SmolnalysisMoE,
    prompt: str,
    *,
    generate: bool = False,
    max_new_tokens: int = 128,
) -> str:
    text = prompt.strip()
    if not text:
        return "Enter a user message."

    messages = [{"role": "user", "content": text}]
    preprocessed, decision = wrapper_model.route(messages)
    selected_adapter = decision.adapter if decision else None
    role = decision.role if decision else "general_agent"
    adapter_source = wrapper_model.adapter_source_for_role(selected_adapter or role)

    lines = [
        "model_wrapper: loaded",
        f"base_model: {wrapper_model.model_base_name}",
        f"router_output_dir: {wrapper_model.router_output_dir}",
    ]
    if decision is None:
        lines.append("router_prediction: none")
    else:
        lines.extend(
            [
                f"router_prediction: {decision.role}",
                f"router_confidence: {decision.confidence:.3f}",
                f"router_logits: {[round(value, 3) for value in decision.logits]}",
            ]
        )

    lines.append(f"selected_adapter: {selected_adapter or 'base'}")
    if adapter_source is None:
        lines.append("adapter_source: base/no adapter")
    else:
        source_type = "path" if adapter_source.is_path else "repo"
        lines.append(f"adapter_source: {source_type}:{adapter_source.source}")

    if generate:
        try:
            output = wrapper_model.generate_text(
                preprocessed,
                adapter="auto",
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            lines.extend(["", "model_output:", output])
        except Exception as exc:
            lines.append(f"model_error: {type(exc).__name__}: {exc}")
    else:
        lines.append("generation: disabled")
    return "\n".join(lines)


def run_cli(wrapper_model: SmolnalysisMoE, *, generate: bool, max_new_tokens: int) -> None:
    mode = "route + generate" if generate else "route only"
    print(f"Model-wrapper router test ({mode}). Type /exit to quit.\n")
    while True:
        prompt = input("user> ").strip()
        if prompt.lower() in {"/exit", "exit", "quit", "/quit"}:
            return
        print(inspect_prompt(wrapper_model, prompt, generate=generate, max_new_tokens=max_new_tokens))
        print()


def run_web(
    wrapper_model: SmolnalysisMoE,
    host: str,
    port: int | None,
    *,
    generate: bool,
    max_new_tokens: int,
) -> Any:
    import gradio as gr

    def inspect(prompt: str) -> str:
        return inspect_prompt(wrapper_model, prompt, generate=generate, max_new_tokens=max_new_tokens)

    demo = gr.Interface(
        fn=inspect,
        inputs=gr.Textbox(label="User message", lines=4),
        outputs=gr.Textbox(label="Model-wrapper router decision", lines=16),
        title="smolnalysis model-wrapper router test",
        allow_flagging="never",
    )
    return demo.launch(server_name=host, server_port=port)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively inspect routing through SmolnalysisMoE only.")
    parser.add_argument("--web", action="store_true", help="Launch a tiny Gradio UI instead of the terminal prompt.")
    parser.add_argument("--generate", action="store_true", help="Also generate through SmolnalysisMoE after routing.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens for --generate.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for --web.")
    parser.add_argument("--port", type=int, default=None, help="Port for --web.")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wrapper_model = SmolnalysisMoE(load_in_4bit=args.load_in_4bit)
    if args.web:
        run_web(wrapper_model, args.host, args.port, generate=args.generate, max_new_tokens=args.max_new_tokens)
    else:
        run_cli(wrapper_model, generate=args.generate, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
