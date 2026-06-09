import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL_ID = "google/gemma-4-E4B-it"
ADAPTER_DIR = "./gemma4-e4b-lora-adapter"


def get_input_device(model):
    return next(model.parameters()).device


def generate_reply(
    model,
    tokenizer,
    messages,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
):
    model.eval()
    device = get_input_device(model)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]

    turn_end_id = tokenizer.convert_tokens_to_ids("<turn|>")

    stop_ids = [tokenizer.eos_token_id]
    if isinstance(turn_end_id, int) and turn_end_id >= 0:
        stop_ids.append(turn_end_id)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0, prompt_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Clean up common chat stop tokens for display
    decoded = decoded.replace("<turn|>", "").replace("<eos>", "").strip()

    return decoded


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

print("\nInteractive chat ready.")
print("Type /exit to quit, /reset to clear history.\n")

messages = []

while True:
    user_text = input("You: ").strip()

    if not user_text:
        continue

    if user_text.lower() in {"/exit", "exit", "quit", "/quit"}:
        print("Bye.")
        break

    if user_text.lower() == "/reset":
        messages = []
        print("Conversation reset.\n")
        continue

    messages.append({
        "role": "user",
        "content": user_text,
    })

    assistant_text = generate_reply(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )

    print(f"Assistant: {assistant_text}\n")

    messages.append({
        "role": "assistant",
        "content": assistant_text,
    })