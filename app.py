import gradio as gr
from transformers import GPT2Tokenizer

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize(text, method):
    if not text.strip():
        return "Please enter some text."

    if method == "Word":
        tokens = text.strip().split()
        ids = ["(N/A)" for _ in tokens]
    elif method == "Character":
        tokens = list(text)
        ids = ["(N/A)" for _ in tokens]
    elif method == "BPE (GPT-2)":
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
    else:
        tokens, ids = [], []

    result = f"Tokenization Type: {method}\n\n"
    for i, (tok, tid) in enumerate(zip(tokens, ids), start=1):
        result += f"{i:>2}. Token: {tok}\t→ ID: {tid}\n"
    return result

# Gradio UI
gr.Interface(
    fn=tokenize,
    inputs=[
        gr.Textbox(lines=5, label="Enter your text"),
        gr.Radio(["Word", "Character", "BPE (GPT-2)"], label="Tokenization Method", value="BPE (GPT-2)")
    ],
    outputs="text",
    title="Tokenization Visualizer",
    description="Enter text and select a tokenization method to visualize how the text is tokenized.",
).launch()
