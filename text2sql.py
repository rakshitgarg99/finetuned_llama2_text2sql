import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import gradio as gr

# Load the entire model on the GPU 0
device_map = {"": 0} # use -1 for cpu


# model_name = "NousResearch/Llama-2-7b-chat-hf" # base model
model_name = "Rakshitgarg99/Llama-2-7b-text2sql-finetune"

new_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



# Define the function that generates the blog response
def generate_blog(input_text):


    pipe = pipeline(task="text-generation", model=new_model, tokenizer=tokenizer, max_length=500)
    result = pipe(f"<s>[INST] {input_text} [/INST]")
    # print(result[0]['generated_text'])
    return result[0]['generated_text'].split("[/INST]")[1].strip()

# Create inputs using Gradio components
input_text = gr.Textbox(label="Enter the SQL question info")

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_blog,
    inputs=[input_text],
    outputs="text",
    title="Generate Text TO SQL 🤖",
    description="Generate SQL queries based on the Textual Information.",
    theme="compact",
)

# Launch the Gradio interface in the notebook
iface.launch()
