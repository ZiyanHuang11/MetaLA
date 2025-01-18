import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PATH = 'your/path/to/hf/checkpoints'

tokenizer = AutoTokenizer.from_pretrained(
    PATH,
    padding_side='left',
    truncation_side='left',
    use_fast=True,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.eval()

def chat(
    model,
    tokenizer,
    ### tokenizer parameter ###
    truncation=True,
    trunc_max_len=512,
    ### generation parameter ###
    # do_sample=False,
    # max_new_tokens=128,
    # temperature=0.1,
    # others=None,
    **kwargs,
):
    print("\n*****Start a conversation!*****")
    print('Tip: Use "stop" to terminate the conversation.')
    print('Tip: Use "clear" to clear the history context (recommended).')
    history = None
    while True:
        prompt = input("\nUser: ")
        if prompt.lower() == "stop":
            break
        if prompt.lower() == "clear":
            history = None
            continue
        inputs = tokenizer([prompt], padding=True, truncation=truncation, max_length=trunc_max_len, return_tensors='pt').to('cuda')
        if history is not None:
            input_ids = torch.cat((history, inputs.input_ids), dim=1)
        else:
            input_ids = inputs.input_ids
        history = model.generate(input_ids, **kwargs)
        outputs = history[0][input_ids.shape[1]:]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        print(f"Assistant: {response}") 

chat(model, tokenizer, do_sample=False, max_new_tokens=64)
# chat(model, tokenizer, do_sample=True, max_new_tokens=512, temperature=0.1)
