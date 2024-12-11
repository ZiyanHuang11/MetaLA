import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = 'your/path/to/hf/checkpoints'

tokenizer = AutoTokenizer.from_pretrained(PATH,
                                          padding_side='left',
                                          truncation_side='left',
                                          use_fast=True,
                                          trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.eval()
print(model)

prompts = ["Austria, is a landlocked country in Central Europe", "小米公司",]

inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
print(inputs.input_ids.shape)

outputs = model.generate(inputs.input_ids, do_sample=False, max_new_tokens=256, attention_mask=inputs.attention_mask)

for i in range(len(prompts)):
    response = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print("=====Begin======")
    print(response)
    print("======End======")
