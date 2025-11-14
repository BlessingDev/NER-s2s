from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-b-b-prefixlm-it")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/t5gemma-b-b-prefixlm-it",
    dtype=torch.bfloat16,
)
model.to("cuda")

messages = [
    {"role": "user", "content": "Tell me an unknown interesting biology fact about the brain."},
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True).to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))