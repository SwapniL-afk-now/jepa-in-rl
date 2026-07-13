import sys

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_path, adapter_dir, tokenizer_dir, out_dir = sys.argv[1:5]

print(f"loading base model from {base_path}")
model = AutoModelForCausalLM.from_pretrained(base_path, dtype="bfloat16")
print(f"loading adapter from {adapter_dir}")
model = PeftModel.from_pretrained(model, adapter_dir)
print("merging and unloading")
model = model.merge_and_unload()
model.save_pretrained(out_dir)

tok = AutoTokenizer.from_pretrained(tokenizer_dir)
tok.save_pretrained(out_dir)
print(f"saved merged model to {out_dir}")
