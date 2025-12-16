!pip -q uninstall -y pyarrow || true
!pip -q install "pyarrow==19.0.0" "transformers >= 4.46.0" "trl >= 0.11.0" datasets accelerate peft bitsandbytes sentencepiece

import torch, os, json, random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
from google.colab import userdata, drive
drive.mount("/content/drive", force_remount=True)
userdata.get('HF_TOKEN')

train_data = "/content/drive/MyDrive/deepseek_lora_sentence/data/sent_train_sft.jsonl"
eval_data = "/content/drive/MyDrive/deepseek_lora_sentence/data/sent_eval_sft.jsonl"

print("Loading data")
ds = load_dataset("json", data_files={"train": train_data, "eval": eval_data})
print(ds)

ds

print("Data loaded")

model_id = "deepseek-ai/DeepSeek-V2-Lite-Chat"
adapter_path = "/content/drive/MyDrive/deepseek_lora_idiom_only/full/adapter"

bnb = BitsAndBytesConfig(
    load_in4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast = True)
tokenizer.padding_size = "right"
tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(model_id)
config.rope_scaling = {
    "type": "linear",
    "factor": 1.0
}

print("loading base model")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb,
    low_cpu_mem_usage = True,
    config = config
).to("cuda")

print("loading idiom adapter")
model = PeftModel.from_pretrained(model, adapter_path).to("cuda")

print("merging idiom adapter")
model.merge_and_unload()

print("merged")

print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("tokenizer loaded")

example = [
    {
      "role": "system", "content": "You are a professional Chinese to English translator"
    },
    {
        "role": "user", "content": "你好"
    },
    {
        "role": "assistant", "content": "Hello"
    }
]

rendered = tokenizer.apply_chat_template(example, tokenize = False, add_generation_prompt = False)

print(rendered)

res_temp = "Assistant:"

def format_data(example):
  text = tokenizer.apply_chat_template(
      example["messages"], tokenize = False, add_generation_prompt = False
  )

  return {"text": text}

train_proc = ds["train"].map(
    format_data,
    remove_columns = ds["train"].column_names,
    num_proc = 2
)

eval_proc = ds["eval"].map(
    format_data,
    remove_columns = ds["eval"].column_names,
    num_proc = 2
)

print("printing...")
print(train_proc[0]["text"])

lora_config = LoraConfig(
    r = 32,
    lora_alpha = 64,
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["q_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)


model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir = "/content/deepseek_lora_sentence",
    num_train_epochs = 5,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps = 2,
    learning_rate = 1e-4,
    warmup_ratio = 0.1,
    logging_steps = 20,
    eval_strategy = 'epoch',
    save_strategy = "epoch",
    fp16 = False,
    bf16 = True,
    report_to = "none",
    max_length = 256,
    packing = False,
    gradient_checkpointing = True
)

trainer = SFTTrainer(
    model = model,
    train_dataset = train_proc,
    eval_dataset = eval_proc,
    args = training_args
)

trainer.train()

save_dir = "/content/drive/MyDrive/deepseek_lora_sentence/full/adapter"
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("saved to", save_dir)