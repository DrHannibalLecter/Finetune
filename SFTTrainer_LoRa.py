from transformers import BitsAndBytesConfig, LlamaTokenizerFast, TrainingArguments, AutoModelForCausalLM, Trainer
from trl import SFTTrainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import pandas as pd


output_dir = "/home/arda/Documents/Finetune/result"
per_device_train_batch_size = 16
gradient_accumulation_steps = 16
num_train_epochs = 3
save_steps = 200
logging_steps = 10
learning_rate = 3e-4
max_grad_norm = 0.3
warmup_ratio = 0.03
lr_scheduler_type = "linear"
resume_from_checkpoint = None

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)





# 4bit quantization
"""bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)"""

# 8 bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)


# Model and Tokenizer load
model_id = 'meta-llama/Llama-2-7b-chat-hf'

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False
tokenizer = LlamaTokenizerFast.from_pretrained(model_id, add_eos_token=True)


tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# rest is datset handler for specific prompt

def concatenate_values(row):
    if row["input"]:
        input = f"<<SYS>>\n{row['instruction']}\n<</SYS>>\n\n{row['input']}"
    else:
        input = row["instruction"]
    output = row["output"]
    return f"[INST] {input} [/INST] {output} "


def dataset_handler(json_file):
    df = pd.read_json(json_file, orient='records')

    df['text'] = df.apply(concatenate_values, axis=1)

    return Dataset.from_pandas(df)



# Load dataset
dataset = dataset_handler("./dataset/onur_alpaca.json")





training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    resume_from_checkpoint=resume_from_checkpoint,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config,
    max_seq_length=512,
    args=training_arguments
)


print(training_arguments)

trainer.train()
trainer.model.save_pretrained(output_dir)
