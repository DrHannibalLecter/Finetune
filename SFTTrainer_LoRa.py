from transformers import BitsAndBytesConfig, AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from trl import SFTTrainer
from datasets import Dataset
from peft import LoraConfig
from torch import bfloat16
import pandas as pd


# 4bit quantization
"""bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)"""

# 8 bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)


# Model and Tokenizer load
model_id = ''

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"



def dataset_handler(json_file):
    df = pd.read_json(json_file, orient='records')

    df['text'] = df.apply(concatenate_values, axis=1)

    return Dataset.from_pandas(df)



# Load dataset
dataset = dataset_handler("pth to json file")




peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

output_dir = "/home/arda/Desktop/result"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
warmup_ratio = 0.03
lr_scheduler_type = "constant"


training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config,
    max_seq_length=512,
    args=training_arguments
)


trainer.train()
trainer.model.save_pretrained(output_dir)



# rest is datset handler for specific prompt

def concatenate_values(row):
    if row["input"]:
        input = f"<<SYS>>\n{row['instruction']}\n<</SYS>>\n\n{row['input']}"
    else:
        input = row["instruction"]
    output = row["output"]
    return f"[INST] {input} [/INST] {output} "
