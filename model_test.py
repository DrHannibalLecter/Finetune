"""Checks whether lora adaptor is empty or not"""

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
lora_model_name_or_path = "./models/7_epoch__merged-alpaca-dolly-bactrianx/checkpoint-7000"

access_token = ""

tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, token=access_token)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name_or_path,
    load_in_8bit=True,
    device_map="auto",
    token=access_token
)

"""loras = ["./models/3_epoch_onur-alpaca/onur-alpaca/checkpoint-1000",
        "./models/3_epoch_onur-alpaca/onur-alpaca/checkpoint-800",
        "./models/3_epoch_onur-alpaca/onur-alpaca/checkpoint-600",
        "./models/5_epoch_merged-alpaca-dolly/merged-alpaca-dolly/checkpoint-2400",
         "./models/5_epoch_merged-alpaca-dolly/merged-alpaca-dolly/checkpoint-2200",
         "./models/5_epoch_merged-alpaca-dolly/merged-alpaca-dolly/checkpoint-2000",
        "./models/5_epoch_merged-alpaca-dolly-bactrainx/checkpoint-5000",
         "./models/5_epoch_merged-alpaca-dolly-bactrainx/checkpoint-4800",
         "./models/5_epoch_merged-alpaca-dolly-bactrainx/checkpoint-4600",
        "./models/5_epoch_onur-alpaca/onur-alpaca/checkpoint-1800",
         "./models/5_epoch_onur-alpaca/onur-alpaca/checkpoint-1600",
         "./models/5_epoch_onur-alpaca/onur-alpaca/checkpoint-1400",
         ]"""

loras = ["./mistral_output/mist_result/checkpoint-50"]


for i in loras:
    lora_model_name_or_path = i
    model2 = PeftModel.from_pretrained(model, lora_model_name_or_path)
    lora_params = {n: p for n, p in model2.named_parameters() if "lora_B" in n}
    for n, p in lora_params.items():
        print(n, p.sum())
