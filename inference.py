from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizerFast
from transformers import pipeline

base_model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
lora_model_name_or_path = "./result/checkpoint-390"

tokenizer = LlamaTokenizerFast.from_pretrained(base_model_name_or_path, add_eos_token=False)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name_or_path,
    load_in_8bit=True,
    device_map="auto",
)


model = PeftModel.from_pretrained(model, lora_model_name_or_path)

model.eval()

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                temperature=0.1,
                repetition_penalty=1.1,
                max_new_tokens=512)


q_list = ["Üç ana renk nedir?",
          "İnsanları tanımlayan üç yaygın sıfatı adlandırın",
          "Nüfusu 10 milyondan fazla olan beş ülkenin listesini oluşturun",
          "Yol bisikleti ile dağ bisikleti arasındaki fark nedir?",
          "Her gün yürümek neden iyi bir fikirdir?",
          "Bir günlüğüne birinin gölgesi olduğunuzu hayal edin. Deneyimlerinizi şimdiki zamanda yazın.",
          "Üçgenin iç açıları toplamı kaçtır?"]


for i in q_list:
    input = i
    question = f"[INST] {input} [/INST]"
    res = pipe(question)
    print(res[0]["generated_text"])
