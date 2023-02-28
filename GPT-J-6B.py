from transformers import GPTJForCausalLM, GPT2Tokenizer

# GPT-J-6B 모델과 토크나이저 로드
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

# 입력 문장
input_text = "Hello, how are you?"

# 토큰화
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# GPT-J-6B 모델에 입력하여 예측
output = model.generate(input_ids, max_length=50, do_sample=True)

# 디코딩하여 출력
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
