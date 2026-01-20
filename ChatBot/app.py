from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Use a lightweight chatbot model
model_name = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Chatbot API is running 🚀"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        user_message = req.message

        # Generate response (short & fast)
        inputs = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")
        outputs = model.generate(inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)

        bot_response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)

        return {"user_message": user_message, "bot_response": bot_response}
    except Exception as e:
        return {"error": str(e)}
