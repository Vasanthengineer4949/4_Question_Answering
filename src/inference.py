import config
from transformers import MT5Tokenizer
from huggingface_hub import hf_hub_download
from model import Model
from fastapi import FastAPI
import uvicorn

model = Model.load_from_checkpoint(hf_hub_download(config.MODEL_OUT, 'lit_model.ckpt'))
model.freeze()
tokenizer = MT5Tokenizer.from_pretrained(config.MODEL_CKPT)

app = FastAPI(debug=True)

def generate_answer(question, model, tokenizer):
    question_encoding = tokenizer(
                question["question"],
                question["context"],
                max_length=config.QUESTION_MAX_LEN,
                padding="max_length",
                truncation="only_second",
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt"   
            )
    
    print(question["question"])
    gen_ids = model.model.generate(
        input_ids = question_encoding["input_ids"],
        attention_mask = question_encoding["attention_mask"],
        num_beams = 1,
        max_length = config.ANSWER_MAX_LEN,
        repetition_penalty = 2.5,
        length_penalty = 1.0,
        early_stopping = True,
        use_cache = True
    )

    preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for gen_id in gen_ids
    ]

    return "".join(preds)


@app.get("/")
def home():
    return {"Project Name": "MLQA"}

@app.get("/predict")
def predict(
    context:str,
    question:str
    ):
    ques_cont = {"context": context, "question": question}
    answer_out = generate_answer(ques_cont, model, tokenizer)
    return answer_out

if __name__ == "__main__":
    uvicorn.run(app)