import json
import os
import torch
from PIL import Image

#project imports
from models.vqa_model import VQAModel
from utils.preprocess import image_transform,get_tokenizer


# paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")



def load_model(device):
    #load ans mapping
    with open(os.path.join(DATA_DIR,"idx2ans_topK.json")) as f:
        idx2ans = json.load(f)

    #model
    model = VQAModel(num_answers=len(idx2ans))
    checkpoint = torch.load(CKPT_PATH, map_location=device,weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device=device)
    model.eval()

    return model, idx2ans

def predict(img_path,question,model,tokenizer,idx2ans,device):
    #load img and preprocess
    img = Image.open(img_path)
    img = image_transform(img).unsqueeze(0).to(device)  #[3,224,224] --> [1,3,224,224]


    #tokenize que
    text = tokenizer(
        question,
        padding = "max_length",
        truncation=True,
        max_length=16,
        return_tensors="pt"
    )

    input_ids = text["input_ids"].to(device)
    attention_mask = text["attention_mask"].to(device)

    #forward pass
    with torch.no_grad():
        logits = model(
            images = img,
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        pred = torch.argmax(logits,dim = 1).item()
    return idx2ans[str(pred)]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    tokenizer= get_tokenizer()
    model , idx2ans = load_model(device)

    #input
    image_path = os.path.join(DATA_DIR,"images","val2014","COCO_val2014_000000001153.jpg")
    question = "What is present in the image?"

    answer = predict(image_path,question,model,tokenizer,idx2ans,device)

    print("Question : ", question)
    print("Answer : ", answer)

if __name__ == "__main__":
    main()
