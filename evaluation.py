import json
import torch
import os
from torch.utils.data import DataLoader, Subset

from datasets.vqa_dataset import VQADataset
from models.vqa_model import VQAModel
from utils.preprocess import image_transform, get_tokenizer
from utils.answer_utils import coco_soft_accuracy

from datasets.vqa_dataset import vqa_collate_fn

import random



PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANNOT_DIR = os.path.join(DATA_DIR, "annotations")
QUESTION_DIR = os.path.join(DATA_DIR, "questions")

SUBSET_SIZE = 5000

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = get_tokenizer()
    transform = image_transform

    with open(os.path.join(DATA_DIR, "ans2idx_topK.json")) as f:
        ans2idx = json.load(f)

    with open(os.path.join(DATA_DIR, "idx2ans_topK.json")) as f:
        idx2ans = json.load(f)

    # ---- Validation dataset ---- 
    val_dataset = VQADataset(
        question_file=os.path.join(QUESTION_DIR, "val.json"),
        answer_file=os.path.join(ANNOT_DIR, "val.json"),
        img_dir=os.path.join(DATA_DIR, "images", "val2014"),
        img_prefix="COCO_val2014",
        tokenizer=tokenizer,
        transforms=transform,
        ans2idx=ans2idx
    )

    # 5k subset for faster evaluation
    indices = random.sample(range(len(val_dataset)),SUBSET_SIZE)
    val_dataset = Subset(val_dataset, indices)


    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=vqa_collate_fn
    )
    print("Validation samples:", len(val_dataset))
    print("Validation batches:", len(val_loader))
    # ---- Load model ----
    model = VQAModel(num_answers=len(ans2idx)).to(device)

    ckpt = torch.load(
        os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth"),
        map_location="cpu",weights_only=True
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    strict_correct = 0
    total = 0
    soft_acc_sum = 0.0

    with torch.no_grad():
        for images, text,_,labels, answers in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            input_ids = text["input_ids"].squeeze(1).to(device)
            attention_mask = text["attention_mask"].squeeze(1).to(device)

            logits = model(images, input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            # strict accuracy
            strict_correct += (preds == labels).sum().item()
            total += labels.size(0)

            # soft accuracy
            pred_answers = [idx2ans[str(p.item())] for p in preds]
            soft_acc_sum += coco_soft_accuracy(pred_answers, answers)

            

    strict_acc = strict_correct / total
    soft_acc = soft_acc_sum / total

    print(f"STRICT ACC : {strict_acc*100:.4f}")
    print(f"SOFT ACC   : {soft_acc*100:.4f}")

if __name__ == "__main__":
    main()
