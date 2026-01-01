import json
import torch
import os
import time
from torch.utils.data import DataLoader, Subset

from datasets.vqa_dataset import VQADataset
from datasets.vqa_dataset import vqa_collate_fn

#import models
from models.vqa_model import VQAModel

#import utils
from utils.preprocess import image_transform,get_tokenizer
from utils.answer_utils import coco_soft_accuracy



import random

random.seed(42)

    


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT,"data")
ANNOT_DIR = os.path.join(DATA_DIR,"annotations")
QUESTION_DIR = os.path.join(DATA_DIR,"questions")




def main():
    USE_SUBSET = False
    SUBSET_SIZE = 1000
    VAL_SUBSET = True
    VAL_SUBSET_SIZE = 5000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    tokenizer = get_tokenizer()
    transform = image_transform

    #load ans2idx dict
    with open(os.path.join(DATA_DIR,"ans2idx_topK.json"),'r') as f:
        ans2idx = json.load(f)

    with open(os.path.join(DATA_DIR,"idx2ans_topK.json"),'r') as f:
        idx2ans = json.load(f)

    # check 
    print("Number of answers:", len(ans2idx))

    #------Trainging data-------
    # dataset
    dataset = VQADataset(
        question_file= os.path.join(QUESTION_DIR,"train.json"),
        answer_file= os.path.join(ANNOT_DIR,"train.json"),
        img_dir=os.path.join(DATA_DIR,"images","train2014"),
        img_prefix="COCO_train2014",
        tokenizer=tokenizer,
        transforms=transform,
        ans2idx= ans2idx
    )
    if USE_SUBSET:
        dataset = Subset(dataset, list(range(SUBSET_SIZE)))
    #data loader
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=vqa_collate_fn,
        persistent_workers=True
    )
    
    #check
    print("Number of training samples:", len(dataset))
    print("Batches per epoch:", len(train_loader))
    #---------Validation data---------
    #dataset
    val_dataset = VQADataset(
        question_file= os.path.join(QUESTION_DIR,"val.json"),
        answer_file= os.path.join(ANNOT_DIR,"val.json"),
        img_dir=os.path.join(DATA_DIR,"images","val2014"),
        img_prefix="COCO_val2014",
        tokenizer=tokenizer,
        transforms=transform,
        ans2idx= ans2idx
    )
    if VAL_SUBSET:
        indices = random.sample(range(len(val_dataset)), VAL_SUBSET_SIZE)
        val_dataset = Subset(val_dataset,indices)
    #eval data loader
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=vqa_collate_fn,
        persistent_workers=True
    )

    #check
    print("Validation samples:", len(val_dataset))
    print("Validation batches:", len(val_loader))


    # model
    model = VQAModel(num_answers= len(ans2idx))
    model = model.to(device)

    #loss fun
    criterion = torch.nn.BCEWithLogitsLoss()

    #optimizer
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()), #filter(fun , iterable)
        lr = 1e-5
    )
    best_val_loss = float("inf")
    best_soft_acc = 0.0
    start_epoch = 0

    start_epoch = 0

    last_ckpt = os.path.join(PROJECT_ROOT,"checkpoints/best_model.pth")

    ## ---------------- loading config  -----------------
    if os.path.exists(last_ckpt):
        print("Found checkpoint")
        checkpoint = torch.load(last_ckpt,map_location=device,weights_only=True)
        
        model.load_state_dict(checkpoint["model_state_dict"],strict = False)
        best_soft_acc = checkpoint.get("soft_acc", 0.0)
        start_epoch = checkpoint.get("epoch", 0)
        best_soft_acc = checkpoint.get("soft_acc", 0.0)
        print(f"Model Loaded Successfully")
    else:
        print("Starting Training from scratch")


    epochs = 5
    
    for epoch in range(start_epoch,epochs):
        epoch_start = time.time()
        #-------------Training-----------
        model.train()
        running_loss = 0.0

        for images,text,soft_target,labels,_ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            soft_target = soft_target.to(device)

            # extract inp ids ans attention mask from text, [B,1,32], squeeze for [B,32]
            input_id = text["input_ids"].squeeze(1).to(device)
            attention_mask = text["attention_mask"].squeeze(1).to(device)

            optimizer.zero_grad()

            # forward pass
            logits = model(images,input_id,attention_mask)
             

            # loss calc
            loss = criterion(logits,soft_target)

            # grad calc
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        #-----------validation------
        model.eval()
        val_loss = 0.0
        strict_correct = 0
        total = 0
        soft_acc_sum = 0.0
        
        with torch.no_grad():
            for images,text,soft_target,labels,answers in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                soft_target = soft_target.to(device)

                input_id = text["input_ids"].squeeze(1).to(device)
                attention_mask = text["attention_mask"].squeeze(1).to(device)

                logits = model(images,input_id,attention_mask)
                loss = criterion(logits,soft_target)
                val_loss += loss.item()

                # ---------- STRICT ----------
                preds = torch.argmax(logits, dim=1)
                strict_correct += (preds == labels).sum().item()
                total += labels.size(0)

                # ---------- SOFT ACCURACY ----------
                pred_answers = [idx2ans[str(p.item())] for p in preds]
                soft_acc_sum += coco_soft_accuracy(pred_answers, answers)


        avg_val_loss = val_loss / len(val_loader)
        strict_acc = strict_correct / total
        soft_acc = soft_acc_sum / total
        avg_train_loss = running_loss / len(train_loader)
        
        print(f"Epoch : {epoch+1} | Train_loss = {avg_train_loss:.4f} | Val_loss = {avg_val_loss:.4f} | Strict Acc = {strict_acc:.4f} | Soft Acc = {soft_acc:.4f} ")

        #-------save last model------
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "soft_acc": soft_acc
            },
            "checkpoints/last_model.pth"
        )

        #-----save best model---------
        if soft_acc > best_soft_acc:
            best_soft_acc = soft_acc

            torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "soft_acc": soft_acc
            },
            "checkpoints/best_model.pth"
            )

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} time: {epoch_time:.2f} seconds")

if __name__ == "__main__":
    main()
