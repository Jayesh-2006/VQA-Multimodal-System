import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils.answer_utils import normalize_answer



def build_soft_target(answer_list, ans2idx, num_answers):
    target = torch.zeros(num_answers)

    for ans in answer_list:
        if ans in ans2idx:
            target[ans2idx[ans]] += 1

    # VQA rule
    target = torch.clamp(target / 3.0, max=1.0)
    return target


class VQADataset(Dataset):
    def __init__(self,question_file,answer_file,img_dir,img_prefix,tokenizer,transforms,ans2idx):

        #---load json files
        #-----question
        with open(question_file,'r') as f:
            que_json = json.load(f)

        #------answer
        with open(answer_file,'r') as f:
            ans_json = json.load(f)

        #---list of question
        self.questions = que_json["questions"]

        self.ann_map = {
            ann["question_id"]: ann for ann in ans_json["annotations"]
        }

        self.img_dir = img_dir
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.ans2idx = ans2idx
        self.num_answers = len(ans2idx)

        

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]

        #image
        image_id = q["image_id"]
        image_name = f"{self.img_prefix}_{image_id:012d}.jpg"
        img_path = os.path.join(self.img_dir, image_name)
        

        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)

        #question
        question = q["question"]
        text = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=16,
            return_tensors="pt"
        )

        #answer
        ann = self.ann_map[q["question_id"]]
        
        # normalize all 10 answers
        answers = [normalize_answer(a["answer"]) for a in ann["answers"]]

        # build soft target for BCE
        soft_target = build_soft_target(
            answers,
            self.ans2idx,
            self.num_answers
        )

        # MCA label (for strict accuracy only)
        mca = normalize_answer(ann["multiple_choice_answer"])
        label = self.ans2idx.get(mca, self.ans2idx["<UNK>"])

        return image, text, soft_target, label, answers
    

from torch.utils.data.dataloader import default_collate

def vqa_collate_fn(batch):
    
    #  Unzip the batch: [(img1, txt1, ...), (img2, txt2, ...)] -> [(img1, img2), (txt1, txt2), ...]
    # batch is a list of tuples. zip(*batch) separates them by columns.
    transposed = list(zip(*batch))

    #  Use default_collate for all but the last element (answers)
    images = default_collate(transposed[0])
    text = default_collate(transposed[1])
    soft_target = default_collate(transposed[2])
    labels = default_collate(transposed[3])
    
    #keep it as a list of lists
    answers = list(transposed[4]) 

    return images, text, soft_target, labels, answers