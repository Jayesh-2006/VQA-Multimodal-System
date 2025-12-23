
# BUILD ANSWER VOCAB (COCO)
import json
from collections import Counter
from utils.answer_utils import normalize_answer

#file paths
ANNOTATIONS_JSON = "data/annotations/train.json"
TOP_K = 2000                
OUT_ANS2IDX = "data/ans2idx_topK.json"
OUT_IDX2ANS = "data/idx2ans_topK.json"


# LOAD ANNOTATIONS

with open(ANNOTATIONS_JSON, "r") as f:
    ann_data = json.load(f)

annotations = ann_data["annotations"]
print("Total annotations:", len(annotations))


# COUNT ANSWERS
counter = Counter()

for ann in annotations:
    for a in ann["answers"]:               # all 10 answers
        ans = normalize_answer(a["answer"])
        counter[ans] += 1

print("Unique normalized answers:", len(counter))


# BUILD TOP-K

most_common = counter.most_common(TOP_K)

ans2idx = {"<UNK>": 0}
idx2ans = {0: "<UNK>"}

for i, (ans, _) in enumerate(most_common, start=1):
    ans2idx[ans] = i
    idx2ans[i] = ans

print("Final vocab size (with UNK):", len(ans2idx))


# SAVE FILES
with open(OUT_ANS2IDX, "w") as f:
    json.dump(ans2idx, f, indent=2)

with open(OUT_IDX2ANS, "w") as f:
    json.dump(idx2ans, f, indent=2)

print("Saved:")
print(" -", OUT_ANS2IDX)
print(" -", OUT_IDX2ANS)


