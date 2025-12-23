import re
import string
from collections import Counter


CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't",
    "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id": "I'd", "I'd've": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
    "itll": "it'll", "lets": "let's", "maam": "ma'am",
    "mightnt": "mightn't", "mightnt've": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't",
    "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock",
    "oughtnt": "oughtn't", "shant": "shan't",
    "shed": "she'd", "shed've": "she'd've", "shes": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "somebodyd": "somebody'd",
    "somebodyd've": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someone'd": "someone'd",
    "someone'd've": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed": "we'd",
    "wed've": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "whod've": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "wont": "won't", "wouldve": "would've",
    "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "yall'd've": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "youll": "you'll",
    "youre": "you're", "youve": "you've"
}

ARTICLES = {"a", "an", "the"}

NUMBER_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10"
}

PUNCT = re.compile(r"[{}]".format(re.escape(string.punctuation.replace("'", "").replace(":", ""))))


# --------------------------------------------------
# Official normalize_answer
# --------------------------------------------------
def normalize_answer(ans: str) -> str:
    ans = ans.lower().strip()

    # convert contractions
    ans_words = ans.split()
    ans_words = [CONTRACTIONS.get(w, w) for w in ans_words]
    ans = " ".join(ans_words)

    # remove punctuation except apostrophe and colon
    ans = PUNCT.sub(" ", ans)

    # convert number words to digits
    ans_words = ans.split()
    ans_words = [NUMBER_MAP.get(w, w) for w in ans_words]

    # remove articles
    ans_words = [w for w in ans_words if w not in ARTICLES]

    ans = " ".join(ans_words)

    # collapse whitespace
    ans = re.sub(r"\s+", " ", ans).strip()

    return ans






def coco_soft_accuracy(pred_answers, gt_answers_batch):
    """
    pred_answers: list[str]  (len = B)
    gt_answers_batch: list[list[str]] (len = B, each has 10 answers)

    returns: sum of soft accuracy over batch
    """
    acc = 0.0

    for pred, gt_answers in zip(pred_answers, gt_answers_batch):
        if pred == "<UNK>":
            continue
        cnt = Counter(gt_answers)
        acc += min(cnt[pred] / 3.0, 1.0)

    return acc
