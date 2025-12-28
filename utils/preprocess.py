from torchvision import transforms
from transformers import BertTokenizer

image_transform = transforms.Compose(  # [3,224,224]
    [
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)



def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")
