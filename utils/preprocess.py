from torchvision import transforms
from transformers import AutoTokenizer



image_transform = transforms.Compose(  # [3,224,224]
    [
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)



def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-base"
    )

    return tokenizer
