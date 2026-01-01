import os
import json
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from models.vqa_model import VQAModel

# ---------------- CONFIG ---------------- #

DEMO_IMAGE_DIR = "demo_images"
MODEL_PATH = "checkpoints/model_only.pth"
VOCAB_PATH = "data/idx2ans_topK.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

DEMO_QUESTIONS = {
    "room.jpg": "Is any human sitting on the chair?",
    "train.jpg": "Which vehicle can you see?",
    "bottle.jpg": "Is this bear?",
    "dog.jpg" : "What is the dog holding in its mouth?",
    "ducks.jpg" : "How many swans can you see?",
    "human.jpg":"How many women are sitting?",
    
    # Default fallback for images not listed here
    "DEFAULT": "What is in this image?" 
}

# ---------------- LOAD RESOURCES ---------------- #

@st.cache_resource
def load_resources():
    with open(VOCAB_PATH, "r") as f:
        idx2ans = json.load(f)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = VQAModel(num_answers=len(idx2ans))
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, tokenizer, idx2ans


@st.cache_data
def load_demo_images():
    if not os.path.exists(DEMO_IMAGE_DIR):
        return []
    return sorted([
        f for f in os.listdir(DEMO_IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png","webp"))
    ])


model, tokenizer, idx2ans = load_resources()

# ---------------- PREPROCESSING ---------------- #

transform = transforms.Compose([
    # transforms.Resize(256),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="VQA Assistant", layout="wide")

st.title("🤖 Visual Question Answering (VQA)")
st.markdown("Select a demo image **or** upload your own, then ask a question.")

if "user_question" not in st.session_state:
    st.session_state["user_question"] = ""
if "last_selected_demo" not in st.session_state:
    st.session_state["last_selected_demo"] = None

# -------- 3 COLUMN LAYOUT -------- #

col1, col2, col3 = st.columns([1, 1.5, 1.3])

image_for_model = None
image_for_display = None
current_demo_selection = None

# -------- COLUMN 1: IMAGE SOURCE -------- #
with col1:
    st.subheader("Image Source")

    image_source = st.radio(
        "Choose input:",
        ["Demo images", "Upload image"]
    )

    if image_source == "Demo images":
        demo_images = load_demo_images()

        if demo_images:
            selected_demo = st.selectbox("Select demo image:", demo_images)
            current_demo_selection = selected_demo
            if selected_demo:
                image_path = os.path.join(DEMO_IMAGE_DIR, selected_demo)
                if os.path.exists(image_path):

                    image_for_display = image_path
                    image_for_model = Image.open(image_path).convert("RGB")
        else:
            st.warning("No demo images found.")

    else:
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png","webp"]
        )
        if uploaded_file:
            uploaded_file.seek(0)
            img_pil = Image.open(uploaded_file).convert("RGB")

            image_for_display = img_pil
            image_for_model = img_pil
            if st.session_state["last_selected_demo"] != "UPLOAD_MODE":
                 st.session_state["user_question"] = ""
                 st.session_state["last_selected_demo"] = "UPLOAD_MODE"

if image_source == "Demo images" and current_demo_selection:
    if current_demo_selection != st.session_state["last_selected_demo"]:
        # Get custom question or default
        new_question = DEMO_QUESTIONS.get(current_demo_selection, DEMO_QUESTIONS["DEFAULT"])
        st.session_state["user_question"] = new_question
        st.session_state["last_selected_demo"] = current_demo_selection

# -------- COLUMN 2: PREVIEW -------- #
with col2:
    st.subheader("Preview")
    if image_for_display is not None:
        st.image(image_for_display, caption="Selected Image", width="stretch")
    else:
        st.info("Select or upload an image.")

# -------- COLUMN 3: QUESTION + ANSWERS -------- #
with col3:
    st.subheader("Ask")

    question = st.text_input(
        "Enter your question:",
        key="user_question"
    )

    if st.button("Get Answer", width="stretch") and image_for_model and question.strip():

        img_tensor = transform(image_for_model).unsqueeze(0).to(device)
        inputs = tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=16,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(
                img_tensor,
                inputs["input_ids"].to(device),
                inputs["attention_mask"].to(device)
            )
            probs = torch.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, k=5, dim=1)

        st.markdown("### Predictions")
        for i in range(5):
            ans_idx = str(top_indices[0][i].item())
            answer = idx2ans.get(ans_idx, "Unknown")
            conf = top_probs[0][i].item() * 100

            if i == 0:
                st.success(f"**Top Answer:** {answer} ({conf:.2f}%)")
            else:
                st.write(f"{i+1}. {answer} — {conf:.2f}%")
