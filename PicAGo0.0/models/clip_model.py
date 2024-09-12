import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_clip_similarity(image: Image.Image, question: str):
    # Preprocess the image (accepts PIL image)
    image = preprocess(image).unsqueeze(0).to(device)

    # Tokenize the question
    text_tokens = clip.tokenize([question]).to(device)

    # Encode image and text
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    return similarity.item()
