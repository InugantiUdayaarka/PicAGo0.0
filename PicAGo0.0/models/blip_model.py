from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to(device)

def generate_answer(image: Image.Image, question: str):
    # Prepare the inputs for BLIP
    inputs = processor(image, question, return_tensors="pt").to(device)

    # Generate the answer
    with torch.no_grad():
        output = model.generate(**inputs)
    
    # Decode the answer
    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer
