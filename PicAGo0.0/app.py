from PIL import Image
from models.clip_model import get_clip_similarity
from models.blip_model import generate_answer

def main(image: Image.Image, question: str):
    # Use CLIP to check similarity (Optional - you can remove this if not needed)
    similarity_score = get_clip_similarity(image, question)
    print(f"CLIP similarity score: {similarity_score}")

    # Generate the answer using BLIP
    answer = generate_answer(image, question)
    print(f"Answer: {answer}")
    
    return answer

if __name__ == "__main__":
    # Example usage: Open an image and ask a question
    image = Image.open("path_to_image.jpg")  # Replace with your image input method
    question = "What is in the image?"

    # Call the main function
    main(image, question)
