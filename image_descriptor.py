import requests
import base64
from PIL import Image
import io
import argparse

def encode_image_to_base64(image_path):
    """Convert image to base64 string."""
    with Image.open(image_path) as img:
        # Convert image to RGB if it's not
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Resize image if it's too large (optional)
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Convert to base64
        return base64.b64encode(img_byte_arr).decode('utf-8')

def describe_image(image_path):
    """Describe image using Ollama's Gemma 3:4b model."""
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the prompt
    prompt = "Please describe this image in detail. Focus on the main subjects, colors, composition, and any notable elements."
    
    # Prepare the request payload
    payload = {
        "model": "gemma3:4b",
        "prompt": prompt,
        "images": [base64_image],
        "stream": False
    }
    
    # Make request to Ollama API
    try:
        response = requests.post('http://localhost:11434/api/generate', json=payload)
        response.raise_for_status()
        result = response.json()
        return result['response']
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Describe an image using Ollama Gemma 3:4b')
    parser.add_argument('image_path', help='Path to the image file')
    args = parser.parse_args()
    
    description = describe_image(args.image_path)
    print("\nImage Description:")
    print("-" * 50)
    print(description)
    print("-" * 50)

if __name__ == "__main__":
    main() 