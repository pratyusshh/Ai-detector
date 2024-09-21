from flask import Flask, request, jsonify
from transformers import pipeline
import torch
from torchvision import models, transforms
from PIL import Image
import google.protobuf
print(google.protobuf.__version__)
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  



text_model = pipeline("text-classification", model="microsoft/deberta-v3-base") 

image_model = models.resnet50(pretrained=True)
image_model.eval()


def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)
def predict_text(text):
    result = text_model(text)

    ai_generated_percentage = result[0]['score'] * 100 if result[0]['label'] == 'LABEL_1' else (1 - result[0]['score']) * 100
    return ai_generated_percentage

def predict_image(image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = image_model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    
    ai_generated_class = probabilities.argmax().item()
    ai_generated_percentage = probabilities[ai_generated_class].item() * 100

    return ai_generated_percentage

@app.route('/predict-text', methods=['POST'])
def predict_text_route():
    data = request.get_json()
    text = data['text']
    ai_generated_percentage = predict_text(text)
    return jsonify({"ai_generated_percentage": ai_generated_percentage})

@app.route('/predict-image', methods=['POST'])
def predict_image_route():
    file = request.files['image']
    image = Image.open(file.stream)  
    ai_generated_percentage = predict_image(image)
    return jsonify({"ai_generated_percentage": ai_generated_percentage})

if __name__ == "__main__":
    app.run(debug=False)
