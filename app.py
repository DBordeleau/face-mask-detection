import pathlib
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from torch.autograd import Variable
import io
import torch
import torch.nn.functional as F 
from torchvision import transforms
from PIL import Image

# Load model with `map_location` set to CPU (or CUDA if GPU is available)
app = Flask(__name__)
CORS(app)
model = torch.jit.load('model_v2_scripted.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()

classes = ['mask_weared_incorrect', 'with_mask', 'without_mask']

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    image_tensor = transform_image(image_bytes=image_bytes)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
        
    input = Variable(image_tensor)
    output = model(input)

    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1).data.cpu().numpy()[0]
    
    # Get the index of the predicted class
    index = probabilities.argmax()

    prediction = classes[index]
    confidence = probabilities[index] 

    confidence_percentage = confidence * 100
    
    return prediction, confidence_percentage

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        prediction, confidence = get_prediction(image_bytes=img_bytes)
        return jsonify({'prediction': prediction, 'confidence': confidence})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)