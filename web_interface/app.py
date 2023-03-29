import os
import torch
import gradio as gr

from PIL import Image
from torchvision import transforms

from model import VGG16

"""
Built following:https://huggingface.co/spaces/hasibzunair/image-recognition-demo
"""

# Load PyTorch model
model = VGG16(num_classes=2)
checkpoint = torch.load("model.pth")
model.load_state_dict(checkpoint)
model.eval()

# Inference!
def inference(input_image_path):
    image = Image.open(input_image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        _, predictions = model.forward(input_batch)
    
    results = {}
    results["cat"] = predictions[0][0].item()
    results["dog"] = predictions[0][1].item()
    print(results)
    return results

# Define ins outs placeholders
inputs = gr.inputs.Image(type='filepath')
outputs = gr.outputs.Label(type="confidences",num_top_classes=2)

# Define style
title = "Dog and cat classifier"
description = "This is a demo of a dog and cat image classifier."

# Run inference
gr.Interface(inference, 
            inputs, 
            outputs, 
            examples=["example1.jpg", "example2.jpg"], 
            title=title, 
            description=description,
            analytics_enabled=False).launch()
