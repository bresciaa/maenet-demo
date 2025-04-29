import os
import io
import base64
import torch
import torchvision.transforms.functional as TF

from flask import Blueprint, request, jsonify
from .model import load_model
from .utils import preprocess_image

main = Blueprint("main", __name__)
model = load_model()

def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@main.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    input_tensor = preprocess_image(file)

    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output[0, 0])
        binary_mask = (mask > 0.5).byte() * 255

    input_image = TF.to_pil_image(input_tensor.squeeze(0))
    mask_image = TF.to_pil_image(binary_mask)

    file_name_without_extension = os.path.splitext(file.filename)[0]

    return jsonify({
        "file_name": file_name_without_extension,
        "input_image": image_to_base64(input_image),
        "mask_image": image_to_base64(mask_image)
    })
