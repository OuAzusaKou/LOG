from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from datetime import datetime
import requests

app = Flask(__name__)

def save_image(image_data):
    # 创建保存图片的文件夹
    if not os.path.exists("captured_images"):
        os.makedirs("captured_images")
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_images/capture_{timestamp}.jpg"
    
    # 保存图片
    with open(filename, 'wb') as f:
        f.write(image_data)
    return filename

def analyze_image(image_path):
    # 这里应该调用您的图像分析API
    # 以下是一个示例实现,您需要替换为实际的API调用
    api_url = "https://your-image-analysis-api.com/analyze"
    with open(image_path, "rb") as image_file:
        response = requests.post(api_url, files={"image": image_file})
    return response.json()["description"]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400
    
    image_file = request.files['image']
    image_path = save_image(image_file.read())
    description = analyze_image(image_path)
    
    return jsonify({"description": description})

if __name__ == "__main__":
    app.run(debug=True)
