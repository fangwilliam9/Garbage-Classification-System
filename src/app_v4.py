from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os

# 创建 Flask 应用
app = Flask(__name__)

# 创建上传文件夹
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载模型
model = keras.models.load_model("classification_model.keras")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 首页：提供上传图片的 HTML 页面
@app.route("/", methods=["GET"])
def home():
    return """
    <!doctype html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <title>图片分类</title>
    </head>
    <body>
        <h2>上传一张图片进行分类</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <input type="submit" value="上传并分类">
        </form>
    </body>
    </html>
    """

# 处理图片上传并进行预测
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "没有检测到上传的文件"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400

    try:
        # 读取并处理图片
        img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0  # 归一化
        img_array = np.expand_dims(img_array, axis=0)

        # 进行预测
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])

        # 保存上传的图片
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        img.save(save_path)

        return jsonify({
            "filename": file.filename,
            "saved_path": save_path,
            "predicted_class": class_names[class_idx],
            "confidence": confidence
        })
    
    except Exception as e:
        return jsonify({"error": f"处理图片时出错: {str(e)}"}), 500

# 启动服务器
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
