from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# 加载训练好的模型
MODEL_PATH = "classification_model.keras"  # 记得换成你的模型路径
model = tf.keras.models.load_model(MODEL_PATH)

# 类别索引映射
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 主页（上传图片界面）
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # 需要创建一个 index.html 页面

# 预测 API
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((224, 224))  # 这里改成你的模型输入大小
        image = np.array(image) / 255.0  # 归一化
        image = np.expand_dims(image, axis=0)  # 扩展维度

        predictions = model.predict(image)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return jsonify({"class": predicted_class, "confidence": confidence})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
