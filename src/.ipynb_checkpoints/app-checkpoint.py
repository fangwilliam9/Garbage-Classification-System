# src/app.py
import os
import time  # 用于计算预测时间
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 初始化 Flask 应用
app = Flask(__name__)

# 设置上传文件夹
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 载入训练好的模型
MODEL_PATH = "model_vgg16.keras"  # 我的模型路径
# 载入模型，不加载优化器
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# 重新编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# 定义类别标签
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 处理图片上传并进行预测
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 读取图片并预处理
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # 记录开始时间
            start_time = time.time()
            predictions = model.predict(img_array)
            end_time = time.time()  # 记录结束时间

            # 计算预测时间（毫秒）
            prediction_time = round((end_time - start_time) * 1000, 2)

            predicted_class = class_labels[np.argmax(predictions)]
            confidence = round(100 * np.max(predictions), 2)
            
            return jsonify({'class': predicted_class, 'confidence': confidence, 'time': prediction_time})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)