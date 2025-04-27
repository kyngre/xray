from flask import Flask, request, jsonify,send_from_directory
import torch
import mlflow.pytorch
from PIL import Image
import io
import torchvision.transforms as transforms
from flask_cors import CORS
import os
import gradcam_return
import cv2
import base64
import uuid


# ===== 기본 세팅 =====
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload/", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 파일 저장
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # GradCAM 결과 생성
    pred,gradcam_result = gradcam_return.predict_and_visualize_advanced(save_path)

    # 결과 저장
    result_filename = f"{uuid.uuid4().hex}.png"
    result_path = os.path.join(UPLOAD_FOLDER, result_filename)
    cv2.imwrite(result_path, gradcam_result)

    # 결과 경로 리턴
    return jsonify({
        "result": pred,
        "gradcam_path": f"uploads/{result_filename}"  # 상대 경로 리턴
    }), 200

    
# 업로드된 파일 제공하는 라우트
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ===== 서버 실행 =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
