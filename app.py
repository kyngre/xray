from flask import Flask, request, jsonify
import torch
import mlflow.pytorch
from PIL import Image
import io
import torchvision.transforms as transforms
from flask_cors import CORS

# ===== 기본 세팅 =====
app = Flask(__name__)
CORS(app)

# device 세팅
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
mlflow.set_tracking_uri("http://10.125.208.184:5000")
model_name = "VIT_2_Model"
model_stage = "Production"   # 또는 "Staging" 가능

model_uri = f"models:/{model_name}/{model_stage}"
model = mlflow.pytorch.load_model(model_uri)
model = model.to(device)
model.eval()

# transform 정의 (※ 모델 입력 사이즈에 맞춰서 수정)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== API 엔드포인트 =====
# 클래스 라벨 매핑
label_map = {0: "비정상", 1: "정상"}

@app.route("/upload/", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    img_bytes = file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        logits = outputs.logits
        preds = logits.argmax(dim=1).cpu().item()
    
    label = label_map[preds]
    print(f"Predicted label: {label}")
    return jsonify({"prediction": label})

# ===== 서버 실행 =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
