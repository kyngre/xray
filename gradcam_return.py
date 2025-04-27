import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import mlflow.pytorch  # MLflow PyTorch 모듈 추가

# MLflow 서버 설정
mlflow.set_tracking_uri("http://10.125.208.184:5000")

# 0. Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. MLflow에서 모델 로드
model_name = "efficientnetb4"
model_stage = "Production"
model_uri = f"models:/{model_name}/{model_stage}"
model = mlflow.pytorch.load_model(model_uri)

# 2. device 이동 + eval 설정
model = model.to(device)
model.eval()
print("\u2705 모델 로딩 완료")

# 이미지 전처리 함수
def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor()
    ])
    return transform(img)

# GradCAM 클래스
def forward_hook(module, input, output):
    forward_hook.activations = output.detach()

def backward_hook(module, grad_in, grad_out):
    backward_hook.gradients = grad_out[0].detach()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self):
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x, target_class=None):
        out = self.model(x)
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        self.model.zero_grad()
        target = out[:, target_class]
        target.backward()

        w = backward_hook.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (w * forward_hook.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (x.size(2), x.size(3)))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# 간단한 GradCAM 시각화 + 결과값 리턴
def predict_and_visualize(img_path):
    img = Image.open(img_path)
    tensor = transform_image(img).unsqueeze(0).to(device)

    cam_gen = GradCAM(model, model.features[-1])
    cam = cam_gen.generate(tensor)

    output = model(tensor)
    probs = torch.softmax(output, dim=1)

    abnormal_prob = probs[0, 0].item()
    normal_prob = probs[0, 1].item()
    pred_class = 0 if abnormal_prob > 0.5 else 1
    pred_label = "Abnormal" if pred_class == 0 else "Normal"

    orig = np.array(img.resize((380, 380)))
    if orig.ndim == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)

    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig, 0.5, heat, 0.5, 0)

    return pred_label, overlay

# 고급 GradCAM 시각화 + 결과값 리턴
def predict_and_visualize_advanced(img_path):
    img = Image.open(img_path)
    tensor = transform_image(img).unsqueeze(0).to(device)

    cam_gen = GradCAM(model, model.features[-1])
    cam = cam_gen.generate(tensor)

    output = model(tensor)
    probs = torch.softmax(output, dim=1)

    abnormal_prob = probs[0, 0].item()
    normal_prob = probs[0, 1].item()
    pred_class = 0 if abnormal_prob > 0.5 else 1
    pred_label = "Abnormal" if pred_class == 0 else "Normal"

    orig = np.array(img.resize((380, 380)))
    if orig.ndim == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)

    threshold = 0.4
    cam_thresholded = np.where(cam > threshold, cam, 0)

    # 🔥 Canny 엣지
    cam_edges = cv2.Canny(np.uint8(cam_thresholded * 255), 50, 150)

    # 🔥 (추가) 엣지 굵게 (dilate)
    kernel = np.ones((4, 4), np.uint8)  # 3x3 커널
    cam_edges = cv2.dilate(cam_edges, kernel, iterations=1)

    # 🔥 RGB 변환 + 빨간색 강조
    cam_edges = cv2.cvtColor(cam_edges, cv2.COLOR_GRAY2RGB)
    cam_edges[:, :, 1:] = 0  # 빨간색만 남기기

    # 🔥 오버레이
    overlay = cv2.addWeighted(orig, 0.7, cam_edges, 1.2, 0)

    return pred_label, overlay
