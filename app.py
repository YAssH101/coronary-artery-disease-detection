import os
import io
import base64
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm as mpl_cm
from flask import Flask, render_template, request, jsonify
from PIL import Image

from model_arch import build_model
from preprocess import infer_transform

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB max upload

MODEL_PATH         = "model/cad_model.pth"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None


class GradCAM:
    """Grad-CAM on EfficientNet-B0's last feature block (features[-1])."""

    def __init__(self, m):
        self._acts = self._grads = None
        layer = m.features[-1]
        self._fh = layer.register_forward_hook(self._save_acts)
        self._bh = layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _, __, out):   self._acts  = out
    def _save_grads(self, _, __, go):   self._grads = go[0]

    def generate(self, tensor):
        model.eval()
        out = model(tensor)
        model.zero_grad()
        out.backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._acts).sum(dim=1).squeeze())
        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-8)
        return cam.cpu().detach().numpy()

    def remove(self):
        self._fh.remove()
        self._bh.remove()


def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = build_model().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"[WARNING] Model not found at {MODEL_PATH}. Run train on Kaggle first.")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    tensor = infer_transform(img).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor.to(DEVICE)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("files")
    files = [f for f in files if f.filename != "" and allowed_file(f.filename)]

    if not files:
        return jsonify({"error": "No valid image files provided."}), 400

    if model is None:
        return jsonify({"error": "Model not loaded. Add cad_model.pth to model/."}), 503

    per_image = []
    for file in files:
        try:
            tensor = preprocess_image(file.read())
            with torch.no_grad():
                prob_sick = float(model(tensor).squeeze())
            prob_normal = 1.0 - prob_sick
            per_image.append({
                "filename":          file.filename,
                "prediction":        "CAD Detected" if prob_sick >= 0.5 else "Normal",
                "probability_sick":  round(prob_sick * 100, 1),
                "probability_normal": round(prob_normal * 100, 1),
            })
        except Exception as e:
            per_image.append({"filename": file.filename, "error": str(e)})

    valid = [r for r in per_image if "error" not in r]
    if not valid:
        return jsonify({"error": "All images failed to process."}), 500

    avg_prob_sick = float(np.mean([r["probability_sick"] for r in valid])) / 100
    avg_prob_normal = 1.0 - avg_prob_sick
    cad_count = sum(1 for r in valid if r["prediction"] == "CAD Detected")

    if avg_prob_sick >= 0.5:
        label      = "CAD Detected"
        confidence = avg_prob_sick
        risk       = "High" if avg_prob_sick >= 0.75 else "Moderate"
    else:
        label      = "Normal"
        confidence = avg_prob_normal
        risk       = "Low"

    return jsonify({
        "aggregate": {
            "prediction":        label,
            "risk_level":        risk,
            "confidence":        round(confidence * 100, 1),
            "probability_sick":  round(avg_prob_sick * 100, 1),
            "probability_normal": round(avg_prob_normal * 100, 1),
            "total_slices":      len(valid),
            "cad_slices":        cad_count,
            "normal_slices":     len(valid) - cad_count,
        },
        "per_image": per_image,
    })


@app.route("/gradcam", methods=["POST"])
def gradcam():
    f = request.files.get("file")
    if not f or not allowed_file(f.filename):
        return jsonify({"error": "No valid image."}), 400
    if model is None:
        return jsonify({"error": "Model not loaded."}), 503

    raw = f.read()

    # Original image for overlay (RGB so we can blend color heatmap)
    orig = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_w, orig_h = orig.size

    # Preprocess for model — enable grad so backward pass works through frozen layers
    tensor = preprocess_image(raw).requires_grad_(True)

    gc = GradCAM(model)
    try:
        cam = gc.generate(tensor)       # shape: (7, 7) for EfficientNet-B0
    finally:
        gc.remove()
        model.zero_grad()

    # Resize CAM to original image dimensions
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.BILINEAR
    )
    cam_np = np.array(cam_pil) / 255.0

    # Apply jet colormap and blend with grayscale MRI shown as RGB
    heatmap_rgb = (mpl_cm.jet(cam_np)[:, :, :3] * 255).astype(np.uint8)
    blended = Image.blend(orig, Image.fromarray(heatmap_rgb), alpha=0.45)

    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"heatmap": f"data:image/png;base64,{b64}"})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
