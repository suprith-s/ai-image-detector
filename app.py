from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from PIL import Image
import io

app = Flask(__name__)

# ✅ FORCE CPU (stable + low memory)
device = torch.device("cpu")

# ✅ LIMIT CPU THREADS (reduces RAM usage)
torch.set_num_threads(1)

# ✅ LOAD LIGHTWEIGHT MODEL
model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=2)
model.classifier = nn.Linear(model.classifier.in_features, 2)

# ✅ LOAD TRAINED WEIGHTS
model.load_state_dict(torch.load("ai_detector.pth", map_location=device))
model = model.to(device)
model.eval()

# ✅ MINIMAL TRANSFORM (fast)
transform = transforms.Compose([
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    if request.method == 'POST':
        file = request.files.get('image')

        if file:
            try:
                # ✅ READ IMAGE (NO DISK STORAGE)
                img = Image.open(io.BytesIO(file.read())).convert("RGB")

                # ✅ RESIZE EARLY (memory + speed optimization)
                img.thumbnail((224, 224))

                # ✅ CONVERT TO TENSOR
                img_tensor = transform(img).unsqueeze(0).to(device)

                # ✅ INFERENCE (FAST + LOW MEMORY)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = F.softmax(outputs, dim=1)[0]

                ai_prob = float(probs[0]) * 100
                real_prob = float(probs[1]) * 100

                result = {
                    "ai": round(ai_prob, 2),
                    "real": round(real_prob, 2),
                    "label": "AI Generated 🤖" if ai_prob > real_prob else "Real Image 📸"
                }

            except Exception as e:
                result = {"error": "Invalid image or processing error"}

    return render_template("index.html", result=result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
