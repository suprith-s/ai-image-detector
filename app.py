import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from PIL import Image
import matplotlib.pyplot as plt
import io

# Device
device = torch.device("cpu")

# Load model
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load("ai_detector.pth", map_location=device))
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Prediction function with histogram
def predict_dashboard(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    ai_prob = float(probs[0]) * 100
    real_prob = float(probs[1]) * 100
    label = "AI Generated 🤖" if ai_prob > real_prob else "Real Image 📸"

    # Create histogram
    fig, ax = plt.subplots(figsize=(3,2))
    ax.bar(["AI Generated", "Real Image"], [ai_prob, real_prob], color=["#ff4b5c", "#00ff99"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Confidence %")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    hist_img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)

    return label, ai_prob, real_prob, image, hist_img

# Custom CSS for nice card layout
custom_css = """
body, .app, .gradio-container {
    background: transparent !important;
}

html {
    background: url('https://images.unsplash.com/photo-1579548122080-c35fd6820ecb?q=80&w=1170&auto=format&fit=crop') 
                no-repeat center center fixed;
    background-size: cover;
}

body {
    background: url('https://images.unsplash.com/photo-1579548122080-c35fd6820ecb?q=80&w=1170&auto=format&fit=crop') 
                no-repeat center center fixed;
    background-size: cover;
}

/* This is IMPORTANT for Gradio */
.gradio-container {
    max-width: 1000px;
    margin: auto;
    background-color: rgba(0,0,0,0.65);
    padding: 25px;
    border-radius: 20px;
    color: white;
    font-family: 'Inter', sans-serif;
}

/* Force app wrapper transparency */
.app {
    background: transparent !important;
}

/* Buttons */
.gr-button { 
    background: linear-gradient(90deg, #fff700, #00ccff); 
    color: black; 
    font-weight: bold;
    border-radius: 8px;
}
"""
# Build interactive dashboard
with gr.Blocks(css=custom_css, title="AI Image Detection Dashboard") as demo:
    gr.Markdown("<h1 style='text-align:center'>AI IMAGE DETECTION SYSTEM</h1>")
    gr.Markdown("<p style='text-align:center'>⚠️ This tool may not be 100% accurate.</p>")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="📤 Upload Image")
            predict_btn = gr.Button("Analyze")
        with gr.Column():
            label_output = gr.Label(label="Prediction")
            ai_bar = gr.Number(label="AI Generated %")
            real_bar = gr.Number(label="Real Image %")
            histogram_output = gr.Image(label="Confidence Histogram", type="pil")

    predict_btn.click(
        fn=predict_dashboard,
        inputs=image_input,
        outputs=[label_output, ai_bar, real_bar, histogram_output]
    )

demo.launch()
