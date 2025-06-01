import os
import requests
from flask import Flask, request, render_template_string
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

app = Flask(__name__)

IMGBB_API_KEY = "94467f7012732626ed8f9885fcb01408"

labels_map = {
    "plástico": ["botella de plástico", "envase plástico", "bolsa de plástico", "film plástico", "envase PET", "envase HDPE", "bolsa plástica"],
    "papel": ["hoja de papel", "periódico", "revista", "folleto", "papel de oficina", "papel de diario", "papel reciclado"],
    "cartón": ["caja de cartón", "empaque de cartón", "cartón corrugado", "cartón de embalaje", "caja para envío"],
    "tetra pak": ["envase de tetra pak", "caja de jugo tetra pak", "envase de leche tetra pak", "cartón combinado"],
    "aluminio": ["lata de aluminio", "envase metálico", "papel aluminio", "envase de bebidas", "hoja de aluminio"],
    "material peligroso": ["pila o batería usada", "producto químico peligroso", "residuo tóxico", "aceite usado", "producto inflamable", "residuo sanitario"],
    "vidrio": ["botella de vidrio", "vaso de vidrio", "frasco de vidrio", "envase de vidrio", "vidrio transparente", "vidrio coloreado"],
    "orgánico": ["restos de comida", "residuo orgánico", "cáscara de fruta", "hojas secas", "restos vegetales", "residuo biodegradable"],
    "otro": ["otro tipo de residuo", "material mixto", "residuo no clasificado", "plásticos duros", "residuo textil"]
}

all_labels = []
label_to_primary = {}
for primary, secondary_list in labels_map.items():
    for s in secondary_list:
        all_labels.append(s)
        label_to_primary[s] = primary

HTML_FORM = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Clasificador de Residuos</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #f0f4f8;
      margin: 0; padding: 0;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      min-height: 100vh;
      padding: 2rem;
    }
    .container {
      background: white;
      max-width: 480px;
      width: 100%;
      padding: 2rem 3rem;
      border-radius: 12px;
      box-shadow: 0 8px 16px rgb(0 0 0 / 0.1);
      text-align: center;
    }
    h1 {
      margin-bottom: 1.5rem;
      color: #2c3e50;
    }
    form input[type=file] {
      margin-bottom: 1rem;
      width: 100%;
      padding: 0.5rem;
      border: 2px solid #2980b9;
      border-radius: 6px;
      cursor: pointer;
      font-size: 1rem;
      transition: border-color 0.3s ease;
    }
    form input[type=file]:hover {
      border-color: #3498db;
    }
    form input[type=submit] {
      background-color: #2980b9;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 6px;
      color: white;
      font-weight: 600;
      font-size: 1.1rem;
      cursor: pointer;
      transition: background-color 0.3s ease;
    }
    form input[type=submit]:hover {
      background-color: #3498db;
    }
    ul {
      list-style: none;
      padding: 0;
      margin-top: 2rem;
      text-align: left;
    }
    ul li {
      background: #ecf0f1;
      margin-bottom: 0.5rem;
      padding: 0.6rem 1rem;
      border-radius: 8px;
      font-size: 1rem;
      color: #34495e;
      display: flex;
      justify-content: space-between;
      font-weight: 600;
    }
    ul li b {
      color: #2c3e50;
    }
    .result-image {
      margin-top: 2rem;
      max-width: 100%;
      border-radius: 12px;
      box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Clasificador de Residuos</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <input type="submit" value="Analizar">
    </form>
    {% if result %}
      <ul>
      {% for label, score in result %}
        <li><b>{{ label }}</b><span>{{ '%.4f'|format(score) }}</span></li>
      {% endfor %}
      </ul>
    {% endif %}
    {% if image_url %}
      <img class="result-image" src="{{ image_url }}" alt="Imagen subida" />
    {% endif %}
  </div>
</body>
</html>
"""

def upload_to_imgbb(image_file):
    url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": IMGBB_API_KEY,
    }
    files = {
        "image": image_file.read(),
    }
    response = requests.post(url, data=payload, files=files)
    if response.status_code == 200:
        return response.json()['data']['url']
    else:
        return None

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None
    if request.method == "POST":
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template_string(HTML_FORM, result=None, image_url=None)

        image_file = request.files["image"]

        image_url = upload_to_imgbb(image_file)
        if not image_url:
            return "Error al subir la imagen a imgbb"

        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        inputs = processor(text=all_labels, images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0].tolist()

        scores_primary = {}
        for label, score in zip(all_labels, probs):
            primary = label_to_primary[label]
            scores_primary[primary] = scores_primary.get(primary, 0) + score

        result = sorted(scores_primary.items(), key=lambda x: x[1], reverse=True)

    return render_template_string(HTML_FORM, result=result, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
