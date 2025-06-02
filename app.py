from flask import Flask, request, jsonify
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import io

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

# Preparación
all_labels = []
label_to_primary = {}
for primary, secondary_list in labels_map.items():
    for s in secondary_list:
        all_labels.append(s)
        label_to_primary[s] = primary

# Carga del modelo
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

@app.route("/", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return "Falta imagen", 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

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
    best_label = result[0][0]

    return best_label  # ← texto plano, ideal para la app

if __name__ == "__main__":
    app.run(debug=True)
