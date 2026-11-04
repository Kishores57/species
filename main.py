"""
TreeVisionAI — FastAPI Web API Server & Interactive UI
=====================================================

Provides REST API endpoints for leaf species prediction and serves
a modern interactive UI for species identification, native habitat
geographic mapping, and botanical insights.

Run locally with:
    uvicorn main:app --reload --port 8000
"""

import os
import io
import sys
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ExifTags
import torch
import torch.nn as nn
from torchvision import transforms

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import BEST_MODEL_PATH, get_device
from scripts.model import build_model
from scripts.dataset import get_val_transforms

# Detailed Botanical and Geographical Knowledge Base
SPECIES_DATABASE: Dict[str, Dict[str, Any]] = {
    "ashoka": {
        "display_name": "False Ashoka / Mast Tree",
        "scientific_name": "Polyalthia longifolia (var. pendula)",
        "family": "Annonaceae (Custard Apple family)",
        "found_in": "Native to Sri Lanka and Peninsular India. Widely introduced across tropical regions worldwide.",
        "common_habitats": "Urban avenue borders, institutional campuses, temple complexes, sound barriers, and landscaped gardens.",
        "leaf_characteristics": "Long, narrow lanceolate leaves with wavy (undulate) margins. Weeping, pendulous foliage structure.",
        "ecological_importance": "Highly effective at noise attenuation and particulate air pollution absorption in urban avenues.",
        "conservation_status": "Least Concern (Widely Cultivated)",
    },
    "banyan": {
        "display_name": "Indian Banyan",
        "scientific_name": "Ficus benghalensis",
        "family": "Moraceae (Fig or Mulberry family)",
        "found_in": "Native to the Indian Subcontinent (India, Pakistan, Bangladesh, Sri Lanka).",
        "common_habitats": "Village squares, roadsides, open deciduous forests, river valleys, and sacred temple groves.",
        "leaf_characteristics": "Broad, leathery elliptic-ovate leaves with prominent pale primary veins and velvety undersides when young.",
        "ecological_importance": "The National Tree of India; keystone ecological species supporting hundreds of frugivorous birds, bats, and insects.",
        "conservation_status": "Least Concern (Culturally Protected)",
    },
    "mango": {
        "display_name": "Mango Tree",
        "scientific_name": "Mangifera indica",
        "family": "Anacardiaceae (Sumac or Cashew family)",
        "found_in": "Native to tropical South Asia (Indo-Burma region and foothills of the Himalayas).",
        "common_habitats": "Agricultural orchards, home gardens, tropical lowland moist forests, and riverine plains.",
        "leaf_characteristics": "Lanceolate to oblong leaves with prominent midrib; tender leaves are reddish-bronze, maturing to glossy deep green.",
        "ecological_importance": "Produces the 'King of Fruits'; rich floral nectar provides critical food for native pollinators and biodiversity.",
        "conservation_status": "Vulnerable in wild state / Extensively Cultivated",
    },
    "neem": {
        "display_name": "Neem / Margosa",
        "scientific_name": "Azadirachta indica",
        "family": "Meliaceae (Mahogany family)",
        "found_in": "Native to the Indian Subcontinent and Southeast Asia (India, Nepal, Myanmar, Sri Lanka, Pakistan). Naturalized in tropical Africa and Australia.",
        "common_habitats": "Dry deciduous forests, arid/semi-arid plains, rural farm boundaries, roadsides, and village common lands.",
        "leaf_characteristics": "Pinnately compound leaves with 20–30 curved, sickle-shaped leaflets with sharply serrated (toothed) edges.",
        "ecological_importance": "Renowned as 'Nature's Pharmacy'; produces azadirachtin (natural biopesticide), tolerates severe droughts, and enriches nutrient-poor soils.",
        "conservation_status": "Least Concern (Highly Resilient)",
    },
    "peepal": {
        "display_name": "Sacred Fig / Peepal",
        "scientific_name": "Ficus religiosa",
        "family": "Moraceae (Fig family)",
        "found_in": "Native to the Indian Subcontinent, Indochina, and southwestern China.",
        "common_habitats": "Temples and shrines, rocky ledges, riverbanks, urban brickwork/walls (epiphytic start), and moist deciduous woodlands.",
        "leaf_characteristics": "Distinctive cordate (heart-shaped) leaves featuring an elongated, slender 'drip tip' that sheds rainwater efficiently.",
        "ecological_importance": "Revered as the Bodhi Tree in Buddhism; acts as a continuous oxygen generator and keystone shelter for native fauna.",
        "conservation_status": "Least Concern (Sacred & Protected)",
    },
    "raintree": {
        "display_name": "Rain Tree / Saman",
        "scientific_name": "Samanea saman",
        "family": "Fabaceae (Legume family)",
        "found_in": "Native to tropical Central and South America. Widely naturalized throughout South and Southeast Asia (India, Singapore, Indonesia).",
        "common_habitats": "Expansive municipal parks, highways, university grounds, and canal banks with ample crown clearance.",
        "leaf_characteristics": "Bipinnately compound leaves with diamond/rhomboid leaflets that fold inward at dusk, overcast skies, or during rainfall.",
        "ecological_importance": "Nitrogen-fixing root nodules enrich soil fertility; wide umbrella-like canopy creates microclimate cooling.",
        "conservation_status": "Least Concern (Widespread)",
    },
    "rubber": {
        "display_name": "Rubber Fig / Indian Rubber Tree",
        "scientific_name": "Ficus elastica",
        "family": "Moraceae (Fig family)",
        "found_in": "Native to eastern South Asia and Southeast Asia (Northeast India: Assam, Meghalaya; Myanmar, Malaysia, Indonesia).",
        "common_habitats": "Subtropical evergreen rainforests, moist foothills, limestone gorges, and grown globally as an ornamental indoor plant.",
        "leaf_characteristics": "Large, thick, glossy leathery oval leaves with fine parallel secondary veins and bright red stipule sheaths on emerging buds.",
        "ecological_importance": "Pliable, vigorous aerial root network famously braided by Meghalaya's indigenous Khasi communities into living root bridges.",
        "conservation_status": "Least Concern (Cultivated Globally)",
    },
}

app = FastAPI(
    title="TreeVisionAI API",
    description="Deep learning leaf species classification API for 7 native tree species.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = get_device()
model = None
class_names = ["ashoka", "banyan", "mango", "neem", "peepal", "raintree", "rubber"]
transform = get_val_transforms()


def extract_exif_gps(image: Image.Image) -> Optional[Dict[str, Any]]:
    """Extract GPS coordinates from image EXIF data if present."""
    try:
        exif = image._getexif()
        if not exif:
            return None

        gps_info = {}
        for key, val in exif.items():
            tag_name = ExifTags.TAGS.get(key, key)
            if tag_name == "GPSInfo":
                for sub_key, sub_val in val.items():
                    sub_tag_name = ExifTags.GPSTAGS.get(sub_key, sub_key)
                    gps_info[sub_tag_name] = sub_val

        if not gps_info:
            return None

        def to_degrees(dms):
            if isinstance(dms, (tuple, list)) and len(dms) >= 3:
                d = float(dms[0])
                m = float(dms[1])
                s = float(dms[2])
                return d + (m / 60.0) + (s / 3600.0)
            return float(dms)

        if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
            lat = to_degrees(gps_info["GPSLatitude"])
            if gps_info["GPSLatitudeRef"] in ["S", "s"]:
                lat = -lat

            lon = to_degrees(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef", "E") in ["W", "w"]:
                lon = -lon

            return {
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "maps_url": f"https://www.google.com/maps?q={lat:.5f},{lon:.5f}",
            }
    except Exception:
        return None
    return None


def load_inference_model():
    global model, class_names
    model_path = BEST_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pth")

    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "class_names" in checkpoint:
        class_names = checkpoint["class_names"]

    num_classes = checkpoint.get("num_classes", len(class_names))
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully from {model_path} with classes: {class_names}")


@app.on_event("startup")
def startup_event():
    load_inference_model()


@app.get("/api/status")
def get_status():
    return {
        "status": "ready" if model is not None else "model_not_loaded",
        "device": str(device),
        "architecture": "EfficientNet-B0 (Transfer Learning)",
        "supported_classes": class_names,
    }


@app.get("/api/species")
def get_all_species():
    return SPECIES_DATABASE


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        load_inference_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model could not be loaded.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Check for EXIF GPS tags in uploaded image
        exif_loc = extract_exif_gps(pil_image)

        # Tensor preprocessing for EfficientNet-B0
        tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=min(4, len(class_names)))

        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            key = class_names[idx.item()]
            info = SPECIES_DATABASE.get(key, {})
            results.append({
                "species_key": key,
                "display_name": info.get("display_name", key.capitalize()),
                "scientific_name": info.get("scientific_name", "Unknown"),
                "confidence": round(prob.item() * 100, 2),
                "family": info.get("family", "N/A"),
                "found_in": info.get("found_in", "N/A"),
                "common_habitats": info.get("common_habitats", "N/A"),
                "leaf_characteristics": info.get("leaf_characteristics", "N/A"),
                "ecological_importance": info.get("ecological_importance", "N/A"),
                "conservation_status": info.get("conservation_status", "N/A"),
            })

        best = results[0]

        return {
            "prediction": best["display_name"],
            "species_key": best["species_key"],
            "scientific_name": best["scientific_name"],
            "confidence": f"{best['confidence']}%",
            "confidence_value": best["confidence"],
            "family": best["family"],
            "found_in": best["found_in"],
            "common_habitats": best["common_habitats"],
            "leaf_characteristics": best["leaf_characteristics"],
            "ecological_importance": best["ecological_importance"],
            "conservation_status": best["conservation_status"],
            "photo_gps": exif_loc,
            "top_predictions": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# Mount UI static files
UI_DIR = os.path.join(PROJECT_ROOT, "ui")
if os.path.exists(UI_DIR):
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

@app.get("/")
def serve_home():
    index_file = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "TreeVisionAI API is live. UI index.html not found."}
