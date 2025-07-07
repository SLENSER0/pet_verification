from pathlib import Path
from typing import List, Dict
import json

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModel, AutoProcessor
from collections import OrderedDict
import numpy as np
import base64
import io
import zipfile, os, uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

import cv2
from ultralytics import YOLO

import boto3
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, ClientError
from openai import OpenAI

import sqlite3
from typing import Dict, Optional
import json

# Add this to your global variables
DB_PATH = "pet_database.db"

def _init_database():
    """Initialize the database with required table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pets (
            pet_id TEXT PRIMARY KEY,
            lat REAL,
            lon REAL,
            text TEXT
        )
    ''')
    
    conn.commit()
    conn.close()


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in kilometers between two lat/lon points using Haversine formula."""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def _cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Return cosine distance (1 - cosine similarity) between two vectors."""
    return 1.0 - float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
def _get_average_embeddings_by_pet_id(target_embedding: List[float]) -> Dict[str, Dict]:
    """Get average embeddings for each pet_id from Qdrant."""
    try:
        scroll_result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000, 
            with_vectors=True
        )
        
        pet_embeddings = {}
        pet_info = {}
        
        for point in scroll_result[0]:
            pet_id = point.payload.get("pet_id")
            if pet_id:
                if pet_id not in pet_embeddings:
                    pet_embeddings[pet_id] = []
                    pet_info[pet_id] = {
                        "s3_key": point.payload.get("s3_key", ""),
                        "metadata": point.payload.get("metadata", "")
                    }
                pet_embeddings[pet_id].append(point.vector)
        
        result = {}
        for pet_id, embeddings in pet_embeddings.items():
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding /= np.linalg.norm(avg_embedding) + 1e-12   # renormalise
            distance = _cosine_distance(target_embedding, avg_embedding)
            result[pet_id] = {
                "average_embedding": avg_embedding.tolist(),
                "distance": float(distance),
                "info": pet_info[pet_id]
            }
        
        return result
    except Exception as e:
        print(f"Error getting average embeddings: {e}")
        return {}


def _get_pet_s3_keys(pet_id: str) -> List[str]:
    """Get all S3 keys for a given pet_id from Qdrant."""
    try:
        scroll_result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter={"must": [{"key": "pet_id", "match": {"value": pet_id}}]},
            limit=1000,
            with_payload=True
        )
        
        s3_keys = []
        for point in scroll_result[0]:
            s3_key = point.payload.get("s3_key")
            if s3_key:
                s3_keys.append(s3_key)
        
        return s3_keys
    except Exception as e:
        print(f"Error getting S3 keys for pet {pet_id}: {e}")
        return []


def _download_from_s3(s3_key: str) -> Optional[bytes]:
    """Download file from S3 and return bytes."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return response['Body'].read()
    except (NoCredentialsError, ClientError) as e:
        print(f"S3 download error for {s3_key}: {e}")
        return None


def _insert_pet_data(pet_id: str, lat: float, lon: float, text: str):
    """Insert or update pet data in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO pets (pet_id, lat, lon, text)
        VALUES (?, ?, ?, ?)
    ''', (pet_id, lat, lon, text))
    
    conn.commit()
    conn.close()

def _get_pet_data(pet_id: str) -> Optional[Dict]:
    """Retrieve pet data from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM pets WHERE pet_id = ?', (pet_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "pet_id": row[0],
            "lat": row[1],
            "lon": row[2],
            "text": row[3]
        }
    return None

QDRANT_HOST = "95.31.5.36"
QDRANT_PORT = 5173
COLLECTION_NAME = "pet_embeddings"
VECTOR_SIZE = 768

PET_MODEL_PATH = "/home/basil/airi_pet/weights/yolov12x.pt"
pet_model = None              
pet_ids = []                 


S3_ENDPOINT = "https://petss3ai.hb.ru-msk.vkcloud-storage.ru"
S3_ACCESS_KEY = "rpkopB5hFGq1kok7L6Z8bs"
S3_SECRET_KEY = "3mJt5mwauh1sYXYKfB9YZ23WVXqi3hatsMQ2CeTMjAsb"
S3_BUCKET_NAME = "petss3ai"

OPENAI_API_KEY = "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"
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"

qdrant_client: QdrantClient | None = None
s3_client = None
openai_client = None

def _init_s3_client():
    """Initialize S3 client with custom endpoint and signature configuration."""
    global s3_client
    
    config = Config(
        signature_version='s3v4',
        s3={
            'addressing_style': 'path'
        },
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required"
    )
    
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=config
    )
    
    s3_client.meta.events.register(
        'before-sign.s3.*',
        _add_unsigned_payload_header
    )

def _add_unsigned_payload_header(request, **kwargs):
    """Add x-amz-content-sha256: UNSIGNED-PAYLOAD header to all S3 requests."""
    request.headers['x-amz-content-sha256'] = 'UNSIGNED-PAYLOAD'

def _init_openai_client():
    """Initialize OpenAI client."""
    global openai_client
    openai_client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )

def _ensure_collection():
    """Create collection if it doesn't exist."""
    collections = qdrant_client.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": VECTOR_SIZE, "distance": "Cosine"},
        )

def _is_image(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))

def _init_pet_model():
    """Load YOLO model once and cache its cat/dog class IDs."""
    global pet_model, pet_ids
    pet_model = YOLO(PET_MODEL_PATH)
    names = pet_model.names
    pet_ids = [k for k, v in names.items() if v in ("cat", "dog")]
    print(f"YOLO model loaded with pet classes {pet_ids} → {[names[i] for i in pet_ids]}")

def _upload_to_s3(file_bytes: bytes, s3_key: str) -> bool:
    """Upload file bytes to S3."""
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_bytes
        )
        return True
    except (NoCredentialsError, ClientError) as e:
        print(f"S3 upload error: {e}")
        return False

def _analyze_pet_with_openai(image_bytes: bytes) -> str:
    """Analyze pet image using OpenAI API."""
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        completion = openai_client.chat.completions.create(        
            model="qwen/qwen2.5-vl-72b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Please analyze this image and provide information about the animal in the following exact format:

animal type: {cat or dog}
breed classify: {specific breed name}
age type: {baby or adult or senior}
description: {brief description of the animal's appearance and characteristics}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"OpenAI analysis error: {e}")
        return f"Analysis failed: {str(e)}"

class FaceRecognizer(torch.nn.Module):
    """Must be identical to the one used for training."""

    def __init__(self, ckpt_name: str = "zer0int/CLIP-GmP-ViT-L-14"):
        super().__init__()
        self.clip = AutoModel.from_pretrained(ckpt_name)
        self.processor = AutoProcessor.from_pretrained(ckpt_name)

    @torch.inference_mode()
    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.clip.device) for k, v in inputs.items()}
        feats = self.clip.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats

def preprocess(image_file: UploadFile) -> Image.Image:
    """Decode file-like object to a RGB PIL image."""
    try:
        img = Image.open(image_file.file).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file") from e

def load_weights(model: torch.nn.Module, ckpt_path: Path, map_location="cpu"):
    """Robustly load checkpoints saved via either Fabric or Lightning."""
    state = torch.load(ckpt_path, map_location=map_location)

    if isinstance(state, dict) and "model" in state:
        fabric_model = state["model"]
        if isinstance(fabric_model, torch.nn.Module):
            if isinstance(fabric_model, torch.nn.parallel.DistributedDataParallel):
                fabric_model = fabric_model.module
            model.load_state_dict(fabric_model.state_dict())
        elif isinstance(fabric_model, (dict, OrderedDict)):
            model.load_state_dict(fabric_model)
        else:
            raise TypeError(f"Unexpected type under 'model': {type(fabric_model)}")
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, (dict, OrderedDict)):
        model.load_state_dict(state)
    else:
        raise RuntimeError("Could not locate weights in checkpoint.")

    model.eval()

def _crop_pets_to_b64(image: Image.Image) -> list[dict]:
    """Run YOLO, return list of dicts with pet crops."""
    cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = pet_model(cv2_img)[0]
    crops_json = []

    for bbox, cls in zip(result.boxes.xyxy.cpu().numpy(),
                         result.boxes.cls.cpu().numpy()):
        if int(cls) not in pet_ids:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        crop_bgr = cv2_img[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)

        buf = io.BytesIO()
        pil_crop.save(buf, format="JPEG", quality=95)
        img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        crops_json.append({
            "label": pet_model.names[int(cls)],
            "image_b64": img_b64,
        })

    return crops_json

app = FastAPI(title="Pet Embedding API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = Path("clip_face_rec_epoch_005.pth")

@app.on_event("startup")
def startup_event():
    """Load model once when the server starts."""
    global model, qdrant_client
    model = FaceRecognizer().to(DEVICE)
    load_weights(model, CKPT_PATH, map_location=DEVICE)
    print(f"Model loaded on {DEVICE} with weights {CKPT_PATH}")

    _init_pet_model()
    _init_s3_client()
    _init_openai_client()
    _init_database()

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    _ensure_collection()
    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"S3 client initialized for bucket: {S3_BUCKET_NAME}")
    print("OpenAI client initialized")

@app.post("/embed")
async def embed_image(file: UploadFile = File(...)):
    """Return a ℓ2-normalized CLIP embedding for the uploaded image."""
    img = preprocess(file)
    with torch.inference_mode():
        emb = model([img])
    return JSONResponse({"embedding": emb.squeeze().tolist()})

@app.post("/crop_pets")
async def crop_pets(file: UploadFile = File(...)):
    """Detect cats & dogs in an uploaded image and return the cropped regions."""
    img = preprocess(file)
    crops = _crop_pets_to_b64(img)

    if not crops:
        return JSONResponse(
            status_code=200,
            content={"detail": "The cat or dog not found"}
        )

    return {"crops": crops}

@app.post("/upload_zip")
async def upload_zip_to_qdrant(file: UploadFile = File(...)):
    """
    Upload a ZIP archive whose folder names are pet IDs.
    Each image is embedded and stored in Qdrant, uploaded to S3, and analyzed with OpenAI.
    Also processes location JSON and creates pet database.
    """
    # Read ZIP into memory
    data = await file.read()
    zf = zipfile.ZipFile(io.BytesIO(data))

    points: list[PointStruct] = []
    pet_metadata: Dict[str, str] = {}
    processed_pets = set()
    s3_uploads = []
    location_data = {}
    
    # First, look for and process the JSON file
    for name in zf.namelist():
        if name.endswith('.json'):
            json_content = zf.read(name).decode('utf-8')
            location_data = json.loads(json_content)
            print(f"Loaded location data for {len(location_data)} pets")
            break
    
    # Process images
    for name in zf.namelist():
        print(name)
        if name.endswith("/") or not _is_image(name):
            continue
            
        pet_id = os.path.normpath(name).split(os.sep)[0]
        filename = os.path.basename(name)
        img_bytes = zf.read(name)
        
        print("uploading to s3")
        s3_key = f"{pet_id}/{filename}"
        if _upload_to_s3(img_bytes, s3_key):
            s3_uploads.append(s3_key)
        
        print("analyzing")
        if pet_id not in processed_pets:
            analysis = _analyze_pet_with_openai(img_bytes)
            pet_metadata[pet_id] = analysis
            processed_pets.add(pet_id)
        
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        print("inference")
        with torch.inference_mode():
            vec = model([pil_img]).squeeze().tolist()

        points.append(
            PointStruct(
                id=uuid.uuid4().int >> 64,
                vector=vec,
                payload={
                    "pet_id": pet_id,
                    "filename": filename,
                    "s3_key": s3_key,
                    "metadata": pet_metadata.get(pet_id, "")
                },
            )
        )

    if not points:
        raise HTTPException(status_code=400, detail="no valid images found")

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    
    database_entries = 0
    print(f"Processed pets: {processed_pets}")
    print(f"Location data keys: {list(location_data.keys())}")

    

    for pet_id in processed_pets:
        print(f"Checking pet_id: {pet_id}")
        if pet_id in location_data:
            lat = location_data[pet_id]["lat"]
            lon = location_data[pet_id]["lon"]
            text = pet_metadata.get(pet_id, "")
            
            _insert_pet_data(pet_id, lat, lon, text)
            database_entries += 1
            print(f"Inserted pet {pet_id} into database")
        else:
            print(f"Pet {pet_id} not found in location data")
    
    print(f"Total database entries: {database_entries}")
    
    return {
        "stored_vectors": len(points),
        "s3_uploads": len(s3_uploads),
        "analyzed_pets": len(pet_metadata),
        "database_entries": database_entries,
        "pet_metadata": pet_metadata
    }


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

@app.get("/pet/{pet_id}")
async def get_pet(pet_id: str):
    """Get pet data by ID."""
    pet_data = _get_pet_data(pet_id)
    if not pet_data:
        raise HTTPException(status_code=404, detail="Pet not found")
    return pet_data

@app.get("/pets")
async def get_all_pets():
    """Get all pets from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM pets')
    rows = cursor.fetchall()
    conn.close()
    
    pets = []
    for row in rows:
        pets.append({
            "pet_id": row[0],
            "lat": row[1],
            "lon": row[2],
            "text": row[3]
        })
    
    return {"pets": pets, "count": len(pets)}

@app.get("/pets/nearby")
async def get_nearby_pets(lat: float, lon: float, radius: float = 1.0):
    """Get pets within a certain radius (in km) of given coordinates."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT *, 
               (6371 * acos(cos(radians(?)) * cos(radians(lat)) * 
                           cos(radians(lon) - radians(?)) + 
                           sin(radians(?)) * sin(radians(lat)))) AS distance
        FROM pets
        HAVING distance < ?
        ORDER BY distance
    ''', (lat, lon, lat, radius))
    
    rows = cursor.fetchall()
    conn.close()
    
    nearby_pets = []
    for row in rows:
        nearby_pets.append({
            "pet_id": row[0],
            "lat": row[1],
            "lon": row[2],
            "text": row[3],
            "distance_km": row[4]
        })
    
    return {"pets": nearby_pets, "count": len(nearby_pets)}

@app.post("/search/nearest_pets")
async def get_k_nearest_pets(file: UploadFile = File(...), k: int = 5):
    """
    Get k nearest pet_ids by calculating distance to average embedding of each pet_id.
    Upload an image to find similar pets.
    """
    try:
        # Get embedding for uploaded image
        img = preprocess(file)
        with torch.inference_mode():
            target_embedding = model([img]).squeeze().tolist()
        
        # Get average embeddings for all pets
        pet_data = _get_average_embeddings_by_pet_id(target_embedding)
        
        if not pet_data:
            return JSONResponse(
                status_code=404,
                content={"detail": "No pets found in database"}
            )
        
        sorted_pets = sorted(pet_data.items(), key=lambda x: x[1]["distance"])[:k]
        
        result = []
        for pet_id, data in sorted_pets:
            pet_location = _get_pet_data(pet_id)
            
            result.append({
                "pet_id": pet_id,
                "distance": data["distance"],
                "metadata": data["info"]["metadata"],
                "s3_key": data["info"]["s3_key"],
                "location": pet_location
            })
        
        return {
            "nearest_pets": result,
            "count": len(result)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    

@app.post("/search/nearest_pets_radius")
async def get_k_nearest_pets_within_radius(
    lat: float,
    lon: float,
    radius: float,
    k: int,
    file: UploadFile = File(...)):
    """
    Get k nearest pet_ids within a geographic radius by calculating distance to average embedding.
    
    Args:
        file: Image file to find similar pets
        lat: Latitude of search center
        lon: Longitude of search center
        radius: Search radius in kilometers
        k: Number of nearest pets to return
    """
    try:
        # Get embedding for uploaded image
        img = preprocess(file)
        with torch.inference_mode():
            target_embedding = model([img]).squeeze().tolist()
        
        # Get average embeddings for all pets
        pet_data = _get_average_embeddings_by_pet_id(target_embedding)
        
        if not pet_data:
            return JSONResponse(
                status_code=404,
                content={"detail": "No pets found in database"}
            )
        
        # Filter by geographic radius
        filtered_pets = []
        for pet_id, data in pet_data.items():
            pet_location = _get_pet_data(pet_id)
            if pet_location:
                distance_km = _haversine_distance(
                    lat, lon, 
                    pet_location["lat"], pet_location["lon"]
                )
                if distance_km <= radius:
                    filtered_pets.append((pet_id, data, pet_location, distance_km))
        
        if not filtered_pets:
            return JSONResponse(
                status_code=404,
                content={"detail": f"No pets found within {radius}km radius"}
            )
        
        # Sort by embedding distance and get top k
        sorted_pets = sorted(filtered_pets, key=lambda x: x[1]["distance"])[:k]
        
        result = []
        for pet_id, data, location, geo_distance in sorted_pets:
            result.append({
                "pet_id": pet_id,
                "embedding_distance": data["distance"],
                "geographic_distance_km": geo_distance,
                "metadata": data["info"]["metadata"],
                "s3_key": data["info"]["s3_key"],
                "location": location
            })
        
        return {
            "nearest_pets": result,
            "count": len(result),
            "search_radius_km": radius,
            "search_center": {"lat": lat, "lon": lon}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/pet/{pet_id}/images")
async def get_pet_images(pet_id: str):
    """Get all images for a specific pet from S3."""
    try:
        # Get all S3 keys for this pet
        s3_keys = _get_pet_s3_keys(pet_id)
        
        if not s3_keys:
            raise HTTPException(status_code=404, detail="No images found for this pet")
        
        images = []
        for s3_key in s3_keys:
            image_bytes = _download_from_s3(s3_key)
            if image_bytes:
                # Convert to base64 for JSON response
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                images.append({
                    "s3_key": s3_key,
                    "filename": os.path.basename(s3_key),
                    "image_b64": image_b64
                })
        
        if not images:
            raise HTTPException(status_code=404, detail="Failed to download images for this pet")
        
        return {
            "pet_id": pet_id,
            "images": images,
            "count": len(images)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving images: {str(e)}")


@app.get("/image/{pet_id}/{filename}")
async def get_single_pet_image(pet_id: str, filename: str):
    """Get a single image for a pet by filename."""
    try:
        s3_key = f"{pet_id}/{filename}"
        image_bytes = _download_from_s3(s3_key)
        
        if not image_bytes:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Convert to base64 for JSON response
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return {
            "pet_id": pet_id,
            "filename": filename,
            "s3_key": s3_key,
            "image_b64": image_b64
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving image: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("face_embed_api:app", host="0.0.0.0", port=8000, reload=False)
