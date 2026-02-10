import os
import re
import torch
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification
)
from googleapiclient.discovery import build
from dotenv import load_dotenv

# ================= LOAD ENV =================
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
MODEL_PATH = "results/minilm_pytorch_quantized"
MAX_COMMENTS_ALLOWED = [100, 200, 500]

if not YOUTUBE_API_KEY:
    raise RuntimeError("❌ YOUTUBE_API_KEY not found in environment")

# ================= DEMO VIDEOS =================
DEMO_VIDEOS = [
    "dQw4w9WgXcQ",
    "9bZkp7q19f0",
    "3JZ_D3ELwOQ",
    "RgKAFK5djSk",
    "kJQP7kiw5Fk",
    "e-ORhEE9VVg",
    "l482T0yNkeo",
    "fRh_vgS2dFE",
    "OPf0YbXqDm0",
    "uelHwf8o7_U",
]

# ================= UTILS =================
def extract_video_id(url_or_id: str) -> str:
    """
    Accepts full YouTube URL or video ID and returns video ID
    """
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})"
    ]

    for p in patterns:
        match = re.search(p, url_or_id)
        if match:
            return match.group(1)

    # assume already an ID
    return url_or_id.strip()


def get_video_meta(video_id: str):
    youtube = build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY,
        cache_discovery=False
    )

    resp = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    if not resp["items"]:
        return None, None

    snippet = resp["items"][0]["snippet"]
    return snippet["title"], snippet["thumbnails"]["high"]["url"]


# ================= MODEL =================
class NativeQuantizedModel:
    def __init__(self, model_folder):
        self.tokenizer = AutoTokenizer.from_pretrained(model_folder)
        config = AutoConfig.from_pretrained(model_folder)

        model_fp32 = AutoModelForSequenceClassification.from_config(config)
        self.model = torch.quantization.quantize_dynamic(
            model_fp32, {torch.nn.Linear}, dtype=torch.qint8
        )

        state_dict = torch.load(
            f"{model_folder}/quantized_weights.pt",
            map_location="cpu"
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print("✅ Quantized model loaded")

    def predict(self, text: str):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        idx = torch.argmax(probs).item()
        score = probs[0][idx].item()

        label_map = self.model.config.id2label or {
            0: "Negative",
            1: "Positive"
        }

        return label_map[idx], round(score * 100, 2)


model = NativeQuantizedModel(MODEL_PATH)

# ================= YOUTUBE COMMENTS =================
def fetch_comments(video_id: str, max_comments: int):
    youtube = build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY,
        cache_discovery=False
    )

    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request and len(comments) < max_comments:
        response = request.execute()

        for item in response.get("items", []):
            comments.append(
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            )
            if len(comments) >= max_comments:
                break

        request = youtube.commentThreads().list_next(request, response)

    return comments


# ================= FASTAPI =================
app = FastAPI(title="YouTube Sentiment Analyzer")

@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("UI/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/demo-videos")
def demo_videos():
    videos = []
    for vid in DEMO_VIDEOS:
        title, thumb = get_video_meta(vid)
        videos.append({
            "video_id": vid,
            "title": title,
            "thumbnail": thumb
        })
    return videos


@app.get("/analyze")
def analyze_video(
    video: str = Query(..., description="Full YouTube URL or video ID"),
    limit: int = Query(100, description="100 / 200 / 500")
):
    if limit not in MAX_COMMENTS_ALLOWED:
        return JSONResponse(
            status_code=400,
            content={"error": "limit must be 100, 200, or 500"}
        )

    video_id = extract_video_id(video)
    title, thumbnail = get_video_meta(video_id)
 
    if not title:
        return {"error": "Invalid or unavailable video"}

    comments = fetch_comments(video_id, limit)

    if not comments:
        return {
            "video_id": video_id,
            "title": title,
            "thumbnail": thumbnail,
            "message": "No comments found"
        }

    pos, neg = 0, 0
    results = []

    for text in comments:
        label, score = model.predict(text)
        results.append({
            "text": text,
            "sentiment": label,
            "confidence": f"{score}%"
        })

        if label.lower().startswith("pos"):
            pos += 1
        else:
            neg += 1

    overall_confidence = round(
        (max(pos, neg) / len(results)) * 100, 2
    )

    return {
        "video_id": video_id,
        "title": title,
        "thumbnail": thumbnail,
        "total_comments": len(results),
        "positive": pos,
        "negative": neg,
        "overall_confidence": f"{overall_confidence}%",
        "results": results
    }
