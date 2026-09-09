from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI(title="SentinelX AI API")

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LIVE SYSTEM STATE
# ============================================================

system_state = {
    "ai_engine": "ready",
    "camera": "not connected",
    "watchlist": "ready",
    "stream": "offline",
    "last_update": None,
}

# ============================================================
# CAMERA STATE
# ============================================================

camera_state = {
    "camera_id": 0,
    "name": "Laptop Camera",
    "status": "offline",
    "detections": 0,
    "faces": 0,
    "recognized": 0,
    "unknown": 0,
    "fps": 0.0,
}

# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "system": "SentinelX",
        "status": "online"
    }


# ============================================================
# SYSTEM STATUS
# ============================================================

@app.get("/api/status")
def status():
    return system_state


# ============================================================
# CAMERA STATUS
# ============================================================

@app.get("/api/cameras")
def cameras():
    return {
        "cameras": [camera_state]
    }


# ============================================================
# LIVE CAMERA DATA
# ============================================================

@app.get("/api/camera/live")
def camera_live():
    return {
        "camera": camera_state,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# TEST / UPDATE CAMERA DATA
# ============================================================

@app.post("/api/camera/update")
def update_camera(data: dict):
    allowed_fields = [
        "camera_id",
        "name",
        "status",
        "detections",
        "faces",
        "recognized",
        "unknown",
        "fps",
    ]

    for field in allowed_fields:
        if field in data:
            camera_state[field] = data[field]

    system_state["camera"] = (
        "connected"
        if camera_state["status"] == "online"
        else "not connected"
    )

    system_state["stream"] = (
        "live"
        if camera_state["status"] == "online"
        else "offline"
    )

    system_state["last_update"] = datetime.now().isoformat()

    return {
        "success": True,
        "camera": camera_state,
        "system": system_state
    }


# ============================================================
# AI ENGINE STATUS
# ============================================================

@app.get("/api/ai")
def ai_status():
    return {
        "ai_engine": system_state["ai_engine"],
        "yolo": "ready",
        "bytetrack": "ready",
        "face_recognition": "ready",
        "watchlist": system_state["watchlist"],
    }