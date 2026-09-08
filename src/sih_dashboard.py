import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from insightface.app import FaceAnalysis

import json
# ============================================================
# PHASE 3B-1 VIRTUAL BORDER
# ============================================================
# B = enter border setup mode
# Left-click twice = define the border
# R = reset/redefine the border
# Q = quit
#
# Phase 3B-1 only draws the border.
# Actual crossing detection is Phase 3B-2.

border_points = []
border_setup_mode = False


def border_mouse_callback(event, x, y, flags, param):
    global border_points
    global border_setup_mode

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if not border_setup_mode:
        return

    border_points.append((int(x), int(y)))

    if len(border_points) == 2:
        border_setup_mode = False

        print()
        print("=" * 60)
        print("VIRTUAL BORDER DEFINED")
        print("Point 1:", border_points[0])
        print("Point 2:", border_points[1])
        print("Ready for Phase 3B-2 crossing detection.")
        print("=" * 60)


def reset_virtual_border():
    global border_points
    global border_setup_mode

    border_points = []
    border_setup_mode = True

    print()
    print("=" * 60)
    print("BORDER SETUP MODE")
    print("Click the first border point.")
    print("Then click the second border point.")
    print("=" * 60)


import os
import time
from datetime import datetime

# ============================================================
# SIH CCTV MONITORING SYSTEM
# YOLO + BYTETRACK + INSIGHTFACE
# ============================================================

from pathlib import Path
print("=" * 70)
print("              SIH CCTV MONITORING SYSTEM")
print("=" * 70)

# ============================================================
# SETTINGS
# ============================================================

print("\n========================================")
print("       SIH DASHBOARD CAMERA SELECT")
print("========================================")
print("[0] Laptop Camera")
print("[1] External USB Webcam")
print("[2] Other Camera")
choice=input("Enter camera number: ").strip()
CAMERA_ID=int(choice) if choice.isdigit() and int(choice) >= 0 else 0
print(f"Selected Camera: {CAMERA_ID}")

YOLO_MODEL = "yolo11n.pt"


DATABASE_FILE = "data/face_database.npz"

WATCHLIST_FOLDER = "data/watchlist"
WATCHLIST_ALERT_LABEL = "ISIS TERRORIST ALERT"

WATCHLIST_CHECK_INTERVAL = 2

# ============================================================
# WATCHLIST EVENT / SCREENSHOT STORAGE
# ============================================================

WATCHLIST_EVENT_COOLDOWN = 15

ALERTS_FOLDER = Path("data") / "alerts"
WATCHLIST_SCREENSHOT_FOLDER = (
    ALERTS_FOLDER / "watchlist_screenshots"
)
WATCHLIST_HISTORY_FILE = (
    ALERTS_FOLDER / "watchlist_history.jsonl"
)

ALERTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

WATCHLIST_SCREENSHOT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


def create_watchlist_event(
    frame,
    name,
    confidence,
    camera_id,
    track_id=None,
    bbox=None,
):
    # Prevent the same person from generating an event every frame.
    now = time.time()

    event_key = (
        f"{camera_id}:"
        f"{track_id if track_id is not None else name}"
    )

    previous = last_watchlist_event_times.get(
        event_key,
        0
    )

    if now - previous < WATCHLIST_EVENT_COOLDOWN:
        return False

    timestamp = datetime.now()

    stamp = timestamp.strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in str(name)
    )

    screenshot_path = (
        WATCHLIST_SCREENSHOT_FOLDER
        / (
            f"watchlist_{safe_name}_"
            f"camera_{camera_id}_"
            f"{stamp}.jpg"
        )
    )

    # Save the full CCTV frame.
    saved = cv2.imwrite(
        str(screenshot_path),
        frame
    )

    event = {
        "event": "WATCHLIST_ALERT",
        "timestamp": timestamp.isoformat(
            timespec="milliseconds"
        ),
        "camera_id": int(camera_id),
        "track_id": (
            int(track_id)
            if track_id is not None
            else None
        ),
        "person": str(name),
        "confidence": round(
            float(confidence),
            2
        ),
        "screenshot": str(
            screenshot_path
        ),
    }

    import json

    with open(
        WATCHLIST_HISTORY_FILE,
        "a",
        encoding="utf-8"
    ) as history_file:
        history_file.write(
            json.dumps(
                event,
                ensure_ascii=False
            ) + "\n"
        )

    last_watchlist_event_times[
        event_key
    ] = now

    print()
    print("=" * 60)
    print("WATCHLIST ALERT EVENT")
    print("Person     :", name)
    print(
        "Confidence :",
        f"{float(confidence):.1f}%"
    )
    print("Camera     :", camera_id)
    print("Track ID   :", track_id)
    print(
        "Screenshot :",
        screenshot_path,
        "OK" if saved else "FAILED"
    )
    print(
        "History    :",
        WATCHLIST_HISTORY_FILE
    )
    print("=" * 60)

    return True


def get_name_from_watchlist_filename(filename):
    name = Path(filename).stem.strip()
    parts = name.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        name = "_".join(parts[:-1])
    return name.strip()


def load_watchlist(verbose=True):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    folder = Path(WATCHLIST_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)

    names = set()

    for item in sorted(folder.iterdir()):
        if not item.is_file():
            continue
        if item.suffix.lower() not in valid_extensions:
            continue

        name = get_name_from_watchlist_filename(item.name)
        if name:
            names.add(name.lower())

    if verbose:
        print()
        print("Watchlist loaded:", len(names), "person(s)")
        for name in sorted(names):
            print("WATCHLIST:", name)

    return names


def is_watchlist_person(name, watchlist_names):
    if not name:
        return False
    if str(name).strip().upper() == "UNKNOWN":
        return False
    return str(name).strip().lower() in watchlist_names

# Face recognition threshold
MATCH_THRESHOLD = 0.45

# YOLO confidence
YOLO_CONFIDENCE = 0.35

# YOLO image size
YOLO_IMGSZ = 640


# ============================================================
# CHECK DATABASE
# ============================================================

if not os.path.exists(DATABASE_FILE):

    print()
    print("ERROR: Face database not found!")
    print(f"Expected: {DATABASE_FILE}")
    raise SystemExit(1)


# ============================================================
# LOAD FACE DATABASE
# ============================================================

print()
print("Loading face database...")

database = np.load(DATABASE_FILE)

reference_embeddings = database["embeddings"].astype(
    np.float32
)

reference_names = database["names"]

print(
    "Reference faces loaded:",
    len(reference_embeddings)
)

registered_names = sorted(
    set(str(name) for name in reference_names)
)

for name in registered_names:
    print("Registered:", name)


# ============================================================
# NORMALIZE DATABASE EMBEDDINGS
# ============================================================

for i in range(len(reference_embeddings)):

    norm = np.linalg.norm(
        reference_embeddings[i]
    )

    if norm > 0:

        reference_embeddings[i] /= norm


# ============================================================
# INITIALIZE ONNX / CUDA
# ============================================================

print()
print("Checking GPU providers...")

try:

    ort.preload_dlls()

except Exception:

    pass


providers = ort.get_available_providers()

print()
print("Available ONNX providers:")
print(providers)


if "CUDAExecutionProvider" in providers:

    face_providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]

else:

    face_providers = [
        "CPUExecutionProvider"
    ]


print()
print("InsightFace providers:")
print(face_providers)


# ============================================================
# INITIALIZE INSIGHTFACE
# ============================================================

print()
print("Initializing InsightFace...")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=face_providers
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace initialized successfully.")


# ============================================================
# INITIALIZE YOLO
# ============================================================

print()
print("Loading YOLO...")

try:

    yolo = YOLO(YOLO_MODEL)

except Exception as e:

    print()
    print("ERROR: Could not load YOLO.")
    print(e)

    raise SystemExit(1)


print("YOLO loaded successfully.")


# ============================================================
# YOLO DEVICE
# ============================================================

if "CUDAExecutionProvider" in providers:

    YOLO_DEVICE = 0

    print("YOLO device: NVIDIA GPU")

else:

    YOLO_DEVICE = "cpu"

    print("YOLO device: CPU")


# ============================================================
# FACE RECOGNITION FUNCTION
# ============================================================

def recognize_face(face):

    try:

        embedding = face.embedding.astype(
            np.float32
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:

            return "UNKNOWN", 0.0

        embedding /= norm

        similarities = np.dot(
            reference_embeddings,
            embedding
        )

        best_index = np.argmax(
            similarities
        )

        best_score = float(
            similarities[best_index]
        )

        best_name = str(
            reference_names[best_index]
        )

        if best_score >= MATCH_THRESHOLD:

            return best_name, best_score

        return "UNKNOWN", best_score

    except Exception:

        return "UNKNOWN", 0.0


# ============================================================

# ============================================================
# PHASE 3B-2 BORDER CROSSING ENGINE
# ============================================================
# Detects a person crossing the configured virtual border in
# either direction using YOLO + ByteTrack track IDs.
#
# Unknown people also trigger a crossing event.
# This phase intentionally keeps crossing identity as UNKNOWN;
# a later association phase will safely connect a crossing to
# the recognized face/watchlist identity.

BORDER_CROSSING_COOLDOWN = 5.0

BORDER_SCREENSHOT_FOLDER = (
    Path("data") / "alerts" / "border_crossing_screenshots"
)

BORDER_HISTORY_FILE = (
    Path("data") / "alerts" / "border_crossing_history.jsonl"
)

BORDER_SCREENSHOT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

BORDER_HISTORY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

border_track_sides = {}
border_track_last_crossing = {}


def border_side(point, p1, p2):
    px, py = point
    x1, y1 = p1
    x2, y2 = p2

    return (
        (x2 - x1) * (py - y1)
        - (y2 - y1) * (px - x1)
    )


def border_crossing_event(
    frame,
    track_id,
    person_name,
    confidence,
    camera_id,
    direction,
    bbox,
):
    timestamp = datetime.now()

    stamp = timestamp.strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in str(person_name)
    )

    screenshot_path = (
        BORDER_SCREENSHOT_FOLDER
        / (
            f"border_{safe_name}_"
            f"camera_{camera_id}_"
            f"track_{track_id}_"
            f"{stamp}.jpg"
        )
    )

    saved = cv2.imwrite(
        str(screenshot_path),
        frame
    )

    event = {
        "event": "BORDER_CROSSING",
        "timestamp": timestamp.isoformat(
            timespec="milliseconds"
        ),
        "camera_id": int(camera_id),
        "track_id": (
            int(track_id)
            if track_id is not None
            else None
        ),
        "person": str(person_name),
        "confidence": round(
            float(confidence),
            2
        ),
        "direction": str(direction),
        "bbox": [int(v) for v in bbox],
        "screenshot": str(screenshot_path),
    }

    with open(
        BORDER_HISTORY_FILE,
        "a",
        encoding="utf-8"
    ) as history_file:
        history_file.write(
            json.dumps(
                event,
                ensure_ascii=False
            ) + "\n"
        )

    print()
    print("=" * 60)
    print("BORDER CROSSING ALERT")
    print("Person     :", person_name)
    print("Confidence :", f"{float(confidence):.1f}%")
    print("Camera     :", camera_id)
    print("Track ID   :", track_id)
    print("Direction  :", direction)
    print(
        "Screenshot :",
        screenshot_path,
        "OK" if saved else "FAILED"
    )
    print(
        "History    :",
        BORDER_HISTORY_FILE
    )
    print("=" * 60)

    return saved


# CAMERA
# ============================================================

print()
print("Opening camera...")

cap = cv2.VideoCapture(
    CAMERA_ID
)

if not cap.isOpened():

    print()
    print("ERROR: Could not open camera.")
    raise SystemExit(1)


# Camera resolution

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


print("Camera started successfully.")

print()
print("Press Q to quit.")
print("=" * 70)


# ============================================================
# MAIN LOOP
# ============================================================

frame_count = 0

fps_start = time.time()

fps = 0.0



# ============================================================
# LIVE WATCHLIST STATE
# ============================================================

watchlist_names = load_watchlist(verbose=True)
alert_count = 0
alerted_names = set()
last_watchlist_event_times = {}
last_watchlist_event_time = 0
last_watchlist_scan = time.time()


# ============================================================
# PHASE 3B-1 BORDER WINDOW SETUP
# ============================================================

cv2.namedWindow("SIH CCTV Monitoring", cv2.WINDOW_NORMAL)
cv2.setMouseCallback(
    "SIH CCTV Monitoring",
    border_mouse_callback
)

print()
print("PHASE 3B-1 VIRTUAL BORDER READY")
print("Press B, then click two points.")
print("Press R to reset/redefine.")
print("Press Q to quit.")

while True:

    success, frame = cap.read()

    if not success:

        print(
            "ERROR: Could not read camera frame."
        )

        break



    # ========================================================
    # CHECK CURRENT WATCHLIST FOLDER
    # ========================================================

    current_time = time.time()

    if current_time - last_watchlist_scan >= WATCHLIST_CHECK_INTERVAL:

        latest_watchlist = load_watchlist(verbose=False)

        if latest_watchlist != watchlist_names:

            added = sorted(latest_watchlist - watchlist_names)
            removed = sorted(watchlist_names - latest_watchlist)

            if added:
                print("WATCHLIST ADDED:", ", ".join(added))

            if removed:
                print("WATCHLIST REMOVED:", ", ".join(removed))

            watchlist_names = latest_watchlist

        last_watchlist_scan = current_time


    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    frame_count += 1

    elapsed = (
        time.time() - fps_start
    )

    if elapsed >= 1.0:

        fps = frame_count / elapsed

        frame_count = 0

        fps_start = time.time()


    # ========================================================
    # YOLO OBJECT DETECTION + BYTETRACK
    # ========================================================

    try:

        results = yolo.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            conf=YOLO_CONFIDENCE,

            imgsz=YOLO_IMGSZ,

            device=YOLO_DEVICE,

            verbose=False

        )

    except Exception as e:

        print(
            "YOLO error:",
            e
        )

        results = []


    # ========================================================
    # DRAW YOLO OBJECTS
    # ========================================================

    if results:

        result = results[0]

        if result.boxes is not None:

            boxes = result.boxes

            xyxy = boxes.xyxy.cpu().numpy()

            classes = boxes.cls.cpu().numpy()

            confidences = boxes.conf.cpu().numpy()


            if boxes.id is not None:

                track_ids = (
                    boxes.id
                    .cpu()
                    .numpy()
                )

            else:

                track_ids = None


            # ------------------------------------------------
            # LOOP THROUGH OBJECTS
            # ------------------------------------------------

            for i in range(len(xyxy)):

                x1, y1, x2, y2 = (
                    xyxy[i].astype(int)
                )

                class_id = int(
                    classes[i]
                )

                confidence = float(
                    confidences[i]
                )


                # YOLO class name

                class_name = result.names[
                    class_id
                ]


                # ------------------------------------------------
                # PHASE 3B-2 BORDER CROSSING
                # ------------------------------------------------

                if (
                    class_name == "person"
                    and len(border_points) == 2
                    and track_ids is not None
                ):
                    center_x = (
                        int(x1) + int(x2)
                    ) // 2

                    center_y = (
                        int(y1) + int(y2)
                    ) // 2

                    current_side_value = border_side(
                        (center_x, center_y),
                        border_points[0],
                        border_points[1],
                    )

                    side_deadband = 3.0

                    if current_side_value > side_deadband:
                        current_side = 1
                    elif current_side_value < -side_deadband:
                        current_side = -1
                    else:
                        current_side = 0

                    if current_side != 0:
                        previous_side = border_track_sides.get(
                            track_id
                        )

                        if (
                            previous_side is not None
                            and previous_side != 0
                            and previous_side != current_side
                        ):
                            now = time.time()

                            previous_crossing = (
                                border_track_last_crossing.get(
                                    track_id,
                                    0
                                )
                            )

                            if (
                                now - previous_crossing
                                >= BORDER_CROSSING_COOLDOWN
                            ):
                                direction = (
                                    "SIDE_A_TO_SIDE_B"
                                    if previous_side > 0
                                    else "SIDE_B_TO_SIDE_A"
                                )

                                border_track_last_crossing[
                                    track_id
                                ] = now

                                border_crossing_event(
                                    frame=frame.copy(),
                                    track_id=track_id,
                                    person_name="UNKNOWN",
                                    confidence=confidence * 100,
                                    camera_id=CAMERA_ID,
                                    direction=direction,
                                    bbox=(x1, y1, x2, y2),
                                )

                        border_track_sides[
                            track_id
                        ] = current_side

                # ------------------------------------------------
                # TRACK ID
                # ------------------------------------------------

                if track_ids is not None:

                    track_id = int(
                        track_ids[i]
                    )

                    label = (
                        f"{class_name} "
                        f"ID:{track_id} "
                        f"{confidence * 100:.0f}%"
                    )

                else:

                    label = (
                        f"{class_name} "
                        f"{confidence * 100:.0f}%"
                    )


                # ------------------------------------------------
                # OBJECT BOX
                # ------------------------------------------------

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (255, 0, 0),

                    2

                )


                # ------------------------------------------------
                # LABEL SIZE
                # ------------------------------------------------

                (
                    text_width,
                    text_height
                ), baseline = cv2.getTextSize(

                    label,

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    2

                )


                label_top = max(

                    0,

                    y1 -
                    text_height -
                    baseline -
                    8

                )


                # ------------------------------------------------
                # LABEL BACKGROUND
                # ------------------------------------------------

                cv2.rectangle(

                    frame,

                    (
                        x1,
                        label_top
                    ),

                    (
                        x1 +
                        text_width +
                        8,

                        y1
                    ),

                    (255, 0, 0),

                    -1

                )


                # ------------------------------------------------
                # LABEL TEXT
                # ------------------------------------------------

                cv2.putText(

                    frame,

                    label,

                    (
                        x1 + 4,

                        y1 - 6
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (255, 255, 255),

                    2

                )


    # ========================================================
    # INSIGHTFACE FACE DETECTION
    # ========================================================

    try:

        faces = face_app.get(
            frame
        )

    except Exception as e:

        print(
            "Face AI error:",
            e
        )

        faces = []


    # ========================================================
    # FACE RECOGNITION
    # ========================================================

    recognized_count = 0

    unknown_count = 0

    watchlist_detected = False
    watchlist_person_names = set()
    watchlist_events_to_create = []


    for face in faces:

        try:

            x1, y1, x2, y2 = (
                face.bbox.astype(int)
            )

            name, score = recognize_face(
                face
            )

            confidence = score * 100


            # ------------------------------------------------
            # COUNTERS
            # ------------------------------------------------

            if name == "UNKNOWN":

                unknown_count += 1

                label = (
                    f"UNKNOWN "
                    f"{confidence:.1f}%"
                )

            else:

                recognized_count += 1

                label = (
                    f"{name} "
                    f"{confidence:.1f}%"
                )


            # ------------------------------------------------
            # WATCHLIST DECISION
            # ------------------------------------------------

            if is_watchlist_person(
                name,
                watchlist_names
            ):

                watchlist_detected = True
                watchlist_person_names.add(name)

                normalized_name = str(name).strip().lower()

                if normalized_name not in alerted_names:
                    alerted_names.add(normalized_name)
                    alert_count += 1

                watchlist_events_to_create.append(
                    (
                        name,
                        confidence,
                        (x1, y1, x2, y2),
                    )
                )

                label = (
                    f"{WATCHLIST_ALERT_LABEL} | "
                    f"{name} {confidence:.1f}%"
                )

                box_color = (0, 0, 255)
                box_thickness = 3
                label_color = (0, 0, 255)

            else:

                if name == "UNKNOWN":
                    label = f"UNKNOWN {confidence:.1f}%"
                else:
                    label = f"{name} {confidence:.1f}%"

                box_color = (255, 0, 0)
                box_thickness = 2
                label_color = (255, 0, 0)


            # ------------------------------------------------
            # FACE BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                box_thickness
            )


            # ------------------------------------------------
            # LABEL SIZE
            # ------------------------------------------------

            (
                text_width,
                text_height
            ), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                2
            )

            label_top = max(
                0,
                y1 - text_height - baseline - 8
            )


            # ------------------------------------------------
            # FACE LABEL BACKGROUND
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, label_top),
                (x1 + text_width + 8, y1),
                label_color,
                -1
            )


            # ------------------------------------------------
            # FACE LABEL
            # ------------------------------------------------

            cv2.putText(
                frame,
                label,
                (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        except Exception as e:

            print(
                "Face processing error:",
                e
            )


    # ========================================================
    # CREATE WATCHLIST EVENTS AFTER FACE ANNOTATIONS
    # ========================================================

    for event_name, event_confidence, event_bbox in watchlist_events_to_create:
        create_watchlist_event(
            frame=frame,
            name=event_name,
            confidence=event_confidence,
            camera_id=CAMERA_ID,
            track_id=None,
            bbox=event_bbox,
        )


    # ========================================================
    # HEADER
    # ========================================================

    cv2.rectangle(

        frame,

        (0, 0),

        (1280, 55),

        (0, 0, 0),

        -1

    )


    cv2.putText(

        frame,

        "SIH CCTV MONITORING",

        (20, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.85,

        (255, 255, 255),

        2

    )


    # ========================================================
    # FACE COUNTER
    # ========================================================

    face_text = (
        f"Faces: {len(faces)}"
    )

    cv2.putText(

        frame,

        face_text,

        (360, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2

    )


    # ========================================================
    # RECOGNIZED COUNTER
    # ========================================================

    recognized_text = (
        f"Recognized: {recognized_count}"
    )

    cv2.putText(

        frame,

        recognized_text,

        (510, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0, 255, 255),

        2

    )


    # ========================================================
    # UNKNOWN COUNTER
    # ========================================================

    unknown_text = (
        f"Unknown: {unknown_count}"
    )

    cv2.putText(

        frame,

        unknown_text,

        (720, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0, 165, 255),

        2

    )


    # ========================================================
    # FPS
    # ========================================================

    fps_text = (
        f"FPS: {fps:.1f}"
    )

    cv2.putText(

        frame,

        fps_text,

        (900, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    # ========================================================
    # GPU STATUS
    # ========================================================

    gpu_text = "GPU: ON"

    cv2.putText(

        frame,

        gpu_text,

        (1060, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0, 255, 0),

        2

    )


    # ========================================================
    # BORDER STATUS
    # ========================================================

    if len(border_points) == 2:
        cv2.putText(
            frame,
            "BORDER: ACTIVE",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )
    else:
        cv2.putText(
            frame,
            "BORDER: NOT SET",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    # ========================================================
    # ========================================================
    # SIH SECURITY INFORMATION PANEL
    # ========================================================

    h, w = frame.shape[:2]
    panel_w = 320

    canvas = np.zeros(
        (h, w + panel_w, 3),
        dtype=np.uint8
    )

    canvas[:h, :w] = frame
    panel_x = w

    cv2.rectangle(
        canvas,
        (panel_x, 0),
        (w + panel_w, h),
        (25, 25, 25),
        -1
    )

    security_status = "ALERT" if watchlist_detected else "NORMAL"

    status_color = (
        (0, 0, 255)
        if watchlist_detected
        else (0, 255, 0)
    )

    cv2.putText(
        canvas,
        "SECURITY STATUS",
        (panel_x + 20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        "SYSTEM",
        (panel_x + 20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1
    )

    cv2.putText(
        canvas,
        security_status,
        (panel_x + 20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )

    cv2.putText(
        canvas,
        "CAMERA",
        (panel_x + 20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1
    )

    cv2.putText(
        canvas,
        f"Camera ID: {CAMERA_ID}",
        (panel_x + 20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        "ONLINE",
        (panel_x + 20, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        canvas,
        "DETECTIONS",
        (panel_x + 20, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1
    )

    cv2.putText(
        canvas,
        f"Faces: {len(faces)}",
        (panel_x + 20, 345),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        f"Recognized: {recognized_count}",
        (panel_x + 20, 380),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        canvas,
        f"Unknown: {unknown_count}",
        (panel_x + 20, 415),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 165, 255),
        2
    )

    cv2.putText(
        canvas,
        f"FPS: {fps:.1f}",
        (panel_x + 20, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        "WATCHLIST",
        (panel_x + 20, 510),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1
    )

    cv2.putText(
        canvas,
        f"Alerts: {alert_count}",
        (panel_x + 20, 545),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )

    cv2.putText(
        canvas,
        f"STATUS: {security_status}",
        (panel_x + 20, 580),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2
    )

    cv2.putText(
        canvas,
        f"Watchlist people: {len(watchlist_names)}",
        (panel_x + 20, 615),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    if watchlist_detected:

        detected_text = (
            f"{WATCHLIST_ALERT_LABEL}: "
            + ", ".join(sorted(watchlist_person_names))
        )

        cv2.putText(
            canvas,
            detected_text[:42],
            (panel_x + 20, 650),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2
        )

    cv2.putText(
        canvas,
        "Q = QUIT",
        (panel_x + 20, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1
    )

    frame = canvas
    # ========================================================
    # DRAW VIRTUAL BORDER
    # ========================================================

    if len(border_points) == 1:
        cv2.circle(
            frame,
            border_points[0],
            7,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            frame,
            "CLICK SECOND POINT",
            (
                border_points[0][0] + 12,
                max(25, border_points[0][1] - 12)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    elif len(border_points) == 2:
        cv2.line(
            frame,
            border_points[0],
            border_points[1],
            (0, 255, 255),
            4
        )

        cv2.circle(
            frame,
            border_points[0],
            7,
            (0, 255, 255),
            -1
        )

        cv2.circle(
            frame,
            border_points[1],
            7,
            (0, 255, 255),
            -1
        )

        midpoint = (
            (border_points[0][0] + border_points[1][0]) // 2,
            (border_points[0][1] + border_points[1][1]) // 2
        )

        cv2.putText(
            frame,
            "VIRTUAL BORDER",
            (
                max(5, midpoint[0] - 100),
                max(25, midpoint[1] - 12)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    if border_setup_mode:
        cv2.putText(
            frame,
            "BORDER SETUP: CLICK 2 POINTS",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    cv2.imshow(

        "SIH CCTV Monitoring",

        frame

    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("b"):
        reset_virtual_border()

    elif key == ord("r"):
        reset_virtual_border()


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("=" * 70)
print("Camera stopped.")
print("SIH CCTV monitoring finished.")
print("=" * 70)