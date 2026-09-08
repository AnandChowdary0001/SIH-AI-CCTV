import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

# ============================================================
# SIH EXTERNAL WEBCAM FACE RECOGNITION TEST
# ============================================================

CAMERA_ID = 1
DATABASE_FILE = "data/face_database.npz"
MATCH_THRESHOLD = 0.45

print("=" * 60)
print("SIH EXTERNAL WEBCAM FACE RECOGNITION")
print("=" * 60)

# ------------------------------------------------------------
# LOAD FACE DATABASE
# ------------------------------------------------------------

database = np.load(DATABASE_FILE)

reference_embeddings = database["embeddings"].astype(np.float32)
reference_names = database["names"]

# Normalize database embeddings
reference_embeddings = (
    reference_embeddings /
    np.linalg.norm(
        reference_embeddings,
        axis=1,
        keepdims=True
    )
)

print("Reference faces loaded:", len(reference_embeddings))

for name in sorted(set(reference_names)):
    print("Registered:", name)

# ------------------------------------------------------------
# PRELOAD CUDA DLLs
# ------------------------------------------------------------

try:
    ort.preload_dlls()
    print("ONNX Runtime DLLs loaded.")
except Exception as e:
    print("DLL preload warning:", e)

providers = ort.get_available_providers()

print("\nAvailable providers:")
print(providers)

if "CUDAExecutionProvider" in providers:
    selected_providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
else:
    selected_providers = [
        "CPUExecutionProvider"
    ]

print("\nUsing:")
print(selected_providers)

# ------------------------------------------------------------
# INSIGHTFACE
# ------------------------------------------------------------

app = FaceAnalysis(
    name="buffalo_l",
    providers=selected_providers
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("\nInsightFace initialized successfully.")

# ------------------------------------------------------------
# CAMERA
# ------------------------------------------------------------

print("\nOpening external webcam...")

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Could not open external webcam.")
    raise SystemExit(1)

print("External webcam opened successfully.")
print("Press Q to quit.")
print("=" * 60)

# ------------------------------------------------------------
# FACE RECOGNITION
# ------------------------------------------------------------

def recognize_face(face):

    embedding = face.embedding.astype(np.float32)

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return "UNKNOWN", 0.0

    embedding = embedding / norm

    similarities = np.dot(
        reference_embeddings,
        embedding
    )

    best_index = int(np.argmax(similarities))

    best_score = float(
        similarities[best_index]
    )

    best_name = str(
        reference_names[best_index]
    )

    if best_score >= MATCH_THRESHOLD:
        return best_name, best_score

    return "UNKNOWN", best_score


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror the external webcam
    frame = cv2.flip(frame, 1)

    # Detect faces
    faces = app.get(frame)

    for face in faces:

        x1, y1, x2, y2 = face.bbox.astype(int)

        name, score = recognize_face(face)

        confidence = score * 100

        if name == "UNKNOWN":
            label = f"UNKNOWN {confidence:.1f}%"
        else:
            label = f"{name} {confidence:.1f}%"

        # Face box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # Label size
        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        # Label background
        cv2.rectangle(
            frame,
            (
                x1,
                max(
                    0,
                    y1 - text_height - baseline - 8
                )
            ),
            (
                x1 + text_width + 8,
                y1
            ),
            (255, 0, 0),
            -1
        )

        # Label
        cv2.putText(
            frame,
            label,
            (
                x1 + 4,
                max(
                    text_height + 4,
                    y1 - 6
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # Number of faces
    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Camera indicator
    cv2.putText(
        frame,
        "EXTERNAL WEBCAM",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "SIH External Webcam Face Recognition",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ------------------------------------------------------------
# CLEANUP
# ------------------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("\nExternal webcam stopped.")
print("Face recognition test finished.")