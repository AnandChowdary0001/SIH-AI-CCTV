import os
import re
import cv2
import numpy as np
import torch
import onnxruntime as ort

# Load CUDA libraries from the PyTorch environment
ort.preload_dlls()

from insightface.app import FaceAnalysis


REFERENCE_DIR = os.path.join("..", "data", "reference")
OUTPUT_FILE = "data/face_database.npz"


def get_name_from_filename(filename):
    """
    Example:
        Anand_01.jpg -> Anand
        Rahul_02.jpg -> Rahul
        Priya.jpg    -> Priya
    """
    name = os.path.splitext(filename)[0]

    # Remove the final _number
    name = re.sub(r"_\d+$", "", name)

    return name


print("=" * 60)
print("SIH FACE DATABASE BUILDER")
print("=" * 60)

print("\nAvailable ONNX providers:")
print(ort.get_available_providers())

if "CUDAExecutionProvider" not in ort.get_available_providers():
    print("\nWARNING: CUDAExecutionProvider is not available.")
    print("Face processing may run on CPU.")

# Initialize InsightFace
app = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("\nInsightFace initialized.")
print("Reference folder:", REFERENCE_DIR)


embeddings = []
names = []
files = []

image_extensions = (".jpg", ".jpeg", ".png")

for filename in sorted(os.listdir(REFERENCE_DIR)):

    if not filename.lower().endswith(image_extensions):
        continue

    image_path = os.path.join(REFERENCE_DIR, filename)

    print("\nProcessing:", filename)

    image = cv2.imread(image_path)

    if image is None:
        print("  ERROR: Could not read image.")
        continue

    faces = app.get(image)

    if len(faces) == 0:
        print("  ERROR: No face detected.")
        continue

    # If multiple faces exist, use the largest face
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) *
                      (f.bbox[3] - f.bbox[1])
    )

    embedding = face.embedding.astype(np.float32)

    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)

    person_name = get_name_from_filename(filename)

    embeddings.append(embedding)
    names.append(person_name)
    files.append(filename)

    print("  Face detected: YES")
    print("  Name:", person_name)
    print("  Embedding size:", embedding.shape)


if len(embeddings) == 0:
    print("\nERROR: No valid face embeddings were created.")
    raise SystemExit(1)


embeddings = np.asarray(embeddings, dtype=np.float32)
names = np.asarray(names)
files = np.asarray(files)

np.savez(
    OUTPUT_FILE,
    embeddings=embeddings,
    names=names,
    files=files
)

print("\n" + "=" * 60)
print("DATABASE CREATED SUCCESSFULLY")
print("=" * 60)

print("Images processed:", len(embeddings))
print("Database file:", OUTPUT_FILE)

for name, filename in zip(names, files):
    print(f"  {filename} -> {name}")

print("=" * 60)
