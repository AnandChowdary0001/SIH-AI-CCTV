import os
import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np


# ============================================================
# CUDA DLL SETUP
# ============================================================

torch_lib = os.path.join(
    sys.prefix,
    "Lib",
    "site-packages",
    "torch",
    "lib"
)

if os.path.isdir(torch_lib):
    os.add_dll_directory(torch_lib)
    os.environ["PATH"] = (
        torch_lib + os.pathsep + os.environ["PATH"]
    )

import onnxruntime as ort

ort.preload_dlls()

from insightface.app import FaceAnalysis
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

REFERENCE_FOLDER = "data/reference"
DATABASE_FILE = "data/face_database.npz"

YOLO_MODEL = "yolo11n.pt"

MATCH_THRESHOLD = 0.45

# Number of recent recognition results used
# to confirm an identity.
CONFIRMATION_FRAMES = 5

# Minimum number of matching results required.
MIN_CONFIRMATIONS = 3

# Check for newly added reference images every N seconds.
REFERENCE_CHECK_INTERVAL = 3


# ============================================================
# CAMERA SELECTION
# ============================================================

print("\n")
print("=" * 70)
print("                 SIH CAMERA SELECTION")
print("=" * 70)
print()
print("0 - Laptop Camera")
print("1 - External USB Webcam")
print()

while True:
    choice = input("Enter camera number (0 or 1): ").strip()

    if choice in ["0", "1"]:
        CAMERA_ID = int(choice)
        break

    print("Invalid choice. Please enter 0 or 1.")

print()
print("Selected Camera ID:", CAMERA_ID)
print("=" * 70)


# ============================================================
# START
# ============================================================

print("\n")
print("=" * 70)
print("SIH INTEGRATED CCTV SYSTEM")
print("YOLO + BYTETRACK + INSIGHTFACE")
print("=" * 70)


# ============================================================
# CUDA / ONNX
# ============================================================

providers = ort.get_available_providers()

print("\nONNX providers:")
print(providers)

if "CUDAExecutionProvider" in providers:

    insightface_providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]

    print("\nInsightFace GPU: ENABLED")

else:

    insightface_providers = [
        "CPUExecutionProvider"
    ]

    print("\nWARNING: InsightFace GPU unavailable.")


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO...")

yolo = YOLO(YOLO_MODEL)

print("YOLO loaded.")


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("\nLoading InsightFace...")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=insightface_providers
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded.")


# ============================================================
# DATABASE
# ============================================================

reference_embeddings = np.empty(
    (0, 512),
    dtype=np.float32
)

reference_names = np.array(
    [],
    dtype=str
)

processed_files = set()


# ============================================================
# GET NAME FROM FILENAME
# ============================================================

def get_person_name(filename):

    base_name = os.path.splitext(
        filename
    )[0]

    # Anand_01 -> Anand
    # Anand_02 -> Anand
    # Rahul -> Rahul

    parts = base_name.split("_")

    if len(parts) > 1 and parts[-1].isdigit():

        base_name = "_".join(
            parts[:-1]
        )

    return base_name.strip()


# ============================================================
# LOAD DATABASE
# ============================================================

def load_database():

    global reference_embeddings
    global reference_names
    global processed_files

    if not os.path.exists(DATABASE_FILE):

        print(
            "\nNo face database found."
        )

        return

    try:

        database = np.load(
            DATABASE_FILE,
            allow_pickle=True
        )

        reference_embeddings = database[
            "embeddings"
        ]

        reference_names = database[
            "names"
        ]

        if "files" in database:

            processed_files = set(
                database["files"].tolist()
            )

        print(
            "\nLoaded face database:",
            len(reference_embeddings),
            "embeddings"
        )

    except Exception as error:

        print(
            "\nDatabase loading error:"
        )

        print(error)


# ============================================================
# SAVE DATABASE
# ============================================================

def save_database():

    np.savez(
        DATABASE_FILE,
        embeddings=reference_embeddings,
        names=reference_names,
        files=np.array(
            list(processed_files)
        )
    )


# ============================================================
# PROCESS NEW REFERENCE IMAGE
# ============================================================

def process_reference_image(filename):

    global reference_embeddings
    global reference_names
    global processed_files

    image_path = os.path.join(
        REFERENCE_FOLDER,
        filename
    )

    print(
        "\nNew reference image:",
        filename
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "  ERROR: Cannot read image."
        )

        return False

    faces = face_app.get(
        image
    )

    if len(faces) == 0:

        print(
            "  ERROR: No face detected."
        )

        return False

    # Use largest face
    face = max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0]) *
        (f.bbox[3] - f.bbox[1])
    )

    embedding = face.embedding.astype(
        np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:

        print(
            "  ERROR: Invalid embedding."
        )

        return False

    embedding = embedding / norm

    person_name = get_person_name(
        filename
    )

    reference_embeddings = np.vstack(
        [
            reference_embeddings,
            embedding
        ]
    )

    reference_names = np.append(
        reference_names,
        person_name
    )

    processed_files.add(
        filename
    )

    save_database()

    print(
        "  Name:",
        person_name
    )

    print(
        "  Face detected: YES"
    )

    print(
        "  Database updated."
    )

    return True


# ============================================================
# SCAN REFERENCE FOLDER
# ============================================================

def scan_reference_folder():

    if not os.path.exists(
        REFERENCE_FOLDER
    ):

        os.makedirs(
            REFERENCE_FOLDER
        )

        return

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    for filename in sorted(
        os.listdir(
            REFERENCE_FOLDER
        )
    ):

        if not filename.lower().endswith(
            valid_extensions
        ):
            continue

        if filename in processed_files:
            continue

        process_reference_image(
            filename
        )


# ============================================================
# INITIAL DATABASE
# ============================================================

load_database()

scan_reference_folder()


# ============================================================
# PRINT REGISTERED PEOPLE
# ============================================================

print("\nRegistered people:")

if len(reference_names) > 0:

    for name in sorted(
        set(
            reference_names.tolist()
        )
    ):

        count = np.sum(
            reference_names == name
        )

        print(
            f"  {name}: {count}"
        )

else:

    print("  NONE")


# ============================================================
# RECOGNITION FUNCTION
# ============================================================

def recognize_face(face):

    if len(reference_embeddings) == 0:

        return "UNKNOWN", 0.0

    embedding = face.embedding.astype(
        np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:

        return "UNKNOWN", 0.0

    embedding = embedding / norm

    similarities = np.dot(
        reference_embeddings,
        embedding
    )

    best_index = int(
        np.argmax(similarities)
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


# ============================================================
# TRACK HISTORY
# ============================================================

track_history = defaultdict(
    lambda: deque(
        maxlen=CONFIRMATION_FRAMES
    )
)

confirmed_identity = {}


# ============================================================
# CAMERA
# ============================================================

print("\nOpening camera...")

cap = cv2.VideoCapture(
    CAMERA_ID
)

if not cap.isOpened():

    print(
        "\nERROR: Selected camera could not be opened."
    )

    print(
        "Try another camera number."
    )

    raise SystemExit(1)


print(
    "Camera started."
)

print(
    "Press Q to quit."
)

print("=" * 70)


# ============================================================
# TIMER
# ============================================================

last_reference_scan = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        print(
            "ERROR: Camera frame unavailable."
        )

        break


    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # CHECK FOR NEW REFERENCE PHOTOS
    # ========================================================

    current_time = time.time()

    if (
        current_time -
        last_reference_scan
        >= REFERENCE_CHECK_INTERVAL
    ):

        scan_reference_folder()

        last_reference_scan = current_time


    # ========================================================
    # YOLO + BYTETRACK
    # ========================================================

    results = yolo.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        device=0,
        verbose=False
    )


    # ========================================================
    # DETECT FACES
    # ========================================================

    faces = face_app.get(
        frame
    )


    # ========================================================
    # MATCH FACE TO PERSON TRACK
    # ========================================================

    for face in faces:

        fx1, fy1, fx2, fy2 = (
            face.bbox.astype(int)
        )


        # Face center

        face_cx = int(
            (fx1 + fx2) / 2
        )

        face_cy = int(
            (fy1 + fy2) / 2
        )


        # Recognize face

        name, score = recognize_face(
            face
        )


        # ====================================================
        # FIND CLOSEST YOLO TRACK
        # ====================================================

        best_track_id = None

        best_distance = float(
            "inf"
        )


        if (
            results and
            len(results) > 0 and
            results[0].boxes.id is not None
        ):

            boxes = results[0].boxes

            track_ids = boxes.id.cpu().numpy()

            person_boxes = boxes.xyxy.cpu().numpy()


            for track_id, box in zip(
                track_ids,
                person_boxes
            ):

                px1, py1, px2, py2 = box


                # Check whether face center
                # is inside person box.

                if (
                    face_cx >= px1 and
                    face_cx <= px2 and
                    face_cy >= py1 and
                    face_cy <= py2
                ):

                    person_cx = (
                        px1 + px2
                    ) / 2

                    person_cy = (
                        py1 + py2
                    ) / 2


                    distance = (
                        (face_cx - person_cx) ** 2
                        +
                        (face_cy - person_cy) ** 2
                    )


                    if distance < best_distance:

                        best_distance = distance

                        best_track_id = int(
                            track_id
                        )


        # ====================================================
        # TEMPORAL CONFIRMATION
        # ====================================================

        if best_track_id is not None:

            track_history[
                best_track_id
            ].append(
                name
            )


            history = track_history[
                best_track_id
            ]


            # Count identities

            counts = {}

            for item in history:

                counts[item] = (
                    counts.get(
                        item,
                        0
                    ) + 1
                )


            strongest_name = max(
                counts,
                key=counts.get
            )

            strongest_count = counts[
                strongest_name
            ]


            if (
                strongest_name != "UNKNOWN"
                and
                strongest_count >=
                MIN_CONFIRMATIONS
            ):

                confirmed_identity[
                    best_track_id
                ] = strongest_name


            elif (
                strongest_name == "UNKNOWN"
                and
                best_track_id not in
                confirmed_identity
            ):

                confirmed_identity[
                    best_track_id
                ] = "UNKNOWN"


        # ====================================================
        # DISPLAY IDENTITY
        # ====================================================

        display_name = name


        if best_track_id is not None:

            if best_track_id in confirmed_identity:

                display_name = confirmed_identity[
                    best_track_id
                ]


        # ====================================================
        # DRAW FACE BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (fx1, fy1),
            (fx2, fy2),
            (255, 0, 0),
            2
        )


        # ====================================================
        # LABEL
        # ====================================================

        if best_track_id is not None:

            label = (
                f"ID {best_track_id} | "
                f"{display_name}"
            )

        else:

            label = display_name


        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2
        )


        label_y = max(
            0,
            fy1 - text_height - baseline - 6
        )


        cv2.rectangle(
            frame,
            (
                fx1,
                label_y
            ),
            (
                fx1 +
                text_width +
                8,
                fy1
            ),
            (255, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            label,
            (
                fx1 + 4,
                fy1 - 6
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    # ========================================================
    # DRAW YOLO PERSON TRACK BOXES
    # ========================================================

    if (
        results and
        len(results) > 0
    ):

        boxes = results[0].boxes

        if (
            boxes is not None and
            boxes.id is not None
        ):

            person_boxes = (
                boxes.xyxy.cpu().numpy()
            )

            track_ids = (
                boxes.id.cpu().numpy()
            )


            for box, track_id in zip(
                person_boxes,
                track_ids
            ):

                px1, py1, px2, py2 = (
                    box.astype(int)
                )


                # Thin green person boundary

                cv2.rectangle(
                    frame,
                    (px1, py1),
                    (px2, py2),
                    (0, 255, 0),
                    1
                )


    # ========================================================
    # STATUS PANEL
    # ========================================================

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "SIH CCTV | Q = Quit",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "SIH Integrated CCTV",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if (
        cv2.waitKey(1) & 0xFF
        == ord("q")
    ):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print(
    "\nSIH Integrated CCTV stopped."
)