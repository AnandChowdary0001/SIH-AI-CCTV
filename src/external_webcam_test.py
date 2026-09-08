import cv2

print("=" * 60)
print("SIH EXTERNAL WEBCAM TEST")
print("=" * 60)

CAMERA_ID = 1

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: External webcam could not be opened.")
    raise SystemExit(1)

print("External webcam opened successfully.")
print("Press Q to close.")
print("=" * 60)

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "EXTERNAL WEBCAM - CAMERA 1",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("SIH External Webcam Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("External webcam test finished.")