import cv2

print("=" * 50)
print("SIH WEBCAM TEST")
print("=" * 50)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit(1)

print("Webcam opened successfully.")
print("Press Q to close the webcam window.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "WEBCAM WORKING",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("SIH Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Webcam test finished.")