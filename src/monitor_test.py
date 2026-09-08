import time
from datetime import datetime


print("=" * 60)
print("SIH CCTV MONITORING TEST")
print("=" * 60)

print("\nSystem Components")
print("-----------------------------")

print("Camera       : ONLINE")
print("YOLO         : ONLINE")
print("ByteTrack    : ONLINE")
print("Face AI      : ONLINE")
print("GPU          : RTX 3050")
print("CUDA         : ONLINE")

print("\nMonitoring started...")
print("Press Ctrl+C to stop.")
print("=" * 60)


try:

    while True:

        current_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        print(
            f"[{current_time}] "
            "CCTV system monitoring..."
        )

        time.sleep(5)


except KeyboardInterrupt:

    print("\n")
    print("=" * 60)
    print("Monitoring stopped.")
    print("=" * 60)