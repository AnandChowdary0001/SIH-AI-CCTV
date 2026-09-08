\# AI-Powered Smart CCTV Surveillance System



\## Smart India Hackathon Project



An AI-powered CCTV surveillance system designed for real-time person detection, face recognition, watchlist monitoring, and alert generation.



\## Features



\- Real-time CCTV/webcam video processing

\- Person detection using YOLO

\- Face recognition using a reference face database

\- Detection of a specific person inside a group

\- Dynamic and user-configurable watchlist

\- Watchlist-based target alerts

\- CCTV dashboard

\- Laptop camera and external USB webcam support

\- Modular AI, API, and utility components



\## System Workflow



Reference Image

&#x20;      ↓

Face Database

&#x20;      ↓

CCTV / Webcam

&#x20;      ↓

Person Detection

&#x20;      ↓

Face Detection \& Recognition

&#x20;      ↓

Watchlist Verification

&#x20;      ↓

Normal Recognition / Target Alert

&#x20;      ↓

Dashboard



\## Watchlist Logic



The reference face database and watchlist are separate.



A person being present in the reference database does not automatically make them a target.



Only people explicitly added to the active watchlist can trigger a target alert.



The watchlist can be modified by the user by adding or removing people.



\## Project Structure



```text

SIH\_Project/

│

├── src/

│   ├── ai/

│   ├── api/

│   ├── utils/

│   ├── build\_face\_database.py

│   ├── face\_recognition\_camera.py

│   ├── integrated\_cctv.py

│   ├── sih\_dashboard.py

│   └── webcam\_test.py

│

├── tests/

├── .gitignore

├── README.md

└── requirements.txt

