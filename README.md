# Culturally Robust Driver Drowsiness Detection

An AI-powered real-time driver drowsiness detection system designed to remain effective when the driver's face is partially covered by cultural attire such as the **niqab, hijab, and shemagh**.

The system uses a fine-tuned **YOLO11n** model to detect whether the driver's eyes are **open or closed**, monitors eye closure over time, and triggers audio alerts when signs of drowsiness are detected.

> Graduation Project — B.Sc. Computer Science, Qassim University

---

## Overview

Drowsy driving is a major road-safety risk that reduces driver attention and reaction time.

Many existing driver-monitoring systems rely on full facial visibility, facial landmarks, or specialized hardware. This creates challenges in environments where parts of the driver's face may be covered by cultural attire such as the niqab or shemagh.

This project addresses this limitation by focusing specifically on the **eye region** rather than requiring the entire face to be visible.

The system:

- Captures live video from a webcam.
- Detects and classifies the driver's eye state using YOLO11n.
- Tracks sustained eye closure across consecutive frames.
- Calculates PERCLOS (Percentage of Eye Closure).
- Generates graduated audio and visual alerts.
- Performs all processing locally for privacy and low latency.

---

## Key Features

- Real-time open/closed eye detection
- Fine-tuned YOLO11n model
- Robustness to partial facial occlusion and cultural attire
- PERCLOS-based drowsiness monitoring
- Consecutive closed-frame tracking
- Two-level audio warning system
- Real-time visual warning interface
- Standard webcam support
- Fully local processing — no cloud dependency
- Lightweight architecture suitable for future edge deployment

---

## How It Works

The system follows the following pipeline:

**Camera Input → YOLO11n Detection → Eye-State Classification → Temporal Monitoring → Drowsiness Evaluation → Alert**

### 1. Video Capture

A webcam continuously captures frames of the driver.

### 2. Eye-State Detection

Each frame is processed by the fine-tuned YOLO11n model, which detects the eye region and classifies it into one of two classes:

- `open`
- `closed`

### 3. Temporal Monitoring

The system does not classify a driver as drowsy from a single closed-eye frame.

Instead, it monitors:

- **Consecutive closed frames**
- **PERCLOS**

This helps distinguish normal blinking from sustained eye closure.

### 4. Drowsiness Evaluation

The current implementation uses:

| Parameter | Value |
|---|---:|
| Confidence threshold | 0.50 |
| Closed-frame threshold | 20 frames |
| PERCLOS window | 90 frames |
| Warning PERCLOS | 70% |
| Critical PERCLOS | 100% |

A warning is activated when:

```text
Consecutive Closed Frames >= 20
OR
PERCLOS >= 70%
```

A critical alert is activated when:

```text
PERCLOS = 100%
```

### 5. Driver Alert

The system provides two alert levels:

**Level 1 — Warning**

- Drowsiness warning overlay
- Red visual indicator
- 880 Hz audio warning

**Level 2 — Critical**

- Critical visual warning
- Full-screen alert effect
- 1200 Hz urgent alarm

---

## Dataset

The final model was trained using a **hybrid dataset of 5,000 annotated images**.

| Dataset Source | Images | Percentage |
|---|---:|---:|
| Public dataset | 4,000 | 80% |
| Locally collected dataset | 1,000 | 20% |
| **Total** | **5,000** | **100%** |

The locally collected dataset was designed to introduce culturally and environmentally challenging cases, including:

- Niqab
- Shemagh
- Glasses
- Lens glare
- Partial facial occlusion
- Different lighting conditions
- Different head positions

Images were manually annotated using bounding boxes around the **eye region**, with two classes:

```text
open
closed
```

This allows the model to focus on visible eye characteristics instead of relying on the lower part of the face.

---

## Model Selection

Three YOLO architectures were initially trained for 30 epochs under the same conditions.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5–0.95 |
|---|---:|---:|---:|---:|
| YOLOv8n | 0.9015 | 0.9440 | 0.9654 | 0.6270 |
| **YOLO11n** | **0.9494** | **0.9543** | **0.9802** | **0.6358** |
| YOLO26n | 0.9041 | 0.8705 | 0.9359 | 0.5950 |

YOLO11n achieved the best overall performance and was therefore selected for the final system.

---

## Final Model Results

After selecting YOLO11n, the model was trained for **100 epochs**.

| Metric | Result |
|---|---:|
| Precision | **96.61%** |
| Recall | **97.52%** |
| mAP@0.5 | **98.77%** |
| mAP@0.5–0.95 | **57.34%** |

These results demonstrate strong eye-state detection performance while maintaining a lightweight architecture suitable for real-time applications.

---

## Technologies Used

- **Python**
- **Ultralytics YOLO11**
- **OpenCV**
- **NumPy**
- **Pygame**
- **Google Colab**
- **LabelImg**

---

## Project Structure

```text
Culturally-Robust-Driver-Drowsiness-Detection/
│
├── best.pt
├── LiveStream.py
├── README.md
└── requirements.txt
```

### `best.pt`

Fine-tuned YOLO11n model weights used for eye-state detection.

### `LiveStream.py`

Main real-time application responsible for:

- Webcam capture
- YOLO inference
- Eye-state detection
- Consecutive-frame tracking
- PERCLOS calculation
- Audio alerts
- Visual warnings
- FPS monitoring

---

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

### 2. Install Dependencies

```bash
pip install ultralytics opencv-python numpy pygame
```

Or, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

---

## Running the System

Make sure `best.pt` and `LiveStream.py` are located in the same directory.

Then run:

```bash
python LiveStream.py
```

The application will:

1. Load the trained YOLO11n model.
2. Open the default webcam.
3. Begin real-time eye-state detection.
4. Display eye state, PERCLOS, closed-frame streak, FPS, and alert status.

Press:

```text
Q
```

or:

```text
ESC
```

to stop the application.

---

## Configuration

The main detection parameters can be changed at the top of `LiveStream.py`.

```python
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.5
CAMERA_INDEX = 0

CLOSED_FRAMES_THRESHOLD = 20
PERCLOS_WINDOW = 90
PERCLOS_THRESHOLD = 0.7
PERCLOS_CRITICAL = 1.0
```

For example, if an external webcam is being used:

```python
CAMERA_INDEX = 1
```

---

## Privacy

Privacy was an important design consideration.

All video processing and model inference are performed **locally on the device**.

The application does not require:

- Cloud inference
- External servers
- Video uploads
- Permanent storage of webcam frames

Only the information required for real-time drowsiness evaluation is processed during execution.

---

## Cultural Robustness

A major contribution of this project is its focus on driver-monitoring scenarios that are often underrepresented in existing datasets.

Instead of depending on full-face landmarks, the model was trained specifically around the **visible eye region**.

This enables the system to operate when the lower face is partially or substantially covered, including cases involving:

- Niqab
- Shemagh
- Hijab
- Face masks
- Glasses

This makes the approach particularly relevant to driving environments in **Saudi Arabia and other GCC countries**.

---

## Future Work

Future improvements could include:

- Deployment on Raspberry Pi or NVIDIA Jetson devices
- Mobile application deployment
- Mobile-phone distraction detection
- Head-pose and micro-nod detection
- Infrared/night-driving support
- Larger and more diverse local datasets
- Further model optimization for embedded hardware
- Secure storage and protection of deployed model weights

---

## Project Team

**Qassim University — College of Computer — Computer Science Department**

- Aleen Mohammad Al-Qwaifel
- Joud Ibrahim Al-Saweed
- Raihanah Yousef Al-Salom
- Lana Suliman Al-Deaiji
- Leen Abdulaziz Al-Jasser

**Supervisor:** Dr. Aicha Ben Makhlouf

---

## Academic Project

This project was developed as a **Final Year Graduation Project** in partial fulfillment of the requirements for the degree of **Bachelor of Science in Computer Science at Qassim University**, 2025–2026.

---

## Disclaimer

This project is an academic prototype developed for research and educational purposes.

It should not be considered a certified automotive safety system or used as a replacement for responsible driving, adequate rest, or professionally validated driver-monitoring technologies.
