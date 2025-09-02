# GraspBot


GraspBot is an intelligent robot arm grasping system that combines **computer vision, natural language understanding, and robotic control**.
It enables users to control a Dobot Magician robotic arm via natural language instructions (e.g., *"pick the green cube"*), with real-time object detection and accurate coordinate mapping between camera space and robot space.

With the integration of **YOLO** for object detection and **Google Gemini API** for natural language understanding, the system supports both simple and complex grasping tasks.

---

## ✨ Features

* **Camera-to-Robot Calibration**: Interactive calibration between pixel coordinates and robot coordinates.
* **Real-Time Object Detection**: Detects colored cubes using a trained YOLOv5 model.
* **Natural Language Command Parsing**:

  * Basic mode: Keyword matching (e.g., *"pick green cube"*).
  * Advanced mode: Gemini-powered reasoning for complex commands (e.g., *"pick the cube between the yellow and black ones"*).
* **Robotic Arm Control**: Executes precise grasping actions using Dobot Magician SDK.
* **Extensible**: Supports both simple grasping (`deploy.py`) and vision-language model–enhanced grasping (`deploy_vlm.py`).

---

## 📂 Project Structure

```
GraspBot/
├── calib.py                 # Camera-to-robot calibration
├── deploy.py                # Basic grasping with YOLOv5 + language commands
├── deploy_vlm.py            # Advanced grasping with Gemini API integration
├── collect_data.py          # Data collection for training
├── convert.py               # Format conversion utilities
├── transform_matrix.npy     # Saved calibration matrix
├── annotations.txt          # Dataset annotation example
├── lib/magician/            # Dobot Magician SDK (DLLs and Python bindings)
├── yolov5/                  # YOLOv5 object detection framework
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Create environment

```bash
conda create -n graspbot python=3.9
conda activate graspbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:

* `torch`, `opencv-python`, `numpy`
* `ultralytics` (YOLOv5)
* `pydobot` or official Dobot DLL SDK
* `google-generativeai` (for Gemini API)

### 3. Configure Gemini API (optional, for `deploy_vlm.py`)

* Create a file `api_key.txt` in the project root.
* Paste your Gemini API key into the **first line**.

---

## 🚀 Usage

### Step 1: Camera–Robot Calibration

Run the calibration script to map camera coordinates to robot coordinates:

```bash
python calib.py
```

* Click **3 reference points** on the camera feed.
* Input the **corresponding robot coordinates** (read from Dobot Studio or Dobot API).
* A transformation matrix will be saved to `transform_matrix.npy`.

---

### Step 2: Run Basic Grasping

For simple keyword-based grasping:

```bash
python deploy.py
```

* The YOLOv5 detector will display the camera feed with detected cubes.
* Enter a command, e.g.:

```
pick the green cube
```

* The robot will execute the grasping action.

---

### Step 3: Run Advanced Grasping (Gemini Extension)

For complex natural language instructions:

```bash
python deploy_vlm.py
```

Example commands:

* `"pick the cube between the yellow and black ones"`
* `"pick all cubes except the green one"`

The Gemini API interprets the command, determines a grasping plan, and the robot executes it.

---

## 📹 Demo

[![GraspBot Demo](https://github.com/aiwtw/GraspBot/blob/main/1.jpg?raw=true)](https://youtu.be/OJhIHHegtQQ)  

👉 Click the image to watch the demo video.

---

## ⚠️ Notes

* Ensure the Dobot Magician is connected via USB and drivers are installed.
* Edit the serial port/connection settings in `deploy.py` and `deploy_vlm.py` if needed.
* Place the camera in a fixed position relative to the robot for consistent calibration.
* If Gemini API is not configured, `deploy_vlm.py` falls back to basic keyword matching.

---

## 🔮 Future Work

* Add **speech input** for voice-controlled grasping.
* Extend to **3D object pose estimation** for more robust grasping.
* Integrate advanced **path planning** and obstacle avoidance.



