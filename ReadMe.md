# Drone Operation and Safety Project

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive suite of tools for enhancing drone operations and safety using computer vision and automation. This project leverages synthetic dataset generation, YOLO-based real-time object detection, and intelligent command logic for advanced drone functionalities like collision avoidance and automated payload delivery.

---

## Table of Contents

- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
    - [1. Dataset Generation](#1-dataset-generation)
    - [2. Model Training](#2-model-training)
    - [3. Real-time Detection and Operation](#3-real-time-detection-and-operation)
- [License](#license)
- [Authors](#authors)

---

## About The Project

This project is developed to enhance the safety and functionality of drones through the power of computer vision. It provides a complete workflow, from generating synthetic training data to deploying real-time detection models for tasks such as collision avoidance and precision payload drops. By automating critical functions, this project aims to make drone operations more reliable, efficient, and safer.

---

## Key Features

- **Synthetic Dataset Generation:** Scripts to automatically create vast amounts of labeled image data for training object detection models, reducing manual data collection and labeling.
- **YOLO Integration:** Seamless integration with Ultralytics YOLO for high-performance detection of objects, landing zones, and other critical elements.
- **Advanced Collision Avoidance:** Module for estimating distances to objects and implementing safety logic to prevent collisions, including hand landmark detection for safe manual interactions.
- **Automated Payload Drop System:** Detects designated "hotspots" or targets in real-time and executes payload release commands based on the drone's position and distance.
- **Comprehensive Dataset Management:** Tools for preparing, splitting, and managing datasets, streamlining the machine learning pipeline from data creation to model training.

---

## Directory Structure

```
.
├── Collision_Avoidance/
│   ├── distance_estimater.py
│   └── hand_landmarker.task
│
├── Objects_Detection/
│   ├── generate_yolo_dataset.py
│   ├── prepare_yolo_dataset.py
│   ├── shapes.yaml
│   └── yolo_shapes_dataset/
│
├── Payload_Drop/
│   ├── generate_dataset.py
│   ├── generate_yolo_dataset.py
│   ├── hotspot_command.py
│   ├── send_command_advance.py
│   ├── send_command_cv.py
│   ├── send_commands.py
│   ├── split_dataset.py
│   ├── hotspot.yaml
│   ├── hotspot_dataset/
│   ├── hotspot_runs/
│   └── yolo_hotspot_dataset/
│
└── README.md
```

---

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Ensure you have the following software installed on your system:

- Python 3.x
- OpenCV
- Ultralytics YOLO
- NumPy
- PIL (Pillow)

You can install these packages using pip:

```sh
pip install opencv-python ultralytics numpy pillow
```

### Installation

Clone the repository:

```sh
git clone https://github.com/Someone-anon-coder/Drone-Operations.git
```

Navigate to the project directory:

```sh
cd Drone-Operations
```

---

## Usage

The project is divided into several modules. Follow the steps below to utilize its full capabilities.

### 1. Dataset Generation

**For general object detection (shapes):**

Navigate to the `Objects_Detection` directory and run the generation script. This will populate the `yolo_shapes_dataset/` directory.

```sh
cd Objects_Detection/
python generate_yolo_dataset.py
```

**For payload drop hotspots:**

Navigate to the `Payload_Drop` directory. These scripts will create hotspot images and YOLO-compatible labels.

```sh
cd Payload_Drop/
python generate_dataset.py
python generate_yolo_dataset.py
```

After generation, you can split your dataset into training and validation sets:

```sh
python split_dataset.py
```

---

### 2. Model Training

After generating and preparing your datasets, you can train your YOLO models. The specific training command will depend on the Ultralytics YOLO library version you are using. A typical training command would look like this:

```sh
# Example for training a hotspot detection model
yolo train model=yolov8n.pt data=Payload_Drop/hotspot.yaml epochs=100 imgsz=640
```

Make sure the `.yaml` files (`shapes.yaml`, `hotspot.yaml`) are correctly configured with paths to your training and validation sets.

---

### 3. Real-time Detection and Operation

Once you have a trained model, you can run the real-time detection and command logic.

**For automated payload drops:**

The `hotspot_command.py` script uses a trained model to detect hotspots and send commands to the drone.

```sh
cd Payload_Drop/
# Ensure your trained model path is specified within the script
python hotspot_command.py
```

**For collision avoidance:**

The `distance_estimater.py` script can be integrated into your main flight control loop to provide real-time distance estimations from objects detected by the camera.

---

## License

This project is intended for research and educational purposes.

---
