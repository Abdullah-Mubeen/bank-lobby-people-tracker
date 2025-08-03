# 🏦 Bank Lobby People Tracker

A real-time people-tracking system for bank lobbies using **YOLOv8**. The system detects and tracks individuals, assigns **consistent IDs**, and counts how many people are **waiting in a designated seating area (ROI)**.

> ✅ Clean ID tracking  
> ✅ Region-of-Interest (ROI) detection  
> ✅ Video output with bounding boxes and IDs  
> ✅ Fast and lightweight using `yolov8m`  

---

## 🖥️ Demo

![demo](demo/screenshot.png)

> People seated in the sofa area are counted and tracked with persistent IDs. The ROI is defined manually in `roi_config.json`.

---

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bank-lobby-people-tracker.git
cd bank-lobby-people-tracker
```

### 2. Create Virtual Environment & Install Packages
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

```

### 3. ▶️ Run the System

```bash
python main.py
```