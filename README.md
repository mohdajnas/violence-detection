
# CLIP-Based Real-Time Violence & Incident Detection

This project uses OpenAI's CLIP model (`ViT-B/32`) for **real-time detection of violence, fires, and accidents** from webcam or RTSP video streams. It classifies incoming frames based on similarity to a set of textual prompts and saves frames with dangerous content.

---

## 🎯 Features

- ✅ Uses [CLIP](https://openai.com/research/clip) (`ViT-B/32`) for zero-shot vision classification
- 🔥 Detects **violence, fire, accidents**, and normal scenes
- 🖼️ Saves detected "dangerous" frames automatically
- 📦 Works with **webcam** or **RTSP camera feeds**
- 📊 Displays real-time classification and confidence bar in OpenCV window

---

## 🧰 Tech Stack

- Python 3.10+
- PyTorch
- OpenCV
- [CLIP (OpenAI)](https://github.com/openai/CLIP)
- NumPy, PIL

---

## 📦 Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
````

---

## 📁 Project Structure

```
violence_detector.py       # Main script
violence_frames/           # Saved images of detected violent/fire scenes
```

---

## 📽️ Running the App

### 🔹 With Webcam (default):

```bash
python vdframe.py
```

### 🔸 With RTSP Stream (edit `main()`):

Uncomment and update:

```python
RTSP_URL = 'rtsp://username:password@ip:port/stream'
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
```

---

## 📋 Labels & Prompts

The model compares video frames with this list of prompts (edit as needed):

```python
[
  "a violent fight happening on a street",
  "street fight",
  "violence happening in an office",
  "a fire burning in a building",
  "a car crash on the road",
  "normal office environment",
  ...
]
```

Any label containing keywords like `"violence"`, `"fight"`, or `"fire"` triggers saving the frame.

---

## 🧠 How It Works

1. Frame is resized and converted to RGB
2. CLIP encodes the image and compares with pre-encoded prompt features
3. Most similar label is selected with a confidence score
4. Dangerous frames are saved to `violence_frames/`
5. A live window shows label and confidence bar

---

## ⚙️ Configuration

You can adjust settings at the top of the script:

```python
"prediction-threshold": 0.26,      # Confidence threshold
"model-name": "ViT-B/32",          # CLIP model
"device": "cpu"                    # Use "cuda" if available
```

---

## 🧪 Example Output

```
Label: street fight (0.42)
[Saved] violence_frames/street_fight_20250515_101530_123456.jpg
```

OpenCV window shows:

* Label and confidence
* Color-coded feedback (red for danger, green for safe)

---

## 🧼 Exit

Press `q` in the OpenCV window to stop the stream.

---

## 🛠️ TODO / Extensions

* [ ] Add audio-based violence detection
* [ ] Push alerts (e.g., email/SMS/Telegram)
* [ ] Integrate into Streamlit or Flask app
* [ ] Upload saved frames to cloud or dashboard

---

## 📄 License

MIT License

```

Let me know if you want to turn this into a Streamlit dashboard, deploy it as a web app, or add object/person detection alongside CLIP!
```
