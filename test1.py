import cv2
import torch
import clip
import numpy as np
from PIL import Image
import os

SETTINGS = {
    "model-settings": {
        "prediction-threshold": 0.26,  # increased sensitivity to reduce false negatives
        "model-name": "ViT-B/32",
        "device": "cpu"
    },
    "label-settings": {
        "labels": [
            # Violence (outdoor)
            "a violent fight happening on a street",
            "street fight",
            "people attacking each other",
            "physical violence in public",

            # Violence (indoor)
            "violence happening in an office",
            "office fight",
            "people fighting indoors",

            # Fire
            "a fire burning in a building",
            "fire on the street",
            "a fire inside an office",
            "flames and smoke",
            "room filled with smoke",

            # Accidents
            "a car crash on the road",
            "vehicles collided",
            "road accident",

            # Safe cases
            "normal office environment",
            "people talking peacefully",
            "people walking normally on a street",
            "people walking normally in office",
            "group of people in office"
        ],
        "default-label": "Unknown"
    }
}




# CLIP-based model
class ViolenceModel:
    def __init__(self, settings: dict):
        self.device = settings['model-settings']['device']
        self.model_name = settings['model-settings']['model-name']
        self.threshold = settings['model-settings']['prediction-threshold']
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.labels = settings['label-settings']['labels']
        self.default_label = settings['label-settings']['default-label']
        self.prompts = ['a photo of ' + label for label in self.labels]
        self.text_features = self.vectorize_text(self.prompts)

    @torch.no_grad()
    def vectorize_text(self, text_list):
        tokens = clip.tokenize(text_list).to(self.device)
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def transform_image(self, image: np.ndarray):
        pil_image = Image.fromarray(image).convert('RGB')
        return self.preprocess(pil_image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        tf_image = self.transform_image(image)
        image_features = self.model.encode_image(tf_image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T
        values, indices = similarity[0].topk(1)
        confidence = values[0].item()
        label = self.default_label
        if confidence >= self.threshold:
            label = self.labels[indices[0].item()]
        return {"label": label, "confidence": confidence}


# -----------------------
# Real-time Webcam Loop
# -----------------------
def main():
    model = ViolenceModel(SETTINGS)

    cap = cv2.VideoCapture(0)
 
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
# ...existing code...
        result = model.predict(rgb_frame)
        label, confidence = result['label'], result['confidence']

        # Choose color based on label
        if "violence" in label or "fight" in label or "fire" in label:
            color = (0, 0, 255)  # Red for violence
        else:
            color = (0, 200, 0)  # Green for non-violence

        display_text = f"{label} ({confidence:.2f})"

        # Draw background rectangle for text
        (text_w, text_h), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (15, 10), (15 + text_w + 10, 10 + text_h + baseline + 10), (30, 30, 30), -1)
        cv2.putText(frame, display_text, (20, 30 + text_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Draw confidence bar
        bar_x, bar_y, bar_w, bar_h = 15, 50 + text_h, 200, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 2)
        fill_w = int(bar_w * min(confidence, 1.0))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

        cv2.imshow("Violence Detection - CLIP", frame)
# ...existing code...

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
