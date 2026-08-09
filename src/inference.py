"""
inference.py

Turns a raw photo into an emotion prediction:
    1. Detect a face in the photo (OpenCV Haar Cascade)
    2. Crop to that face and preprocess it (grayscale, resize to 48x48, normalize)
    3. Run it through the trained EmotionCNN
    4. Return the predicted class + probabilities for all 7 classes

This is the module app.py (the Streamlit app) calls into. It has no
Streamlit-specific code itself, so it can also be tested directly from
the command line or reused elsewhere.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import EmotionCNN

# Class names IN THE SAME ORDER torchvision's ImageFolder assigned them
# during training (it sorts subfolder names alphabetically). This order
# MUST match what was printed during training (get_dataloaders prints
# "Classes (7): [...]") -- if it doesn't, predictions will be mislabeled
# even though the model itself is working correctly.
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Haar Cascade for face detection, shipped with OpenCV itself -- no
# separate download needed.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Same normalization used during training (see dataset.py's eval_transform)
# so the model sees inputs in the same numeric range it was trained on.
PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def load_model(checkpoint_path: str, device: torch.device) -> EmotionCNN:
    """Builds an EmotionCNN and loads trained weights from checkpoint_path."""
    model = EmotionCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # disables dropout/batchnorm training behavior for inference
    return model


def detect_face(image: Image.Image) -> Image.Image:
    """
    Finds the largest face in a PIL image and returns a cropped PIL image
    of just that face. If no face is detected, returns the original image
    unchanged (with a printed warning) so the pipeline still produces a
    prediction rather than crashing -- useful for pre-cropped test images.
    """
    # Haar Cascade works on OpenCV's format: grayscale numpy arrays
    image_np = np.array(image.convert("L"))

    faces = FACE_CASCADE.detectMultiScale(
        image_np, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        print("Warning: no face detected, using full image instead.")
        return image

    # If multiple faces are detected, use the largest one (by bounding box area)
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face_crop = image.crop((x, y, x + w, y + h))
    return face_crop


def predict_emotion(image_input, model: EmotionCNN, device: torch.device) -> dict:
    """
    Runs the full pipeline on an image and returns a dict like:
        {
            "predicted_class": "happy",
            "confidence": 0.82,
            "all_probabilities": {"angry": 0.01, "disgust": 0.00, ...}
        }

    image_input can be either:
        - a file path (str) to an image on disk, OR
        - an already-loaded PIL.Image.Image (e.g. from a Streamlit upload)
    This avoids having to write a file to disk just to pass a path around.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    face = detect_face(image)
    input_tensor = PREPROCESS(face).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)  # convert to probabilities

    predicted_idx = int(torch.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    all_probabilities = {
        CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
    }


if __name__ == "__main__":
    # Quick manual test from the command line:
    #   python inference.py path/to/photo.jpg
    import sys

    if len(sys.argv) != 2:
        print("Usage: python inference.py <path_to_image>")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("../saved_models/best.pt", device)

    result = predict_emotion(sys.argv[1], model, device)

    print(f"\nPredicted emotion: {result['predicted_class']} "
          f"({result['confidence']*100:.1f}% confidence)")
    print("\nAll probabilities:")
    for emotion, prob in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
        print(f"  {emotion:10s}: {prob*100:5.1f}%")
