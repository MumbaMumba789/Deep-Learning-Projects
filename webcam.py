
import cv2
import torch
from torchvision import transforms
from PIL import Image
from model_builder import create_model

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
class_names = ["bags", "books"]

# Load trained model
model = create_model(num_classes=2)

model.load_state_dict(
    torch.load("books_bags_model.pth", map_location=device)
)


model.to(device)
model.eval()

# Image transformation (must match testing transformation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Open webcam
camera = cv2.VideoCapture(0)

print("Press SPACE to capture an image.")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access webcam.")
        break

    cv2.imshow("Books vs Bags - Press SPACE to classify", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord(" "):
        # Convert webcam image to PIL
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Transform image
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction
        with torch.inference_mode():
            prediction = model(image_tensor)
            probabilities = torch.softmax(prediction, dim=1)

            confidence, predicted_class = torch.max(
                probabilities, dim=1
            )

        label = class_names[predicted_class.item()]
        confidence_score = confidence.item() * 100

        print(f"Prediction: {label.upper()}")
        print(f"Confidence: {confidence_score:.2f}%")

camera.release()
cv2.destroyAllWindows()
