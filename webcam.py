
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

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        prediction = model(image_tensor)
        probabilities = torch.softmax(prediction, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    label = class_names[predicted_class.item()].rstrip("s").upper()
    confidence_score = confidence.item() * 100
    display_text = f"{label}  {confidence_score:.1f}%"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 5
    (text_width, text_height), baseline = cv2.getTextSize(
        display_text, font, font_scale, thickness
    )
    cv2.rectangle(
        frame,
        (15, 15),
        (35 + text_width, 35 + text_height + baseline),
        (0, 0, 0),
        cv2.FILLED
    )
    cv2.putText(
        frame,
        display_text,
        (25, 25 + text_height),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    cv2.imshow("Books vs Bags - Live classification", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
