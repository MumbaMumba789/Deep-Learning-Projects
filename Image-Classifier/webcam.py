
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

result_text = None
result_confidence = None

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access webcam.")
        break

    capture_frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    prompt = "SHOW ME A BOOK OR A BAG  |  SPACE: CAPTURE"
    prompt_scale = min(0.9, (frame.shape[1] - 30) / max(
        cv2.getTextSize(prompt, font, 0.9, 2)[0][0], 1
    ) * 0.9)
    (prompt_width, prompt_height), prompt_baseline = cv2.getTextSize(
        prompt, font, prompt_scale, 2
    )
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], prompt_height + prompt_baseline + 30),
        (0, 0, 0),
        cv2.FILLED
    )
    cv2.putText(
        frame,
        prompt,
        ((frame.shape[1] - prompt_width) // 2, prompt_height + 15),
        font,
        prompt_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    if result_text is not None:
        result_scale = 1.8
        result_thickness = 5
        result_size = cv2.getTextSize(
            result_text, font, result_scale, result_thickness
        )[0]
        result_scale = min(
            result_scale,
            (frame.shape[1] - 40) / max(result_size[0], 1) * result_scale
        )
        (result_width, result_height), result_baseline = cv2.getTextSize(
            result_text, font, result_scale, result_thickness
        )
        result_y = (frame.shape[0] + result_height) // 2
        cv2.rectangle(
            frame,
            (0, result_y - result_height - 25),
            (frame.shape[1], result_y + result_baseline + 25),
            (0, 0, 0),
            cv2.FILLED
        )
        cv2.putText(
            frame,
            result_text,
            ((frame.shape[1] - result_width) // 2, result_y),
            font,
            result_scale,
            (0, 255, 0),
            result_thickness,
            cv2.LINE_AA
        )
        confidence_text = f"Confidence: {result_confidence:.1f}%"
        confidence_size = cv2.getTextSize(
            confidence_text, font, 0.7, 2
        )[0]
        cv2.putText(
            frame,
            confidence_text,
            ((frame.shape[1] - confidence_size[0]) // 2, result_y + 45),
            font,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imshow("Books vs Bags - Press SPACE to capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    if key == ord(" "):
        image = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            prediction = model(image_tensor)
            probabilities = torch.softmax(prediction, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)

        label = class_names[predicted_class.item()].rstrip("s").upper()
        result_text = f"IT IS A {label}"
        result_confidence = confidence.item() * 100

camera.release()
cv2.destroyAllWindows()
