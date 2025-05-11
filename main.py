from ultralytics import YOLO
from PIL import Image

# Load the model
model_path = 'your path\\best.pt'
model = YOLO(model_path)

image_path = 'your path\\image.png'  # Replace this with the image you want to detect

# Run inference
img = Image.open(image_path)
results = model(img)  # Run inference with the model

# Display the results
results[0].show()  # This shows the image with bounding boxes
