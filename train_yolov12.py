import os
import cv2
import random
import subprocess
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv
from supervision.metrics import MeanAveragePrecision

# Set up environment (Replace with actual API key if needed)
os.environ["YOLOV12_API_KEY"] = "your_api_key_here"
HOME = os.getcwd()
print("Working directory:", HOME)

# Check GPU availability (Colab-specific, replace with subprocess for local use)
try:
    subprocess.run(["nvidia-smi"])
except FileNotFoundError:
    print("NVIDIA-SMI not found. Ensure you have a GPU setup.")

# Install dependencies (if not installed, uncomment)
# subprocess.run(["pip", "install", "git+https://github.com/sunsmarterjie/yolov12.git", "supervision", "flash-attn"])

# Download example image
subprocess.run(["wget", "-O", "dog.jpeg", "https://media.example.com/notebooks/examples/dog.jpeg"])

# Load YOLOv12 model
model = YOLO('yolov12l.pt')
image_path = f"{HOME}/dog.jpeg"
image = cv2.imread(image_path)

# Run inference
results = model(image, verbose=False)[0]
detections = sv.Detections.from_ultralytics(results)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)

# Download dataset (Replace with actual URL)
dataset_url = 'https://your-dataset-link.com'
subprocess.run(["wget", "-O", "dataset.zip", dataset_url])
subprocess.run(["unzip", "dataset.zip", "-d", "dataset"])

# Update data.yaml file for YOLOv12 compatibility
DATASET_PATH = f"{HOME}/dataset"
subprocess.run(["sed", "-i", "$d", f"{DATASET_PATH}/data.yaml"])
subprocess.run(["sed", "-i", "$d", f"{DATASET_PATH}/data.yaml"])
subprocess.run(["sed", "-i", "$d", f"{DATASET_PATH}/data.yaml"])
subprocess.run(["sed", "-i", "$d", f"{DATASET_PATH}/data.yaml"])
subprocess.run(["bash", "-c", f"echo -e 'test: ../test/images\ntrain: ../train/images\nval: ../valid/images' >> {DATASET_PATH}/data.yaml"])

# Train YOLOv12 model
model = YOLO('yolov12s.yaml')
results = model.train(data=f'{DATASET_PATH}/data.yaml', epochs=100)

# Evaluate trained model
print("Evaluation results:")
subprocess.run(["ls", f"{HOME}/runs/detect/train/"])

# Display results
conf_matrix_path = f'{HOME}/runs/detect/train/confusion_matrix.png'
result_path = f'{HOME}/runs/detect/train/results.png'

for img_path in [conf_matrix_path, result_path]:
    image = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

# Compute mean Average Precision (mAP)
ds = sv.DetectionDataset.from_yolo(
    images_directory_path=f"{DATASET_PATH}/test/images",
    annotations_directory_path=f"{DATASET_PATH}/test/labels",
    data_yaml_path=f"{DATASET_PATH}/data.yaml"
)

model = YOLO(f'{HOME}/runs/detect/train/weights/best.pt')

predictions = []
targets = []

for _, image, target in ds:
    results = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    predictions.append(detections)
    targets.append(target)

map = MeanAveragePrecision().update(predictions, targets).compute()

print("mAP 50:95", map.map50_95)
print("mAP 50", map.map50)
print("mAP 75", map.map75)

map.plot()

# Run inference with fine-tuned model
model = YOLO(f'{HOME}/runs/detect/train/weights/best.pt')

i = random.randint(0, len(ds))
image_path, image, target = ds[i]

results = model(image, verbose=False)[0]
detections = sv.Detections.from_ultralytics(results).with_nms()

annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)