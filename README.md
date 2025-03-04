# ObjectDetectionProjects
Repository for object depository projects
* Thanks to Roboflow https://www.youtube.com/watch?v=etjkjZoG2F0
  
# YOLOv12 Object Detection Pipeline

This script demonstrates how to use the **YOLOv12** model for object detection, including setup, model training, evaluation, and inference. It integrates **Supervision** for visualization and performance metrics.

## Features
- Sets up the environment and checks for GPU availability.
- Installs required dependencies.
- Downloads and preprocesses an example dataset.
- Loads and runs inference using **YOLOv12**.
- Trains YOLOv12 on a custom dataset.
- Evaluates the trained model using **Mean Average Precision (mAP)**.
- Performs inference with the fine-tuned model and visualizes results.

---

## Installation & Setup

### 1️ Install Dependencies
Ensure you have the required packages installed (uncomment in the script if needed).
```bash
pip install git+https://github.com/sunsmarterjie/yolov12.git supervision flash-attn
```

2️ Check GPU Availability
The script attempts to detect an NVIDIA GPU:

python
subprocess.run(["nvidia-smi"])
If not found, a warning is printed.

3️ Set API Key (Optional)
If an API key is required for accessing YOLOv12:

python
os.environ["YOLOV12_API_KEY"] = "your_api_key_here"
Download & Prepare Data
4️ Download Example Image
An example dog image is downloaded for inference.

bash
wget -O dog.jpeg https://media.example.com/notebooks/examples/dog.jpeg

5️ Download Dataset (Replace with actual URL)
bash
wget -O dataset.zip https://your-dataset-link.com
unzip dataset.zip -d dataset
This dataset will be used for training the YOLO model.

6️ Update data.yaml for YOLOv12 Compatibility
bash
sed -i "$d" dataset/data.yaml
echo -e 'test: ../test/images\ntrain: ../train/images\nval: ../valid/images' >> dataset/data.yaml
This ensures the dataset is correctly formatted for YOLOv12 training.

Run YOLOv12 Inference
7️ Load YOLOv12 Model
python
model = YOLO('yolov12l.pt')
8️ Perform Object Detection
python
```
results = model(image, verbose=False)[0]
detections = sv.Detections.from_ultralytics(results)
```
Supervision is used to annotate the image with bounding boxes and labels.

python
```
annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
sv.plot_image(annotated_image)
```
This will display the detected objects on the image.

Train YOLOv12 on a Custom Dataset
9️ Train Model with Custom Data
python
```
model = YOLO('yolov12s.yaml')
results = model.train(data='dataset/data.yaml', epochs=100)
```
The model is trained for 100 epochs using the downloaded dataset.

Model Evaluation
10 Check Training Results
bash
```
ls runs/detect/train/
```
Results are stored in runs/detect/train/.

11 Display Training Performance
The script visualizes:

Confusion Matrix (confusion_matrix.png)
Training Progress (results.png)
python
```
image = cv2.imread(conf_matrix_path)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
```

11 Compute Mean Average Precision (mAP)
mAP is a key metric for object detection performance.

python
```
map = MeanAveragePrecision().update(predictions, targets).compute()
print("mAP 50:95", map.map50_95)
print("mAP 50", map.map50)
print("mAP 75", map.map75)
map.plot()
```
This evaluates the model across different confidence thresholds.

🏁 Run Inference with Fine-Tuned Model
13 Load Best Model
python
```
model = YOLO('runs/detect/train/weights/best.pt')
```
14 Perform Detection on Random Image
python
```
i = random.randint(0, len(ds))
image_path, image, target = ds[i]
```
Runs inference on a random test image.
Uses Non-Maximum Suppression (NMS) to refine detections.
Annotates and displays the detected objects.
python
```
detections = sv.Detections.from_ultralytics(results).with_nms()
annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
sv.plot_image(annotated_image)
```
Summary
This script:

Prepares an object detection dataset.
Trains YOLOv12 on the dataset.
Evaluates performance using mAP metrics.
Runs inference with the trained model.

🔗 References
YOLOv12 GitHub
Supervision Documentation
mAP Metric
Notes
Ensure NVIDIA drivers and CUDA are installed for GPU acceleration.
Replace placeholder dataset URLs with real links.
Adjust training epochs based on dataset size.
