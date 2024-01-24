from ultralytics import YOLO
# import torch
# torch.cuda.set_device(0)
# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

# # Evaluate the model's performance on the validation set
results = model.val()

# # Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')
# Print  result
# print("Inference Results:", results)

# Export the model to ONNX format
# success = model.export(format='tflite')
