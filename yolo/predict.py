from ultralytics import YOLO

# Image import
from PIL import Image


# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images from url
# results = model(['https://images.immediate.co.uk/production/volatile/sites/30/2017/07/pineapple-6ee23f3.jpg?quality=90&resize=556,505',"https://farmfreshbangalore.com/cdn/shop/products/i6i3gdx_1500x.jpg?v=1647265311"])  # return a list of Results objects


# Run batched inference on a list of images from file
img1 = Image.open('../food/person/human.jpg')
img2 = Image.open('../food/cake/cake1.jpg')


results = model.predict(source=[img1,img2], save=True,conf=0.5) 

# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs

