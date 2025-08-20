from ultralytics import YOLO
import os 

source = "../videos/video5.mp4"
weights = "../runs/detect/train/weights/best.pt" # ya da kendi hazır coco model için "yolo11m.pt" 
tracker = "botsort.yaml"

imgsz = 512
iou = 0.5
 
model_tag = os.path.splitext(os.path.basename(weights))[0]
print(model_tag)
if model_tag.startswith("yolo"):
    model_tag = "coco_model_track" 
else:
    model_tag = "track"
    
tracker_tag = os.path.splitext(os.path.basename(tracker))[0]
     

project_dir = os.path.join("../runs/", model_tag, tracker_tag)

model = YOLO(weights)
print(model.names)

results = model.track(
    source=source,
    tracker=tracker,
    iou = iou,
    imgsz=imgsz,
    persist=True,
    save=True,
    project=project_dir,
    name="ppe_track",
    show=False,
    save_txt=False,
    save_conf=False
)