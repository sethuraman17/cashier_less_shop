import cvzone
from ultralytics import YOLO
import cv2
import numpy as np
import math
from sort import *

cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

model = YOLO('../Yolo-Files/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"]

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker1 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
id_dict = {}
id_dict1 = {}

# mask = cv2.imread('sample.png')

while True:
    success, img = cap.read()
    ret, img1 = cap1.read()
    # imgRegion = cv2.bitwise_and(img, mask)
    results = model(img, stream=True)
    results1 = model(img1, stream=True)
    detections = np.empty((0, 5))
    detections1 = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1, y2-y1

            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.4:
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    resultsTracker = tracker.update(detections)
    for result in resultsTracker:
        x1, y1, x2, y2, tracking_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2-x1, y2-y1
        if tracking_id in id_dict:
            object_id = id_dict[tracking_id]
        else:
            object_id = len(id_dict) + 1
            id_dict[tracking_id] = object_id

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))
        cvzone.putTextRect(img, f'Object ID: {object_id}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    for r in results1:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1, y2-y1

            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.4:
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections1 = np.vstack((detections1, currentArray))
    resultsTracker = tracker1.update(detections1)
    for result in resultsTracker:
        x1, y1, x2, y2, tracking_id1 = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2-x1, y2-y1
        if tracking_id1 in id_dict1:
            object_id = id_dict1[tracking_id1]
        else:
            object_id = len(id_dict1) + 1
            id_dict1[tracking_id1] = object_id

        cvzone.cornerRect(img1, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))
        cvzone.putTextRect(img1, f'Object ID: {object_id}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Images", img)
    cv2.imshow("Image", img1)
    cv2.waitKey(1)
