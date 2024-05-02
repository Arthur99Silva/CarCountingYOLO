from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import*

cap = cv2.VideoCapture("C:/Users/Arthur/Documents/FaceDetection/FacialDetection/Videos/cars3.mp4")  # Diretorio p/ video

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
 
model = YOLO("../Yolo-Weights/yolov8n.pt")
 
classNames = ["person", "bicycle", "car", "motorbike", "bus", "train", "truck"]

mask = cv2.imread("../CarCountingYOLO/CarCountingYOLO/mask2.png")
 
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            if cls < len(classNames):
                currrenClass = classNames[cls]
                if currrenClass == "car" or currrenClass == "truck" or currrenClass == "bus" or currrenClass == "motorbike" and conf > 0.3:
                    # cvzone.putTextRect(img, f'{currrenClass} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)
                    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt= 5)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)


    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
