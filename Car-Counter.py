from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("C:/Users/Arthur/Documents/YOLO_Learning/CarCountingYOLO/CarCountingYOLO/Cars/cars.mp4")  # Directory for video

# Ensure the video is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [225, 400, 1020, 400]

totalCount = []

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "bus", "train", "truck"]

mask = cv2.imread("C:/Users/Arthur/Documents/YOLO_Learning/CarCountingYOLO/CarCountingYOLO/mask2.png")

# Create a matplotlib figure and axis
fig, ax = plt.subplots()

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    # Resize the mask to match the frame dimensions
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

    imgRegion = cv2.bitwise_and(img, mask_resized)

    imgGraphics = cv2.imread("C:/Users/Arthur/Documents/YOLO_Learning/CarCountingYOLO/CarCountingYOLO/graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    
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
                currentClass = classNames[cls]
                if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                    # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt= 5)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # Display the image using matplotlib
    ax.clear()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.pause(0.001)

cap.release()
plt.close()
