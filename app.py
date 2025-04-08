import cv2
from ultralytics import YOLO


model = YOLO('model/best.pt')  


cap = cv2.VideoCapture('video2.mp4')
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera feed started...")

class_labels = {
    0: 'Gloves',
    1: 'Helmet',
    2: 'Non-Helmet',
    3: 'Person',
    4: 'Shoes',
    5: 'Vest',
    6: 'Bare-Arms'
}

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Error: Failed to capture image.")
        break

    
    results = model(frame, stream=True)

    persons = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = class_labels.get(cls, "Unknown")

            
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            
            if label == "Person":
                persons.append({'box': (x1, y1, x2, y2), 'helmet': False, 'vest': False, 'gloves': False, 'boots': False})
            else:
                
                for person in persons:
                    px1, py1, px2, py2 = person['box']
                    if x1 > px1 and y1 > py1 and x2 < px2 and y2 < py2:
                        if label == "Helmet":
                            person['helmet'] = True
                        elif label == "Vest":
                            person['vest'] = True
                        elif label == "Gloves":
                            person['gloves'] = True
                        elif label == "Boots":
                            person['boots'] = True


    
    cv2.imshow('Worker Safety Detection', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
