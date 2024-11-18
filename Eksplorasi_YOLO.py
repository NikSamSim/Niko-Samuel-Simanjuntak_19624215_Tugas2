import cv2
from ultralytics import YOLO

yolo = YOLO('yolov8s.pt')
videoCap = cv2.VideoCapture('video_YOLO.mp4')

def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors) 
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

while True:
    ret, frame = videoCap.read() 
    if not ret:
        continue
    results = yolo.track(frame, stream=True)

    for result in results:
        classes_names = result.names

        for box in result.boxes:
            if box.conf[0] > 0.4:
                # Mendapatkan koordinat
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cls = int(box.cls[0])  
                class_name = classes_names[cls]  
                colour = getColours(cls)

                # Menggambar kotak
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                
                # Tampilkan nama kelas di atas kotak
                cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    cv2.imshow('Deteksi Objek', frame)

    # Keluar jika tombol spasi (32) atau enter (13) ditekan
    key = cv2.waitKey(1) & 0xFF
    if key in [32, 13]: 
        break

videoCap.release()
cv2.destroyAllWindows()