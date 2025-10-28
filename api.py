from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
# model = YOLO('runs/detect/train/weights/best.pt')
model = YOLO('best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # results = model(img, imgsz=1280)
    results = model(img, imgsz=1280, conf=0.6, iou=0.4)
    detections = []

    # Considera todas as detecções, independentemente da confiança
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'class': model.names[cls]
        })

    # Ordena da esquerda para a direita
    detections_sorted = sorted(detections, key=lambda d: d['bbox'][0])
    placa = ''.join([d['class'] for d in detections_sorted])

    # Desenhar todas as detecções na imagem para debug
    for d in detections_sorted:
        x1, y1, x2, y2 = map(int, d['bbox'])
        color = (0, 255, 0) if d['confidence'] >= 0.3 else (0, 0, 255)  # verde se confiança alta, vermelho se baixa
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{d['class']}:{d['confidence']:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imwrite("result_debug.jpg", img)

    return jsonify({
        'detections': detections_sorted,
        'placa': placa
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
