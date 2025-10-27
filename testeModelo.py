from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

# Teste com uma imagem
results = model.predict(source='placa.jpg', show=True, conf=0.5)
