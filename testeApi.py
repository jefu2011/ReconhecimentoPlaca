# import requests

# url = "http://127.0.0.1:5000/detect"
# # image_path = "2.JPG"
# # image_path1 = "1.png"
# image_path = "placa.jpg"


# with open(image_path, "rb") as f:
#     files = {"image": f}
#     response = requests.post(url, files=files)

# if response.status_code == 200:
#     data = response.json()
#     detections = data.get('detections', [])
#     placa = data.get('placa', '')

#     print("Detecções recebidas:")
#     for det in detections:
#         print(det)

#     print("\nPlaca detectada:", placa)

# else:
#     print(f"Erro {response.status_code}: {response.text}")
import requests
import cv2

# URL da API
url = "http://127.0.0.1:5000/detect"
image_path = "placa.jpg"

# Abrir a imagem e enviar para a API
with open(image_path, "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    detections = data.get('detections', [])
    placa = data.get('placa', '')

    print("Detecções recebidas:")
    for det in detections:
        print(f"Classe: {det['class']}, Confiança: {det['confidence']:.2f}, BBox: {det['bbox']}")

    print("\nPlaca detectada:", placa)

    # Desenhar detecções na imagem
    img = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        color = (0, 255, 0) if det['confidence'] >= 0.3 else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{det['class']}:{det['confidence']:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Salvar imagem de resultado
    cv2.imwrite("result_test.jpg", img)
    print("Imagem com detecções salva em 'result_test.jpg'.")

else:
    print(f"Erro {response.status_code}: {response.text}")
