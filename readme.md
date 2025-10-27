🚗 ReconhecePlacas API

Microserviço Python para detecção de placas de veículos utilizando YOLOv11 da Ultralytics. Pode rodar em CPU ou GPU (CUDA). A API é construída com FastAPI e recebe imagens via HTTP POST, retornando coordenadas das caixas delimitadoras, classes e scores.

🗂 Estrutura do projeto
reconhecePlacas/
│
├─ yolov11n.pt           # Modelo pré-treinado YOLOv11
├─ main.py               # API FastAPI
├─ dataset/              # Opcional: imagens para teste
├─ requirements.txt      # Dependências Python
├─ yolov11-env/          # Ambiente virtual Python
└─ README.md             # Documentação

⚡ Requisitos

Python 3.11+

pip instalado

GPU NVIDIA com CUDA (opcional, mas acelera treino/inferência)

Internet para instalar dependências

1️⃣ Criar ambiente virtual
cd C:\caminho\para\reconhecePlacas
python -m venv yolov11-env


Ativar ambiente:

PowerShell:

.\yolov11-env\Scripts\Activate.ps1


CMD:

.\yolov11-env\Scripts\activate.bat

2️⃣ Instalar dependências

Crie requirements.txt com:

fastapi
uvicorn[standard]
torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
ultralytics==8.3.147
opencv-python
numpy
Pillow


Instale:

pip install --upgrade pip
pip install -r requirements.txt


💡 Se não tiver GPU, pode instalar apenas torch, torchvision, torchaudio sem CUDA.

3️⃣ Estrutura do código principal (main.py)

A API tem um endpoint:

POST /detect → Recebe uma imagem e retorna as detecções.

Exemplo mínimo:

from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()
model = YOLO("yolov11n.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    detections = []
    for r in results:
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(score),
                "class": int(cls)
            })
    return {"detections": detections}

4️⃣ Rodar a API

Com o ambiente virtual ativado:

uvicorn main:app --host 0.0.0.0 --port 8000


A API estará disponível em:

http://127.0.0.1:8000

5️⃣ Testar a API
Via curl
curl -X POST "http://127.0.0.1:8000/detect" -F "file=@placa.jpg"

Via Python
import requests

url = "http://127.0.0.1:8000/detect"
files = {"file": open("placa.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())

Exemplo de resposta esperada
[
    {
        "bbox": [100.5, 50.2, 300.8, 120.7],
        "confidence": 0.92,
        "class": "A"
    },
    {
        "bbox": [110.0, 50.5, 310.1, 120.9],
        "confidence": 0.95,
        "class": "B"
    },
    {
        "bbox": [120.2, 51.0, 320.0, 121.3],
        "confidence": 0.93,
        "class": "C"
    }
]


Interpretação:

bbox: coordenadas da caixa delimitadora [x1, y1, x2, y2] em pixels

confidence: confiança do modelo para essa detecção (0 a 1)

class: caractere detectado (0-9 ou letra, conforme seu dataset)

Para reconstruir a placa completa:

Ordene as caixas pelo valor x1 (da esquerda para direita).

Concatene os caracteres na ordem.

Exemplo em Python:

detections = response.json()
detections_sorted = sorted(detections, key=lambda d: d['bbox'][0])
placa = ''.join([d['class'] for d in detections_sorted])
print("Placa detectada:", placa)

6️⃣ Usar GPU ou CPU

A API detecta automaticamente se há GPU CUDA disponível.

Para forçar CPU:

model = YOLO("yolov11n.pt", device="cpu")

7️⃣ Treinamento (opcional)

Se quiser treinar o modelo com seu próprio dataset:

Criar arquivo data.yaml com caminhos das imagens e classes.

Rodar:

yolo detect train data=dataset/data.yaml model=yolov11n.pt epochs=100 imgsz=640 device=0


device=0 → primeira GPU

device='0,1' → múltiplas GPUs

device='cpu' → força CPU

8️⃣ Dicas

Reduza batch_size se a GPU tiver pouca memória.

O aviso AMP é normal em GPUs pequenas (como Quadro T1000) e não impede o treinamento.

Use nvidia-smi para monitorar uso da GPU.