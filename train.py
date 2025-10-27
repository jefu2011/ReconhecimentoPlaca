from ultralytics import YOLO
import torch

def main():
    print("Python path:", torch.__file__)
    print("Versão CUDA:", torch.version.cuda)
    print("CUDA disponível:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # modelo base YOLOv8 (pode ser n, s, m, l, x)
    model = YOLO("yolov8n.pt")

    # treino com GPU
    results = model.train(
        data="800/data.yaml",  # caminho do seu arquivo Roboflow
        epochs=30,
        imgsz=640,
        device=0,  # usa GPU 0
        batch=8,
        workers=2  # reduza se tiver pouca RAM
    )

if __name__ == "__main__":
    main()

# import torch
# print("Python path:", torch.__file__)
# print("Versão CUDA:", torch.version.cuda)
# print("CUDA disponível:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))

