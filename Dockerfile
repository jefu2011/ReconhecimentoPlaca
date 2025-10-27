# Use a imagem base com suporte CUDA
FROM nvidia/cuda:12.4.1-cudnn8-runtime-ubuntu22.04

# Evitar interação do apt
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências básicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip python3-dev \
    git curl wget unzip \
    libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Criar diretório do app
WORKDIR /app

# Copiar requirements.txt
COPY requirements.txt .

# Atualizar pip e instalar dependências
RUN python3.11 -m pip install --upgrade pip

# Instalar PyTorch com CUDA, Ultralytics e dependências
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copiar o restante da aplicação
COPY . .

# Expor porta da API
EXPOSE 8000

# Comando para rodar a API (FastAPI/uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
