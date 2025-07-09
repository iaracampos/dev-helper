# Use uma imagem Debian bullseye 
FROM python:3.10-bullseye

# Diretório de trabalho dentro do container
WORKDIR /app

# Atualiza apt e instala ferramentas para build nativo
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copia o requirements.txt para o container
COPY requirements.txt /tmp/requirements.txt

# Instala as dependências Python (incluindo llama-cpp-python)
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copia o restante do código para dentro do container
COPY . /app/

# Cria pasta para os modelos se ainda não existir
RUN mkdir -p /app/models

# Comando padrão para rodar o generator
CMD ["python", "-u", "src/main.py"]
