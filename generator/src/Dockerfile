FROM python:3.10-slim

WORKDIR /app

# Instalar dependências de compilação
RUN apt-get update && \
    apt-get install -y build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Permitir instalação de pacotes do sistema
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Instalar dependências com build otimizado
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicação
COPY ./src ./src

# Criar diretório para modelos
RUN mkdir -p /app/models

CMD ["python", "src/main.py"]