#!/bin/bash

# Configuração
PROJECT_DIR=$(pwd)
VENV_NAME=".venv"

echo " Configurando ambiente RAG Dev Helper"

# Instalar dependências do sistema
echo " Instalando dependências do sistema..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip docker.io docker-compose

# Iniciar Redis (Docker)
echo "Iniciando Redis via Docker..."
docker run -d --name redis -p 6379:6379 redis:alpine

# Função para configurar cada serviço
setup_service() {
    local service=$1
    echo -e "\n Configurando $service"
    
    cd "$PROJECT_DIR/$service" || exit
    
    # Criar venv
    python3 -m venv $VENV_NAME
    source $VENV_NAME/bin/activate
    
    # Instalar dependências
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Verificar instalação
    echo -e "\n $service configurado:"
    pip list --format=columns | head -n 5
    deactivate
}

# Configurar cada serviço
setup_service "gateway"
setup_service "retriever"
setup_service "generator"

# Voltar para diretório raiz
cd "$PROJECT_DIR" || exit