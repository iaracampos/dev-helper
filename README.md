# Dev Helper

O *Dev Helper* é uma aplicação baseada na arquitetura *RAG (Retrieval-Augmented Generation)* com foco em auxiliar desenvolvedores a obter respostas mais precisas e contextualizadas por meio de uma interface conversacional inteligente.

O sistema está dividido em três componentes principais: gateway, retriever e generator, organizados com uso de contêineres Docker para garantir modularidade, portabilidade e escalabilidade.

---

## 🧠 Arquitetura

A arquitetura segue o padrão *A2A (Application-to-Application)* e é composta pelos seguintes módulos:


[ Client ] --> [ Gateway (API REST) ] --> [ Retriever ] --> [ Generator ]


- *Client:* Interface de uso (ex: curl, Postman, frontend etc.)
- *Gateway:* Responsável por receber as requisições do cliente via API REST e orquestrar os módulos internos.
- *Retriever:* Responsável por buscar o contexto relevante (documentos, embeddings) com base na pergunta recebida.
- *Generator:* Utiliza o contexto fornecido para gerar uma resposta com base em LLMs (Large Language Models).

---

## 🚀 Tecnologias

- Python
- FastAPI
- Docker
- Docker Compose

---

## 📦 Instalação

1. *Clone o repositório:*
bash
git clone https://github.com/seu-usuario/dev-helper.git
cd dev-helper


2. *Construa e suba os contêineres:*
bash
docker-compose up --build


3. *Acesse a aplicação:*
- A API estará disponível em http://localhost:8000
---

## 🔁 Fluxo de funcionamento

1. O usuário envia uma pergunta para a API REST no gateway.
2. O gateway envia a pergunta ao retriever, que consulta a base de conhecimento e retorna os documentos relevantes.
3. O gateway repassa o contexto recuperado ao generator.
4. O generator utiliza uma LLM (Llama) para formular uma resposta baseada no contexto.
5. A resposta é enviada de volta ao cliente.

---

## 🧪 Exemplo de uso

### Enviar uma pergunta:

bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{ "question": "Como funciona a arquitetura RAG?" }'


---

## 👤 Alunos
- Iara Campos Rodrigues
-Fernando J. Gregatti Noronha