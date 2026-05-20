# Movie Genre Classifier API

API REST para clasificación de géneros de películas usando NLP y FastAPI.

## Tecnologías
- Python
- FastAPI
- Docker
- Scikit-learn
- Sentence Transformers

## Estructura del proyecto

movie-genre-classifier-api/
│
├── app/
│   ├── main.py
│   ├── predict.py
│   └── models/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
└── .gitignore

## Cómo ejecutar localmente

### 1. Clonar repositorio

git clone <repo-url>

### 2. Construir imagen Docker

docker build -t movie-genre-appi .

### 3. Ejecutar contenedor

docker run -p 8000:8000 movie-genre-appi

## Documentación Swagger

http://localhost:8000/docs

## Ejemplo de request

POST /predict

{
  "title": "Batman Begins",
  "plot": "Bruce Wayne becomes Batman and fights crime in Gotham City",
  "year": 2005
}

## Ejemplo de respuesta

{
  "predicted_genres": [
    "Action",
    "Crime"
  ]
}
