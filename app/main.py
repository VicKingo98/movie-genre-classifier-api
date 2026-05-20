import os

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/sentence_transformers"

from fastapi import FastAPI
from pydantic import BaseModel
from mangum import Mangum
from predict import predict_genres


# =========================================
# CREAR APP
# =========================================

app = FastAPI(
    title="Movie Genre Classifier API",
    version="1.0"
)


# =========================================
# INPUT SCHEMA
# =========================================

class MovieRequest(BaseModel):
    title: str
    plot: str
    year: int


# =========================================
# ROOT
# =========================================

@app.get("/")
def root():
    return {
        "message": "Movie Genre Classifier API running"
    }

# =========================================
# PREDICT ENDPOINT
# =========================================

@app.post("/predict")
def predict(movie: MovieRequest):

    genres = predict_genres(
        title=movie.title,
        plot=movie.plot,
        year=movie.year
    )

    return {
        "predicted_genres": genres
    }
handler = Mangum(app)
