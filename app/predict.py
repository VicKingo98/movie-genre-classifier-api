import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sentence_transformers import SentenceTransformer


# =====================================
# CARGAR ARTEFACTOS
# =====================================

vect_word = joblib.load('models/vect_word.pkl')

vect_char = joblib.load('models/vect_char.pkl')

scaler_year = joblib.load('models/scaler_year.pkl')

clf_tfidf = joblib.load('models/clf_tfidf.pkl')

clf_embeddings = joblib.load('models/clf_embeddings.pkl')

mlb = joblib.load('models/mlb.pkl')


# =====================================
# CARGAR MODELO EMBEDDINGS
# =====================================

model_emb = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =====================================
# PESOS FINALES ENSEMBLE
# =====================================

BEST_TFIDF_WEIGHT = 0.40

BEST_EMBEDDING_WEIGHT = 0.60


# =====================================
# FUNCIÓN PREDICCIÓN
# =====================================

def predict_genres(title, plot, year):

    # ==============================
    # TEXTO
    # ==============================

    text = f"{title} {plot}"

    # ==============================
    # TF-IDF
    # ==============================

    sample_word = vect_word.transform([text])

    sample_char = vect_char.transform([text])

    sample_year = scaler_year.transform(
	pd.DataFrame([[year]], columns=["year"])
	)	

    sample_year = csr_matrix(sample_year)

    sample_tfidf = hstack([
        sample_word,
        sample_char,
        sample_year
    ])

    pred_tfidf = clf_tfidf.predict_proba(
        sample_tfidf
    )

    # ==============================
    # EMBEDDINGS
    # ==============================

    sample_emb = model_emb.encode([text])

    pred_emb = clf_embeddings.predict_proba(
        sample_emb
    )

    # ==============================
    # ENSEMBLE
    # ==============================

    pred_final = (
        BEST_TFIDF_WEIGHT * pred_tfidf +
        BEST_EMBEDDING_WEIGHT * pred_emb
    )

    # ==============================
    # TOP GÉNEROS
    # ==============================

    threshold = 0.30

    genres = [
        mlb.classes_[i]
        for i, p in enumerate(pred_final[0])
        if p >= threshold
    ]

    return genres
