from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.staticfiles import StaticFiles
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "imdb_top_1000.csv")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------------ LOAD DATA ------------------
df = pd.read_csv(file_path)
df = df.dropna().reset_index(drop=True)

# ------------------ CLEAN STAR NAMES ------------------
for col in ['Star1','Star2','Star3','Star4']:
    df[col] = df[col].str.replace(" ", "", regex=False)
    df[col] = df[col].str.replace(":", "", regex=False)

# ------------------ COMBINE FEATURES ------------------
df['combined'] = (
    df['Overview'] + " " + df['Genre'] + " " + df['Director'] + " " +
    df['Star1'] + " " + df['Star2'] + " " + df['Star3'] + " " + df['Star4']
)

# ------------------ TF-IDF ------------------
vectorizer = TfidfVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(df['combined'])
similarity = cosine_similarity(matrix)

# ------------------ CLEAN TITLE (FULL NORMALIZATION) ------------------
df['Series_Title_clean'] = df['Series_Title'].apply(
    lambda x: re.sub(r'[:$^a-z0-9]', '', str(x).lower())
)

# ------------------ ROUTES ------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/recommend", response_class=HTMLResponse)
def recommend(request: Request, movie: str, min_rating: float = 7.5):

    # Clean user input same way
    movie = re.sub(r'[^a-z0-9]', '', movie.lower())

    # Check existence
    if movie not in df['Series_Title_clean'].values:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error": "Movie not found",
            "recommendations": []
        })

    # Get index
    idx = df[df['Series_Title_clean'] == movie].index[0]

    # Similarity
    distances = list(enumerate(similarity[idx]))
    movies = sorted(distances, key=lambda x: x[1], reverse=True)

    # Recommendations
    recommendations = []
    for i in movies[1:]:
        movie_data = df.iloc[i[0]]

        if movie_data['IMDB_Rating'] > min_rating:
            recommendations.append({
                "title": movie_data['Series_Title'],
                "poster": movie_data['Poster_Link'],
                "rating": float(movie_data['IMDB_Rating'])
            })

        if len(recommendations) == 10:
            break

    return templates.TemplateResponse("results.html", {
        "request": request,
        "recommendations": recommendations,
        "movie": movie,
        "min_rating": min_rating
    })