from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from difflib import get_close_matches

# ------------------ PATH SETUP ------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "imdb_top_1000.csv")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------------ LOAD DATA ------------------
df = pd.read_csv(file_path)
df = df.dropna().reset_index(drop=True)

# ------------------ CLEAN STAR NAMES ------------------
for col in ['Star1', 'Star2', 'Star3', 'Star4']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(" ", "", regex=False)
        df[col] = df[col].str.replace(":", "", regex=False)

# ------------------ COMBINE FEATURES ------------------
df['combined'] = (
    df['Overview'].astype(str) + " " +
    df['Genre'].astype(str) + " " +
    df['Director'].astype(str) + " " +
    df['Star1'].astype(str) + " " +
    df['Star2'].astype(str) + " " +
    df['Star3'].astype(str) + " " +
    df['Star4'].astype(str)
)

# ------------------ TF-IDF ------------------
vectorizer = TfidfVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(df['combined'])
similarity = cosine_similarity(matrix)

# ------------------ CLEAN TITLE (FIXED) ------------------
def clean_text(x):
    return re.sub(r'[^a-z0-9]', '', str(x).lower())

df['Series_Title_clean'] = df['Series_Title'].apply(clean_text)

# ------------------ ROUTES ------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/recommend", response_class=HTMLResponse)
def recommend(request: Request, movie: str, min_rating: float = 6.5):

    movie_clean = clean_text(movie)

    # ------------------ EXACT MATCH ------------------
    if movie_clean in df['Series_Title_clean'].values:
        idx = df[df['Series_Title_clean'] == movie_clean].index[0]

    else:
        # ------------------ FUZZY MATCH ------------------
        matches = get_close_matches(movie_clean, df['Series_Title_clean'], n=1, cutoff=0.6)

        if not matches:
            return templates.TemplateResponse("results.html", {
                "request": request,
                "error": "Movie not found",
                "recommendations": []
            })

        matched_title = matches[0]
        idx = df[df['Series_Title_clean'] == matched_title].index[0]

    # ------------------ SIMILARITY ------------------
    distances = list(enumerate(similarity[idx]))
    movies = sorted(distances, key=lambda x: x[1], reverse=True)

    # ------------------ RECOMMENDATIONS ------------------
    recommendations = []

    for i in movies[1:]:
        movie_data = df.iloc[i[0]]

        try:
            rating = float(movie_data['IMDB_Rating'])
        except:
            continue

        if rating >= min_rating:
            recommendations.append({
                "title": movie_data['Series_Title'],
                "poster": movie_data['Poster_Link'],
                "rating": rating
            })

        if len(recommendations) >= 10:
            break

    # ------------------ FALLBACK ------------------
    if not recommendations:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error": "No recommendations found. Try lowering rating filter.",
            "recommendations": []
        })

    return templates.TemplateResponse("results.html", {
        "request": request,
        "recommendations": recommendations,
        "movie": movie,
        "min_rating": min_rating
    })