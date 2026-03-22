from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =========================
# LOAD + PREPROCESS
# =========================
df = pd.read_csv(r"data/imdb_top_1000.csv")
df = df.dropna().reset_index(drop=True)

for col in ['Star1','Star2','Star3','Star4']:
    df[col] = df[col].str.replace(" ", "", regex=False)

df['combined'] = (
    df['Overview'] + " " + df['Genre'] + " " + df['Director'] + " " +
    df['Star1'] + " " + df['Star2'] + " " + df['Star3'] + " " + df['Star4']
)

vectorizer = TfidfVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(df['combined'])
similarity = cosine_similarity(matrix)

df['Series_Title_lower'] = df['Series_Title'].str.lower()

# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/recommend", response_class=HTMLResponse)
def recommend(request: Request, movie: str, min_rating: float = 7.5):
    movie = movie.lower()

    if movie not in df['Series_Title_lower'].values:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error": "Movie not found",
            "recommendations": []
        })

    idx = df[df['Series_Title_lower'] == movie].index[0]
    distances = list(enumerate(similarity[idx]))
    movies = sorted(distances, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in movies[1:]:
        movie_data = df.iloc[i[0]]
        if movie_data['IMDB_Rating'] > min_rating:
            recommendations.append({
                "title": movie_data['Series_Title'],
                "poster": movie_data['Poster_Link'],
                "rating": float(movie_data['IMDB_Rating'])
            })
        if len(recommendations) == 5:
            break

    return templates.TemplateResponse("results.html", {
        "request": request,
        "recommendations": recommendations,
        "movie": movie,
        "min_rating": min_rating
    })