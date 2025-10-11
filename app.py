import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

# Download NLTK data if not present (PorterStemmer doesn't require extra, but ensure)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@st.cache_data
def load_and_preprocess_data():
    # Load data
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    
    # Merge
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # Handle missing values
    movies.dropna(inplace=True)
    movies.drop_duplicates(inplace=True)
    
    # Helper functions
    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    
    def convert3(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L
    
    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L
    
    # Apply conversions
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    # Split overview
    movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
    
    # Remove spaces
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    
    # Create tags
    movies['tags'] = (movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'])
    
    # New dataframe
    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    # Stemming
    ps = PorterStemmer()
    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)
    
    new_df['tags'] = new_df['tags'].apply(stem)
    
    # Vectorization
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    
    # Similarity
    similarity = cosine_similarity(vectors)
    
    return new_df, similarity

@st.cache_data
def recommend(movie_title, new_df, similarity):
    if movie_title not in new_df['title'].values:
        return []
    
    movie_index = new_df[new_df['title'] == movie_title].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movie_list:
        title = new_df.iloc[i[0]].title
        recommended_movies.append(title)
    
    return recommended_movies

# Streamlit UI
st.title('Movie Recommender System')
st.write('Enter a movie title to get personalized recommendations based on content similarity.')

new_df, similarity = load_and_preprocess_data()

movie_list = sorted(new_df['title'].unique())

selected_movie = st.selectbox('Select a movie:', movie_list)

if st.button('Recommend'):
    recommendations = recommend(selected_movie, new_df, similarity)
    if recommendations:
        st.subheader(f'Recommendations for {selected_movie}:')
        for title in recommendations:
            st.write(title)
    else:
        st.write('Movie not found in the dataset.')
