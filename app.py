# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import pickle, pandas as pd, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    with open(os.path.join(BASE, 'model', name), 'rb') as f:
        return pickle.load(f)

movies      = load('movies_df.pkl')
cosine_sim  = load('cosine_sim.pkl')
user_sim_df = load('user_sim_df.pkl')
ratings     = load('ratings_df.pkl')

indices    = pd.Series(movies.index, index=movies['title']).drop_duplicates()
all_titles = sorted(movies['title'].tolist())

def content_recommend(title, top_n=8):
    if title not in indices:
        return []
    idx  = indices[title]
    sims = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    out  = []
    for i, score in sims:
        row = movies.iloc[i]
        out.append({
            'title':       row['title'],
            'genres':      row['genres'].replace('|', ' / '),
            'year':        int(row['year']),
            'rating':      float(row['rating']),
            'director':    row['director'],
            'score':       round(float(score), 3),
            'description': row['description']
        })
    return out

def collab_recommend(user_id, top_n=8):
    user_id = int(user_id)
    if user_id not in user_sim_df.index:
        return []
    similar_users = user_sim_df[user_id].sort_values(ascending=False).iloc[1:6].index.tolist()
    rated_by_user = set(ratings[ratings['userId'] == user_id]['movieId'])
    candidates    = ratings[(ratings['userId'].isin(similar_users)) &
                            (~ratings['movieId'].isin(rated_by_user))]
    if candidates.empty:
        return []
    scores = candidates.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(top_n)
    result = movies[movies['movieId'].isin(scores.index)].copy()
    result = result.merge(scores.rename('predicted_rating').reset_index(), on='movieId')
    out = []
    for _, row in result.iterrows():
        out.append({
            'title':       row['title'],
            'genres':      row['genres'].replace('|', ' / '),
            'year':        int(row['year']),
            'rating':      float(row['rating']),
            'director':    row['director'],
            'score':       round(float(row['predicted_rating']), 2),
            'description': row['description']
        })
    return sorted(out, key=lambda x: x['score'], reverse=True)

def hybrid_recommend(user_id, liked_title, top_n=8):
    cb = content_recommend(liked_title, top_n=20)
    cf = collab_recommend(user_id, top_n=20)
    if not cb:
        return cf[:top_n]
    if not cf:
        return cb[:top_n]
    scaler = MinMaxScaler()
    cb_df  = pd.DataFrame(cb)
    cf_df  = pd.DataFrame(cf)
    cb_df['cb_norm'] = scaler.fit_transform(cb_df[['score']])
    cf_df['cf_norm'] = scaler.fit_transform(cf_df[['score']])
    merged = pd.merge(cb_df[['title','genres','year','rating','director','description','cb_norm']],
                      cf_df[['title','cf_norm']], on='title', how='outer').fillna(0)
    merged['hybrid_score'] = 0.5 * merged['cb_norm'] + 0.5 * merged['cf_norm']
    merged = merged.sort_values('hybrid_score', ascending=False).head(top_n)
    out = []
    for _, row in merged.iterrows():
        out.append({
            'title':       row['title'],
            'genres':      row['genres'],
            'year':        int(row['year']),
            'rating':      float(row['rating']),
            'director':    row['director'],
            'score':       round(float(row['hybrid_score']), 3),
            'description': row['description']
        })
    return out

HTML = open(os.path.join(BASE, 'templates', 'index.html'), 'r', encoding='utf-8').read()

@app.route('/')
def index():
    page = HTML.replace('__MOVIES__', str(all_titles)).replace('__USERS__', str(list(range(1, 11))))
    return page

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data   = request.get_json()
    method = data.get('method', 'content')
    top_n  = int(data.get('top_n', 6))
    try:
        if method == 'content':
            results = content_recommend(data.get('title', ''), top_n)
        elif method == 'collab':
            results = collab_recommend(data.get('user_id', 1), top_n)
        else:
            results = hybrid_recommend(data.get('user_id', 1), data.get('title', ''), top_n)
        return jsonify({'success': True, 'results': results, 'method': method})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("\n  Movie Recommender running at http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
