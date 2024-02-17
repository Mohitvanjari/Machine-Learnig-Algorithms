import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#sample movie data with genre tags and user ratings
movies = pd.DataFrame({
    'movie_title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'genre_action': [1, 1, 0, 1, 0],
    'genre_adventure': [1, 0, 1, 0, 1],
    'genre_comedy': [0, 1, 1, 0, 0],
    'genre_drama': [0, 0, 0, 1, 1],
    'user_rating': [5, 4, 3, 2, 1]
})

print(movies.head())

#split the data into features (genre tags) and labels (user ratings)
x = movies.drop(['movie_title', 'user_rating'], axis=1)
y = movies['user_rating']

#training the model
clf = MultinomialNB()
clf.fit(x, y)

new_movie = pd.DataFrame({
    'genre_action': [1],
    'genre_adventure': [1],
    'genre_comedy': [0],
    'genre_drama': [0]
})

user_rating_pred = clf.predict(new_movie)
print("Predicted user rating for the new movie:", user_rating_pred[0])