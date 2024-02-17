from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Sample data (you can replace this with your own dataset)
documents = ["good bad good", "bad good bad", "good good good", "bad bad good", "good bad bad"]
labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

# Vectorize the documents using CountVectorizer with binary=True for Bernoulli model
vectorizer = CountVectorizer(binary=True)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create and train the Bernoulli Naive Bayes classifier
clf = BernoulliNB()
clf.fit(X_train_vectorized, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test_vectorized)

# Evaluate the classifier
accuracy = metrics.accuracy_score(y_test, predictions)
precision = metrics.precision_score(y_test, predictions)
recall = metrics.recall_score(y_test, predictions)
f1_score = metrics.f1_score(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
