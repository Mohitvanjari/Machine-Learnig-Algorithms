import numpy as np

n = 100
credit_scores = np.random.normal(loc=650, scale=100, size=n)
income = np.random.normal(loc=50000, scale=10000, size=n)
default = np.zeros(n)

default_idx = np.random.choice(range(n), size=20, replace=False)
default[default_idx] = 1

dataset = np.column_stack((credit_scores, income, default))

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset[:, :2], 
                                                    dataset[:, 2], 
                                                    test_size=0.2, random_state=42)

from sklearn.svm import SVC

# Train an SVM model on the training set
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)