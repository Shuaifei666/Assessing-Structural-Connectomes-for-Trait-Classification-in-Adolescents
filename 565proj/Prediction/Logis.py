from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

data = pd.read_csv('/nas/longleaf/home/yifzhang/zhengwu/565proj/psc_connect/final_data/merged_dataset_cbcl.csv')
X = data.drop(['ID', 'CBCL'], axis=1)

X = X.select_dtypes(include=[np.number])
y = data['CBCL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('logreg', LogisticRegression(solver='liblinear',max_iter=10000))
])

param_grid = {
    'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #'logreg__C': [10],
    'logreg__penalty': ['l1', 'l2']
}
print("Starting grid search with SS...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameter: ", grid_search.best_params_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))
