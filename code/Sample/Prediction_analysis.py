# Import the necessary packages
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

# Load data
try:
    data = pd.read_csv('merged_dataset_NIH.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("File not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("No data found. Please check the file.")
except Exception as e:
    print(f"An error occurred: {e}")

print(data.head())
print(data.info())
print(data.describe())

X = data.drop(['ID', 'NIH'], axis=1)

X = X.select_dtypes(include=[np.number])
#Target variable y 
y = data['NIH']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters You can change it based on your own dataset
n_components = 50  # Number of principal components for PCA
n_estimators = 100  # Number of trees in the Random Forest
threshold = "median"  # Threshold for feature selection, retaining half of the features


# Define the pipeline
pipeline = Pipeline(steps=[
   # ('scaler', StandardScaler()),  # Standardization step, can be removed if not needed
    ('pca', PCA(n_components=n_components)),  # PCA for dimensionality reduction
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=n_estimators), threshold=threshold)),  # Feature selection
    #('classifier', SVC(kernel='linear' , probability=True))  # SVM_linear classifier
   # ('logreg', LogisticRegression(solver='liblinear',max_iter=10000)) # Logistic classifier
   ('classifier', SVC(kernel='rbf', probability=True)) 
   # ('classifier', SVC(kernel='poly' , probability=True))# SVM_poly classifier
])

# Define the parameter grid

##SVM_Lin
param_grid_SVMlin = {
    'classifier__C': [0.01, 0.05, 0.1, 1, 10,100]
}

##SVM_Logistic
param_grid_logistic = {
    'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'logreg__penalty': ['l1', 'l2']# L1_Lasso, L2_Ridge
}

##SVM_RBF
param_grid_SVMRBF = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1, 10, 100]
}
##SVM_Poly
param_grid_Poly = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__degree': [2, 3, 4, 5],
    'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
    'classifier__coef0': [0.0, 0.1, 0.5, 1.0]  # Independent term in kernel function
}

#Choose your parameters
param_grid=param_grid_SVMRBF

grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameter: ", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Predict the test set
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Print the classification report
print(classification_report(y_test, y_pred))

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
