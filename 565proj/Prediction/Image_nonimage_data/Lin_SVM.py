import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load your dataset
data = pd.read_csv('/nas/longleaf/home/yifzhang/zhengwu/565proj/psc_connect/final_data/merged_dataset_NIH.csv')
X = data.drop(['ID', 'NIH'], axis=1)
y = data['NIH']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identifying categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_train.select_dtypes(exclude=['object', 'category']).columns

# Creating transformers for preprocessing
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))
])

# Combining transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Creating the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('svm', SVC(kernel='linear'))
])

# Setting up the parameter grid
param_grid = {
    'svm__C': [0.01,0.05,0.1, 1, 10, 100]
}

# Running GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Displaying the best parameters
print("Best parameter: ", grid_search.best_params_)

# Predicting on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Printing the classification report
print(classification_report(y_test, y_pred))
