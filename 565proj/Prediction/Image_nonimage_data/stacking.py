from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

data = pd.read_csv('/nas/longleaf/home/yifzhang/zhengwu/565proj/psc_connect/final_data/merged_dataset_NIH.csv')
X = data.drop(['ID', 'NIH'], axis=1)
y = data['NIH']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_train.select_dtypes(exclude=['object', 'category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),  # 数值特征不做改变
        ('cat', OneHotEncoder(), categorical_cols)  # 对分类特征进行独热编码
    ])


model1 = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', LogisticRegression(max_iter=10000))])

model2 = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', SVC(probability=True, kernel='rbf'))])

voting_clf = VotingClassifier(estimators=[('lr', model1), ('svc', model2)], voting='soft')
voting_clf.fit(X_train, y_train)


y_pred = voting_clf.predict(X_test)
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred))
