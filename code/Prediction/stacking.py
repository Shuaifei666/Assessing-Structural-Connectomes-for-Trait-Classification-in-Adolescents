from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

data = pd.read_csv('/nas/longleaf/home/yifzhang/zhengwu/565proj/psc_connect/final_data/merged_dataset_cbcl.csv')
X = data.drop(['ID', 'CBCL'], axis=1)

# 仅保留数值型数据
X = X.select_dtypes(include=[np.number])

# 'y' 作为预测目标（目标变量）
y = data['CBCL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = LogisticRegression(max_iter=10000)
model2 = SVC(probability=True, kernel='rbf')

voting_clf = VotingClassifier(estimators=[('lr', model1), ('svc', model2)], voting='soft')
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred))

stacking_clf = StackingClassifier(estimators=[('lr', model1), ('svc', model2)], final_estimator=LogisticRegression(max_iter=10000))
stacking_clf.fit(X_train, y_train)
y_pred_stack = stacking_clf.predict(X_test)
print("Stacking Classifier Accuracy:", accuracy_score(y_test, y_pred_stack))
