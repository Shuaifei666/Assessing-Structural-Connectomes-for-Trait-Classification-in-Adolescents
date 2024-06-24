from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np

scaler = StandardScaler()

pca = PCA(n_components=50)

rf = RandomForestClassifier(n_estimators=100)
select = SelectFromModel(rf, threshold="median") 

svm = SVC(kernel='linear')

pipeline = Pipeline(steps=[('scaler', scaler), 
                           ('pca', pca), 
                           ('feature_selection', select), 
                           ('classifier', svm)])

