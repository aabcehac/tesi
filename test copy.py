from test_sqlite import *


from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

num_items = len(ultimate_data_dict['id'])
ultimate_data_list = [*ultimate_data_dict.values()]
X = ultimate_data_dict
y = X.pop('decile_score')

X = list(map(list, zip(*X.values()))) # Transpose the list
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=40)

scalers={
    'standard_scaler': StandardScaler(),
    'minmax_scaler': MinMaxScaler(),
    'quantile_scaler': QuantileTransformer()
}

models = {
    'logistic_regression': LogisticRegression(max_iter=1000),
    'svr': SVR(),
    'knn': KNeighborsRegressor(),
    'random_forest': RandomForestRegressor(),
    'voting': VotingRegressor(estimators=[
        ('knn', KNeighborsRegressor()),
        ('random_forest', RandomForestRegressor()), 
        ])
}

pipelines = {(scaler, estimator): make_pipeline(scalers[scaler], models[estimator]) for estimator in models for scaler in scalers}
scorers = {'mse score:': mean_squared_error, 'r2 score:': r2_score}
for scorer_name, scorer in scorers.items():
    print("\n", scorer_name)
    for key, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        pipeline.score(X_test, y_test)
        print(f'{scorer(y_test, prediction)} in {key}')

pca = PCA(n_components=2)
pca.fit(X)
print("\n", pca.explained_variance_ratio_)
# print("\n", pca.components_)
#tentativo identificazione contributo di ogni variabile
feature_names=list(ultimate_data_dict.keys())[:-1]
for i, component in enumerate(pca.components_):
    print(f"\nComponent {i+1}:")
    for feature, loading in zip(feature_names,component):
        print(f"{feature}: {loading:.4f}")
"""
"""
'''
#foresta amazing - approccio mongospastico stile primo anno

best_score=float('-inf')
best_pipeline=None
best_tree=None
best_scaler=None
'''

'''
for scaler in scalers:          #e quale scaler? tutti, uno ciascuno.
    #pipelines[(scaler, 'random_forest')]
    pipeline = make_pipeline(scalers[scaler])
'''
