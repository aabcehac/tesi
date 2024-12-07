from test_sqlite import ultimate_data_dict
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.tree import plot_tree
import copy

get_rfr = lambda pipeline: pipeline.named_steps['randomforestregressor']


def PCA_shower(n_components, preprocessor, X_train):
    print(f"{preprocessor.__name__}".capitalize())
    pca = PCA(n_components=n_components)
    pca.fit(preprocessor(X_train))
    t = PrettyTable(["Feature"] + [f"Component {i+1}" for i in range(len(pca.components_))])
    t.add_row(["Explained variance ratio"] + [f"{pca.explained_variance_ratio_[i]:.4f}" for i in range(len(pca.explained_variance_ratio_))], divider=True)
    t.add_rows([[feature] + [f"{pca.components_[j][i]:.4f}" for j in range(len(pca.components_))] for i, feature in enumerate(ultimate_data_dict.keys())])
    return t


if __name__ == "__main__":
    Z = copy.deepcopy(ultimate_data_dict)
    Z.pop('id')
    y = Z.pop('decile_score')
    X = list(map(list, zip(*Z.values()))) # Transpose the list
    X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=40)

    scalers={'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'quantile': QuantileTransformer()}
    models = {
        'logistic': LogisticRegression(max_iter=1000),
        'svr': SVR(),
        'knn': KNeighborsRegressor(),
        'random_forest': RandomForestRegressor(),
        'voting': VotingRegressor(estimators=[
            ('knn', KNeighborsRegressor()),
            ('random_forest', RandomForestRegressor()), 
            ])
    }

    pipelines = {(scaler, estimator): make_pipeline(scalers[scaler], models[estimator]).fit(X_train, y_train) for estimator in models for scaler in scalers}
    scorers = {'mse': make_scorer(mean_squared_error), 'r2': make_scorer(r2_score)}
    t = PrettyTable(["Model", "Pre-scaler", "mse_score", "r2_score"])
    t.add_rows([[key[1], key[0], f"{scorers['mse'](pipeline, X_test, y_test):.4f}", f"{scorers['r2'](pipeline, X_test, y_test):.4f}"] for key, pipeline in pipelines.items()])
    print(t)

    print(PCA_shower(5, quantile_transform, X_train))
    print(PCA_shower(5, minmax_scale, X_train))
    print(PCA_shower(5, scale, X_train))

    random_forest=get_rfr(pipelines[('standard', 'random_forest')])
    ########print(get_rfr(pipelines[('standard', 'random_forest')]))
    # Qui esce la foresta della pipeline allenata alla linea 30, se la pipeline ne ha uno.

    tree_scores=[(i, mean_squared_error(y_test, tree.predict(X_test))) for i, tree in enumerate(random_forest.estimators_)]
    tree_scores_sorted=sorted(tree_scores, key=lambda x: x[1])
    best_tree_index, best_tree_mse=tree_scores_sorted[0]
    best_tree=random_forest.estimators_[best_tree_index]
    print(f"best tree index: {best_tree_index}")
    print(f"MSE best tree: {best_tree_mse:.4f}")
    plt.figure(figsize=(10,5)) 
    plot_tree(best_tree, max_depth=2, feature_names=list(Z.keys()), filled=True)
    plt.show()