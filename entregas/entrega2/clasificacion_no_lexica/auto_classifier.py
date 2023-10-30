import pandas
import sklearn.preprocessing
import sklearn.metrics
import sklearn.linear_model
import sklearn.svm
import sklearn.neighbors
import sklearn.tree
import sklearn.ensemble
import sklearn.model_selection
import xgboost.sklearn
import catboost
import lightgbm
import sklearn.naive_bayes

model_abc = sklearn.ensemble.AdaBoostClassifier()
model_cbc = catboost.CatBoostClassifier(silent=True)
model_etc = sklearn.ensemble.ExtraTreesClassifier()
model_gbc = sklearn.ensemble.GradientBoostingClassifier()
model_knc = sklearn.neighbors.KNeighborsClassifier()
model_lgbm = lightgbm.LGBMClassifier()
model_logreg = sklearn.linear_model.LogisticRegression()
model_gnb = sklearn.naive_bayes.GaussianNB()
model_rfc = sklearn.ensemble.RandomForestClassifier()
model_svc = sklearn.svm.SVC()
model_tree = sklearn.tree.DecisionTreeRegressor()
model_xgbc = xgboost.sklearn.XGBClassifier()

CLASSIFIERS = [model_abc, model_cbc, model_etc, model_gbc, model_knc, model_lgbm, model_logreg, model_gnb, model_rfc, model_svc, model_tree, model_xgbc]


def auto_classifier(X_train, y_train, X_test, y_test, models=CLASSIFIERS):
    df_results = pandas.DataFrame(columns=["Model", "Accuracy Train", "Recall Train", "F1 Train", "Accuracy Test", "Recall Test", "F1 Test"])
    for index, model in enumerate(models):
        try:
            model.fit(X_train, y_train)

            train_predictions = model.predict(X_train)
            accuracy_train = sklearn.metrics.accuracy_score(y_true=y_train, y_pred=train_predictions)
            recall_train = sklearn.metrics.recall_score(y_true=y_train, y_pred=train_predictions, average="weighted")
            f1_score_train = sklearn.metrics.f1_score(y_true=y_train, y_pred=train_predictions, average="weighted")

            test_predictions = model.predict(X_test)
            accuracy_test = sklearn.metrics.accuracy_score(y_true=y_test, y_pred=test_predictions)
            recall_test = sklearn.metrics.recall_score(y_true=y_test, y_pred=test_predictions, average="weighted")
            f1_score_test = sklearn.metrics.f1_score(y_true=y_test, y_pred=test_predictions, average="weighted")

            # print(str(model))
            # print(accuracy_train, f1_score_train)
            # print(accuracy_test, f1_score_test)
            # print("\n")

            df_results.loc[index, "Model"] = str(model)
            df_results.loc[index, "Accuracy Train"] = accuracy_train
            df_results.loc[index, "Recall Train"] = recall_train
            df_results.loc[index, "F1 Train"] = f1_score_train
            df_results.loc[index, "Accuracy Test"] = accuracy_test
            df_results.loc[index, "Recall Test"] = recall_test
            df_results.loc[index, "F1 Test"] = f1_score_test
        except:
            print(f"NO: {model}")

    return df_results
