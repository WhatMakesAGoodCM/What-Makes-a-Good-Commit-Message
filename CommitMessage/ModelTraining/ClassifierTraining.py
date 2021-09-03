import datetime
import warnings
from collections import Counter

import numpy as np
import pymysql
import statsmodels.api as sm
import timeout_decorator
from imblearn.over_sampling import ADASYN
from joblib import dump
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, roc_curve, auc, \
    roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
# 评估
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

SCORING = {'accuracy': 'accuracy', 'precision': make_scorer(precision_score), 'recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score),
           'AUC': make_scorer(roc_auc_score)}

resConn = pymysql.connect(host='localhost', port=3306, user='root', passwd='******', db='trainset', charset='utf8')
cursor = resConn.cursor()


def LRPValue(train_features, train_labels):
    train_features = [x[-5:] for x in train_features]
    lr_clf = sm.Logit(train_labels, train_features)
    result = lr_clf.fit(maxiter=300)
    print(result.summary())
    logger.info(result.summary())


def LRClassifier(trainFeatures, trainLabels):
    warnings.filterwarnings("ignore")
    parameters = {'C': np.linspace(0.0001, 20, 20),
                  'random_state': np.arange(1, 5),
                  'solver': ["newton-cg", "lbfgs", "liblinear", "sag"],
                  'multi_class': ['ovr'],
                  'dual': [False],
                  'verbose': [False],
                  'max_iter': [500]
                  }
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(LogisticRegression(), parameters, scoring=SCORING, refit="accuracy", cv=fold, n_jobs=25)
    grid.fit(trainFeatures, trainLabels)
    bestParameter = get_score_by_grid(grid)
    print("LR Best: using %s " % (bestParameter))
    model = LogisticRegression(C=bestParameter['C'], random_state=bestParameter['random_state'],
                               solver=bestParameter['solver'], multi_class='ovr', dual=False, verbose=False,
                               max_iter=500)
    return model


@timeout_decorator.timeout(24 * 60 * 60, timeout_exception=StopIteration)
def MyMLPClassifier(trainFeatures, trainLabels):
    # 多层感知器分类模型
    model = MLPClassifier()
    parameters = {"hidden_layer_sizes": [(100,), (100, 50), (100, 100), (100, 80), (100, 30)],
                  "activation": ["logistic"],
                  "solver": ['adam'],
                  "verbose": [False],
                  'learning_rate': ['constant', 'adaptive']
                  }
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(model, parameters,
                        scoring=SCORING, refit="accuracy", cv=fold, n_jobs=25)
    grid.fit(trainFeatures, trainLabels)
    bestParameter = get_score_by_grid(grid)
    print("MLP Best using %s " % (bestParameter))
    model = MLPClassifier(hidden_layer_sizes=bestParameter['hidden_layer_sizes'],
                          activation=bestParameter['activation'],
                          solver=bestParameter['solver'],
                          verbose=False,
                          learning_rate=bestParameter['learning_rate'])
    return model


def KNNClassifier(trainFeatures, trainLabels):
    model = KNeighborsClassifier()
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    parameter = {'n_neighbors': np.arange(1, 10, 1),
                 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                 }
    grid = GridSearchCV(estimator=model, param_grid=parameter, cv=fold,
                        scoring=SCORING, refit="accuracy", n_jobs=25)
    grid.fit(trainFeatures, trainLabels)
    bestParameter = get_score_by_grid(grid)
    print("KNN Best using %s " % (bestParameter))
    model = KNeighborsClassifier(n_neighbors=bestParameter['n_neighbors'], algorithm=bestParameter['algorithm'])
    return model


@timeout_decorator.timeout(3 * 60 * 60, timeout_exception=StopIteration)
def GradientBosstingClassifier(trainFeatures, trainLabels):
    model = GradientBoostingClassifier()
    parameter = {"loss": ["deviance"],
                 "learning_rate": [0.1, 0.15, 0.2],
                 "min_samples_split": np.linspace(0.1, 0.4, 6),
                 "min_samples_leaf": np.linspace(0.1, 0.3, 6),
                 "max_depth": [3, 5, 8],
                 "max_features": ["log2", "sqrt"],
                 "criterion": ["friedman_mse", "mse"],
                 "subsample": [0.8, 0.85, 0.9, 0.95, 1.0],
                 "n_estimators": [10]
                 }
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(estimator=model, param_grid=parameter, cv=fold,
                        scoring=SCORING, refit="accuracy", n_jobs=25)
    grid.fit(trainFeatures, trainLabels)
    bestParameter = get_score_by_grid(grid)
    print("GBossting Best using %s " % (bestParameter))
    model = GradientBoostingClassifier(loss="deviance",
                                       learning_rate=bestParameter['learning_rate'],
                                       min_samples_split=bestParameter['min_samples_split'],
                                       min_samples_leaf=bestParameter['min_samples_leaf'],
                                       max_depth=bestParameter['max_depth'],
                                       max_features=bestParameter['max_features'],
                                       criterion=bestParameter['criterion'],
                                       subsample=bestParameter['subsample'],
                                       n_estimators=10
                                       )
    return model


@timeout_decorator.timeout(3 * 60 * 60, timeout_exception=StopIteration)
def MyAdaBoostClassifier(trainFeatures, trainLabels):
    dtc = MLPClassifier()
    model = AdaBoostClassifier(base_estimator=dtc)
    parameter = {"base_estimator__hidden_layer_sizes": [(100,), (100, 50), (100, 100), (100, 80), (100, 30)],
                 "base_estimator__activation": ["logistic"],
                 "base_estimator__solver": ['adam'],
                 "base_estimator__verbose": [False],
                 'base_estimator__learning_rate': ['constant', 'adaptive'],
                 "n_estimators": [100, 10, 50, 60, 70]}

    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    grid = GridSearchCV(estimator=model, param_grid=parameter, cv=fold,
                        scoring=SCORING, refit="accuracy", n_jobs=25)
    grid.fit(trainFeatures, trainLabels)
    bestParameter = get_score_by_grid(grid)
    print("GBossting Best using %s " % (bestParameter))
    model = GradientBoostingClassifier(
        base_estimator__hidden_layer_sizes=bestParameter["base_estimator__hidden_layer_sizes"],
        base_estimator__activation=bestParameter['base_estimator__activation'],
        base_estimator__learning_rate=bestParameter['base_estimator__learning_rate'],
        n_estimators=bestParameter['n_estimators'],
        base_estimator__solver=['adam'],
        base_estimator__verbose=[False],
    )
    return model


def RFClassifier(trainFeatures, trainLabels):
    model = RandomForestClassifier()
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    tree_param_grid = {'max_features': [28, 150, 768, 772],
                       'min_samples_split': [i for i in np.arange(7, 16, 1)],
                       'n_estimators': list(range(50, 100, 20))
                       }
    grid = GridSearchCV(estimator=model, param_grid=tree_param_grid,
                        scoring=SCORING, refit="accuracy", n_jobs=25, cv=fold)
    grid.fit(trainFeatures, trainLabels)
    bestParameter = get_score_by_grid(grid)
    print("RForest Best using %s " % (bestParameter))
    model = RandomForestClassifier(max_features=bestParameter['max_features'],
                                   min_samples_split=bestParameter['min_samples_split'],
                                   n_estimators=bestParameter['n_estimators'])
    return model


@timeout_decorator.timeout(3 * 60 * 60, timeout_exception=StopIteration)
def DTreeClassifier(trainFeatures, trainLabels):
    model = DecisionTreeClassifier()
    fold = KFold(n_splits=10, random_state=5, shuffle=True)
    tree_param_grid = {'criterion': ['gini', 'entropy'],
                       'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20],
                       'min_samples_leaf': list(range(5, 15, 1))
                       }
    grid = GridSearchCV(estimator=model, param_grid=tree_param_grid,
                        scoring=SCORING, refit="accuracy", n_jobs=25, cv=fold)
    grid.fit(trainFeatures, trainLabels)
    bestParameter = get_score_by_grid(grid)
    print("DTree Best using %s " % (bestParameter))
    model = DecisionTreeClassifier(criterion=bestParameter['criterion'], max_depth=bestParameter['max_depth'],
                                   min_samples_leaf=bestParameter['min_samples_leaf'])
    return model


def get_score_by_grid(grid: GridSearchCV):
    print("GridSearchCV is complate!")
    accuMean = grid.cv_results_['mean_test_accuracy']
    accuStd = grid.cv_results_['std_test_accuracy']
    accuRank = grid.cv_results_['rank_test_accuracy']
    preMean = grid.cv_results_['mean_test_precision']
    preStd = grid.cv_results_['std_test_precision']
    preRank = grid.cv_results_['rank_test_precision']
    recallMean = grid.cv_results_['mean_test_recall']
    recallStd = grid.cv_results_['std_test_recall']
    rocMean = grid.cv_results_['mean_test_AUC']
    rocStd = grid.cv_results_['std_test_AUC']
    f1Mean = grid.cv_results_['mean_test_f1']
    f1Std = grid.cv_results_['std_test_f1']
    bestParam = grid.cv_results_['params']
    bestIndex = grid.best_index_
    i = bestIndex
    rank = 1
    while preMean[i] < 0.5:
        rank += 1
        indx = 0
        if rank > 20:
            break
        for num in accuRank:
            if num == rank:
                i = indx
                break
            indx += 1
    bestIndex = i

    res = "refit by:" + str(grid.refit) + " Parameters: " + str(bestParam[bestIndex])
    logger.info(res)
    print(res)
    return bestParam[bestIndex]


def modelScore(testFeatures, testLabels, trainedModel):
    # logger.info(trainedModel.score(testFeatures, testLabels))
    yvalid = testLabels
    y_pred = trainedModel.predict(testFeatures)
    y_score = trainedModel.predict_proba(testFeatures)[:, 1]

    weightedPrecision, weightedRecall, weightedF1, accuracy_mean, tn, fp, fn, tp, f1Score, recallScore, precisionScore, fpr, tpr, aucArea, aucScore = binClassify(
        yvalid, y_pred, y_score)

    return accuracy_mean, precisionScore, recallScore, f1Score, aucScore, tn, fp, fn, tp, weightedPrecision, weightedRecall, weightedF1


def binClassify(yValid, yPred, yScore):
    accuracy_mean = accuracy_score(yValid, yPred)
    tn, fp, fn, tp = confusion_matrix(yValid, yPred).ravel()
    f1Score = f1_score(yValid, yPred, average=None)
    recallScore = recall_score(yValid, yPred, average=None)
    precisionScore = precision_score(yValid, yPred, average=None)
    weightedPrecision = precision_score(yValid, yPred, average="weighted")
    weightedRecall = recall_score(yValid, yPred, average="weighted")
    weightedF1 = f1_score(yValid, yPred, average="weighted")

    fpr, tpr, thresholds = roc_curve(yValid, yScore, pos_label=0)
    aucArea = auc(fpr, tpr)
    aucScore = roc_auc_score(y_true=yValid, y_score=yScore)

    return weightedPrecision, weightedRecall, weightedF1, accuracy_mean, tn, fp, fn, tp, f1Score, recallScore, precisionScore, fpr, tpr, aucArea, aucScore


def multiClassify(yValid, yPred):
    accuracy_mean = accuracy_score(yValid, yPred)
    accuracy_matric = confusion_matrix(yValid, yPred)
    f1Score = f1_score(yValid, yPred, average=None)
    recallScore = recall_score(yValid, yPred, average=None)
    precisionScore = precision_score(yValid, yPred, average=None)
    return accuracy_mean, accuracy_matric, f1Score, recallScore, precisionScore


def updatePreLabel(testId, testRepoId, predLabel, modelName, whatOrWhy):
    modeName = modelName.lower()
    field = whatOrWhy + "_prediction_" + modeName
    for i in range(0, len(testId)):
        sql = "update message_train_result set " + field + " = " + str(predLabel[i]) + " where id = " + str(testId[i]) \
              + " and repo_id = " + str(testRepoId[i]) + " and planA_B = 1"
        try:
            cursor.execute(sql)
            resConn.commit()
        except:
            print(sql)
            resConn.rollback()


def refitModel(model, info, modelNameSX, features, test_features, labels, test_labels, idList, repoIdList):
    today = datetime.datetime.now()
    modelName = str(today.month) + "_" + str(today.day) + "_" + str(today.hour) + "_" + info + "_" + str(modelNameSX)
    features = np.array(features)
    labels = np.array(labels)
    logger.info("label test: " + str(sorted(Counter(test_labels).items())))

    model.fit(features, labels)

    dump(model, "./model/" + modelName + ".joblib")
    accuracyScore, precisionScore, recallScore, f1Score, aucScore, tn, fp, fn, tp, weightedPrecision, weightedRecall, weightedF1 = modelScore(
        test_features, test_labels, model)

    print('TN:%d, FP:%d, FN:%d, TP:%d' % (tn, fp, fn, tp))

    if precisionDict.get(modelNameSX) is None:
        precisionDict[modelNameSX] = [precisionScore[1], precisionScore[0], weightedPrecision, weightedRecall,
                                      weightedF1, accuracyScore, 1, recallScore[1], recallScore[0], f1Score[0],
                                      f1Score[1]]
    else:
        tempPre = precisionDict[modelNameSX]
        tempPre[0] += precisionScore[1]
        tempPre[1] += precisionScore[0]
        tempPre[2] += weightedPrecision
        tempPre[3] += weightedRecall
        tempPre[4] += weightedF1
        tempPre[5] += accuracyScore
        tempPre[6] += 1
        tempPre[7] += recallScore[1]
        tempPre[8] += recallScore[0]
        tempPre[9] += f1Score[1]
        tempPre[10] += f1Score[0]
        precisionDict[modelNameSX] = tempPre
    tempPre = precisionDict[modelNameSX]
    res = "\n flod " + str(tempPre[6]) + "\tAccuaryMean: " + str(tempPre[5] / tempPre[6]) \
          + "\tWeightedPrecisionMean: " + str(tempPre[2] / tempPre[6]) + "\tWeightedRecallMean: " \
          + str(tempPre[3] / tempPre[6]) + "\t WeightedF1Mean： " + str(tempPre[4] / tempPre[6]) \
          + "\tPrecisonMean: " + str(tempPre[0] / tempPre[6]) + "\t NegativePrecisionMean： " + str(
        tempPre[1] / tempPre[6]) \
          + "\tRecallMean: " + str(tempPre[7] / tempPre[6]) + "\t NegativeRecallMean： " + str(tempPre[8] / tempPre[6]) \
          + "\tF1Mean: " + str(tempPre[9] / tempPre[6]) + "\t NegativeF1Mean： " + str(tempPre[10] / tempPre[6])
    print(res)
    logger.info(res)


def trainClassifierNegative(features, labels, idList, repoIdList, info, mylogger):
    warnings.filterwarnings("ignore")
    global logger
    logger = mylogger

    global precisionDict
    precisionDict = dict()
    fold = KFold(n_splits=10, random_state=25, shuffle=True)
    labels = np.array(labels)
    for train_index, test_index in fold.split(features, labels):
        train_features, test_features, train_labels, test_labels = \
            features[train_index], features[test_index], labels[train_index], labels[test_index]

        logger.info("label: %s" % str(sorted(Counter(labels).items())))
        logger.info("train_label: %s" % str(sorted(Counter(train_labels).items())))

        x_resample, y_resample = ADASYN(random_state=666, n_neighbors=8, n_jobs=20).fit_resample(train_features,
                                                                                                 train_labels)
        train_features, train_labels = x_resample, y_resample
        logger.info("resample train_label: %s" % str(sorted(Counter(train_labels).items())))

        estimatorSelect(idList, info, logger, repoIdList, test_features, test_labels, train_features, train_labels)


def estimatorSelect(idList, info, logger, repoIdList, test_features, test_labels, train_features, train_labels):
    # logger.info("================================AdaBoostClassifier================================")
    # try:
    #     AdaBoostModel = MyAdaBoostClassifier(train_features, train_labels)
    #     refitModel(AdaBoostModel, info, 'DTree', train_features, test_features, train_labels, test_labels, idList,
    #                repoIdList)
    # except:
    #     print("train AdaBoostClassifier time out")
    #     logger.info("train AdaBoostClassifier time out")

    logger.info("================================KNNClassifier================================")
    KNNModel = KNNClassifier(train_features, train_labels)
    refitModel(KNNModel, info, 'KNN', train_features, test_features, train_labels, test_labels, idList, repoIdList)

    logger.info("================================LRClassifier==================================")
    try:
        LRModel = LRClassifier(train_features, train_labels)
        refitModel(LRModel, info, "LR", train_features, test_features, train_labels, test_labels, idList, repoIdList)
    except:
        print("train LRClassifier time out")

    logger.info("================================MLPClassifierClassifier================================")
    try:
        MLPModel = MyMLPClassifier(train_features, train_labels)
        refitModel(MLPModel, info, "MLP", train_features, test_features, train_labels, test_labels, idList, repoIdList)
    except:
        print("train MLPClassifier time out")

    logger.info("================================RForestClassifier================================")
    RForestModel = RFClassifier(train_features, train_labels)
    refitModel(RForestModel, info, 'RF', train_features, test_features, train_labels, test_labels, idList, repoIdList)

    logger.info("================================DTreeClassifier================================")
    try:
        DTreeModel = DTreeClassifier(train_features, train_labels)
        refitModel(DTreeModel, info, 'DTree', train_features, test_features, train_labels, test_labels, idList,
                   repoIdList)
    except:
        print("train DTreeClassifier time out")
        logger.info("train DTreeClassifier time out")

    logger.info("================================GradientBosstingClassifier================================")
    try:
        GBosstingModel = GradientBosstingClassifier(train_features, train_labels)
        refitModel(GBosstingModel, info, "GBossting", train_features, test_features, train_labels, test_labels, idList,
                   repoIdList)
    except:
        print("train GradientBosstingClassifier time out")
        logger.info("train GradientBosstingClassifier time out")

    logger.info("================================DummyClassifier==============================")
    dummyModel = DummyClassifier(strategy="stratified")
    refitModel(dummyModel, info, "Dummy", train_features, test_features, train_labels, test_labels, idList, repoIdList)
