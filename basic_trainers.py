from seinfeld_playground import *
from train_utils import split_train_test
from compare_models import compare_models_roc_curve, plot_confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def Model_OneHotEncoding(df_train, df_test):
    y_train = df_train.is_funny

    # encode the train text using OneHotEncoding
    cv, X_train = getOneHotEncoding(df_train.txt)
    X_test = cv.transform(df_test.txt)

    # Train LogisticRegression
    lr = LogisticRegression(solver='lbfgs', n_jobs=-1, max_iter=300)
    lr.fit(X_train, y_train)
    y_hat_lr = lr.predict_proba(X_test)

    # Train SVM
    # svm = SVC(probability=1, gamma='auto')
    # svm.fit(X_train, y_train)
    # y_hat_svm = svm.predict_proba(X_test)

    # Train RandomForest
    # rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    # rf.fit(X_train, y_train)
    # y_hat_rf = rf.predict_proba(X_test)
    # return [y_hat_lr[:, 1], y_hat_svm[:, 1], y_hat_rf[:, 1]], ['lr', 'svm', 'rf']
    return y_hat_lr[:, 1]

def plot_scores(y_test, y_hats, labels):
    # labels = ['lr', 'svm', 'rf']
    # y_hats = [y_hat_lr[:, 1], y_hat_svm[:, 1], y_hat_rf[:, 1]]
    auc = compare_models_roc_curve(y_test, y_hats, labels)
    print(auc)
    plot_confusion_matrix(y_test, y_hats, labels)

if __name__ == "__main__":
    df = load_corpus()

    df = getSceneData(df)
    df_train, df_test = split_train_test(df, 0.2)
    y_test = df_test.is_funny
    y_hats, labels = Model_OneHotEncoding(df_train, df_test)
    plot_scores(y_test, y_hats, labels)