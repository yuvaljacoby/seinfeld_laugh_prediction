from sklearn.model_selection import train_test_split


def split_train_test(X, y, test_ratio=0.2):
    '''
    Split data to train and test based on episode --> episodes will be fully in train / test
    :param X: Features
    :param y: labels
    :param test_ratio: float [0,1] ratio of samples to keep in test
    :return: X_train, X_test, y_train, y_test
    '''

    return train_test_split(X, y, test_ratio)
