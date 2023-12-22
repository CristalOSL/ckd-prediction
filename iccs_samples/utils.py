import pickle
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score

LEARNING_DATA_PATH = "data/learning_data.pkl"


def load_sample_data():
    """ Load sample train and test matrices. Sample data"""

    print("Loading data...")
    with (open(LEARNING_DATA_PATH, "rb") as file):
        (data_train, data_test, labels_train, labels_test) = pickle.load(file)
    print("All data loaded!")
    return data_train, data_test, labels_train, labels_test


def binarize_labels(y):
    """ Turns CKD stages into binary labels. In the context of our paper, we consider stages 4 and 5 to be the
    positive cases."""

    class_map = {"stage1": 0, "stage2": 0, "stage3": 0, "stage4": 1, "stage5": 1}
    return pd.DataFrame(y)[0].apply(lambda stage: class_map[stage])


def impute_missing_values(x):
    """ If x is a 3D matrix (patient history, records, analyses), returns a copy of x with mean-imputed values. """

    print(f"Imputing values for {len(x)} matrices...")
    x_ = x.copy()
    for i, matrix in tqdm(enumerate(x)):
        imputer = SimpleImputer(keep_empty_features=True, strategy="mean")
        matrix_imputed = imputer.fit_transform(matrix)
        x_[i] = matrix_imputed
    return x_


def scale_data(x_train, x_test):
    """ Standardize data in x_train and x_test. """

    scaler = StandardScaler()

    x_train_shape = x_train.shape
    x_test_shape = x_test.shape
    num_columns = x_train_shape[1] # x_train_shape is (num_samples, num_columns, num_rows)

    print(x_train_shape, x_test_shape)
    print("Reshaping x_train and x_test...")
    x_train_reshape = np.stack(x_train).reshape(-1, num_columns)
    x_test_reshape = np.stack(x_test).reshape(-1, num_columns)

    print(x_train_reshape.shape, x_test_reshape.shape)
    print("Fitting the imputer...")
    scaler.fit(x_train_reshape)

    print("Scaling values...")
    x_train_reshape = scaler.transform(x_train_reshape)
    x_test_reshape = scaler.transform(x_test_reshape)

    print("Reshaping x_train and x_test...")
    x_train_scaled = x_train_reshape.reshape(x_train_shape)
    x_test_scaled = x_test_reshape.reshape(x_test_shape)

    print("Done scaling!")
    return x_train_scaled, x_test_scaled


def resample(x, y):
    """ Use random oversampling to resample data."""

    ros = RandomOverSampler(random_state=42, sampling_strategy=0.5)
    data_2d = x.reshape((x.shape[0], -1))  # Reshape the 3D array to 2D for oversampling
    data_resampled, labels_resampled = ros.fit_resample(data_2d, y)
    data_resampled = data_resampled.reshape((-1, x.shape[1], x.shape[2]))  # 3D
    return data_resampled, labels_resampled


# Evaluation
def evaluate_model(model, x_test, y_test):
    threshold = 0.5

    _, test_accuracy = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    y_pred_binary = pd.DataFrame(y_pred).apply(lambda val: (val > threshold).astype(int))

    # Metrics
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
    print("ROC-AUC: {:.2f}%".format(roc_auc_score(y_test, y_pred_binary) * 100))
    print(classification_report(y_test, y_pred_binary))

    # Confusion matrix
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    sns.heatmap(confusion_matrix(y_test, y_pred_binary, normalize="true"),annot=True)
    plt.subplot(122)
    sns.heatmap(confusion_matrix(y_test, y_pred_binary, normalize=None),annot=True)
    plt.show()