import os
import numpy as np
import cv2
import gradio as gr
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Feature Extraction Functions
def extract_glcm_features(image, distances=[1], angles=[0]):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0]
    ]
    return np.array(features)


def extract_lbp_features(image, radius=1, points=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize
    return hist


# Load and Process Dataset
def load_dataset(dataset_path):
    labels = {'stone': 0, 'brick': 1, 'wood': 2}
    X_glcm, X_lbp, y = [], [], []
    for label in labels:
        folder_path = os.path.join(dataset_path, label)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            image = cv2.imread(img_path)
            X_glcm.append(extract_glcm_features(image))
            X_lbp.append(extract_lbp_features(image))
            y.append(labels[label])
    return np.array(X_glcm), np.array(X_lbp), np.array(y)


# SVM Classifier with Cross-validation and GridSearchCV for parameter optimization
def optimize_svm(X_train, y_train):
    param_dist = {
        'C': [0.1, 1, 10],  # Fewer values to search through
        'kernel': ['linear', 'rbf'],  # Limited kernels for faster search
        'gamma': ['scale', 0.1, 1],  # Smaller gamma range
        'degree': [3],  # Fix degree to 3 for poly kernel
    }
    svm = SVC()
    randomized_search = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, verbose=2,
                                           random_state=42)
    randomized_search.fit(X_train, y_train)

    print(f"Best SVM parameters: {randomized_search.best_params_}")
    return randomized_search.best_estimator_


# k-NN Classifier with Cross-validation and RandomizedSearchCV for parameter optimization
def optimize_knn(X_train, y_train):
    param_dist = {
        'n_neighbors': [3, 5, 7, 10, 15],  # Number of neighbors
        'weights': ['uniform', 'distance'],  # Weight function used in prediction
        'metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metric
    }
    knn = KNeighborsClassifier()
    randomized_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, verbose=2)
    randomized_search.fit(X_train, y_train)

    print(f"Best k-NN parameters: {randomized_search.best_params_}")
    return randomized_search.best_estimator_

# Train Classifiers
def train_classifiers(X, y):
    X1_train, X1_test, y1_train, y1_test = train_test_split(X[:50], y[:50], test_size=0.3, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X[50:100], y[50:100], test_size=0.3, random_state=42)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X[100:150], y[100:150], test_size=0.3, random_state=42)
    X_train = np.concatenate((X1_train, X2_train, X3_train))
    X_test = np.concatenate((X1_test, X2_test, X3_test))
    y_train = np.concatenate((y1_train, y2_train, y3_train))
    y_test = np.concatenate((y1_test, y2_test, y3_test))


    # SVM Classifier
    svm = optimize_svm(X_train, y_train)
    y_pred = svm.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred))

    # k-NN Classifier
    knn = optimize_knn(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("k-NN Accuracy:", accuracy_score(y_test, y_pred))

    return svm, knn


# Gradio Interface
def predict_texture(image, algorithm):
    if algorithm == "GLCM (SVM)":
        features = extract_glcm_features(image).reshape(1, -1)
        prediction = svm_model_glcm.predict(features)[0]
    elif algorithm == "LBP (SVM)":
        features = extract_lbp_features(image).reshape(1, -1)
        prediction = svm_model_lbp.predict(features)[0]
    elif algorithm == "GLCM (k-NN)":
        features = extract_glcm_features(image).reshape(1, -1)
        prediction = knn_model_glcm.predict(features)[0]
    else:
        features = extract_lbp_features(image).reshape(1, -1)
        prediction = knn_model_lbp.predict(features)[0]
    class_map = {0: 'Stone', 1: 'Brick', 2: 'Wood'}
    return f"Prediction: {class_map[prediction]}"


# Main Execution
# dataset_path = "data"
# X_glcm, X_lbp, y = load_dataset(dataset_path)
# svm_model_glcm, knn_model_glcm = train_classifiers(X_glcm, y)
# svm_model_lbp, knn_model_lbp = train_classifiers(X_lbp, y)
# joblib.dump(svm_model_glcm, "svm_model_glcm.pkl")
# joblib.dump(knn_model_glcm, "knn_model_glcm.pkl")
# joblib.dump(svm_model_lbp, "svm_model_lbp.pkl")
# joblib.dump(knn_model_lbp, "knn_model_lbp.pkl")

# Load models for inference
svm_model_glcm = joblib.load("svm_model_glcm.pkl")
knn_model_glcm = joblib.load("knn_model_glcm.pkl")
svm_model_lbp = joblib.load("svm_model_lbp.pkl")
knn_model_lbp = joblib.load("knn_model_lbp.pkl")

gr.Interface(
    fn=predict_texture,
    inputs=[gr.Image(type="numpy"), gr.Radio(["GLCM (SVM)", "LBP (SVM)", "GLCM (k-NN)", "LBP (k-NN)"])],
    outputs="text"
).launch()
