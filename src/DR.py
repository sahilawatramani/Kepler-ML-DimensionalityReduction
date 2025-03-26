#%% Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_curve, auc, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

#%% Load and preprocess the dataset
df = pd.read_csv("C:\\Users\\Victus\\OneDrive\\Desktop\\USL\\DR_Project\\data\\cumulative.csv")

df.dropna(axis=1, how="all", inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

if 'koi_disposition' in df.columns:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['koi_disposition'])  # Target variable
else:
    raise ValueError("Dataset must contain 'koi_disposition' column for supervised learning.")

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numerical_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#%% Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

#%% Define function to train and evaluate models
def train_and_evaluate(X_train_reduced, X_test_reduced, title, model_type="RandomForest"):
    if model_type == "RandomForest":
        clf = RandomForestClassifier()
    else:
        clf = KNeighborsClassifier(n_neighbors=5)
    
    clf.fit(X_train_reduced, y_train)
    y_pred = clf.predict(X_test_reduced)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"\n{title} Model Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion Matrix
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{title} Confusion Matrix")
    plt.show()
    
    # ROC Curve (if applicable)
    if len(np.unique(y_test)) == 2:
        y_prob = clf.predict_proba(X_test_reduced)
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{title} (AUC = {roc_auc:.2f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

#%% Apply PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Explained Variance Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_), marker="o", linestyle="--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid()
plt.show()

train_and_evaluate(X_train_pca, X_test_pca, "PCA")

#%% Apply LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
train_and_evaluate(X_train_lda, X_test_lda, "LDA")

#%% Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_train_tsne = tsne.fit_transform(X_train)
X_test_tsne = tsne.fit_transform(X_test) 
train_and_evaluate(X_train_tsne, X_test_tsne, "t-SNE", model_type="KNN")

#%% Apply MDS
mds = MDS(n_components=2, random_state=42)
X_train_mds = mds.fit_transform(X_train)
X_test_mds = mds.fit_transform(X_test)  
train_and_evaluate(X_train_mds, X_test_mds, "MDS", model_type="KNN")

#%% Show combined ROC curves
plt.title("ROC Curves for All Models")
plt.show()
print("\nModel Training and Evaluation Completed.")
