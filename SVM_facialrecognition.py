import deeplake
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

lfw_dataset = deeplake.load('hub://activeloop/lfw-deep-funneled')

data_loader = lfw_dataset.pytorch(num_workers=0, batch_size=64, shuffle=False)

images, labels = [], []
for img, label, _ in data_loader:
    images.append(img.view(img.size(0), -1).type(torch.FloatTensor))
    labels.append(label.type(torch.LongTensor))

images = torch.cat(images)
labels = torch.cat(labels)

X = images.view(images.size(0), -1).numpy()
y = labels.numpy()

# only keep faces with 6+ samples
(unique_labels, counts) = np.unique(y, return_counts=True)
valid_labels = unique_labels[counts >= 10]

mask = np.isin(y, valid_labels).ravel()   
X = X[mask]
y = y[mask]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

pca = PCA(n_components=150, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Kernels to test
kernels = ["linear", "rbf", "poly"]
accuracies = {}
f1_scores = {}

for kernel in kernels:
    print(f"\nTraining SVM with {kernel} kernel...")
    clf = svm.SVC(kernel=kernel, class_weight='balanced', C=10, gamma=0.01)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracies[kernel] = acc
    f1_scores[kernel] = f1
    print(f"Accuracy ({kernel}): {acc*100:.2f}%")
    print(f"Macro F1 Score ({kernel}): {f1:.4f}")

    if kernel == "rbf":
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap='Blues')
        plt.title(f"Confusion Matrix ({kernel} kernel)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

# Plot Accuracy and F1 Score Comparison
plt.figure(figsize=(10, 6))
bar_width = 0.35
kernels_list = list(accuracies.keys())
indices = np.arange(len(kernels_list))

plt.bar(indices, [accuracies[k]*100 for k in kernels_list], bar_width, label='Accuracy (%)', color='skyblue')
plt.bar(indices + bar_width, [f1_scores[k]*100 for k in kernels_list], bar_width, label='Macro F1 Score (%)', color='orange')

plt.ylabel("Score (%)")
plt.title("SVM Kernel Comparison on LFW Dataset")
plt.xticks(indices + bar_width / 2, kernels_list)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.show()