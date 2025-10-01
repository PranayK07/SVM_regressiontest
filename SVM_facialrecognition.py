import deeplake
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

lfw_dataset = deeplake.load('hub://activeloop/lfw-deep-funneled')

data_loader = lfw_dataset.pytorch(num_workers=0, batch_size=64, shuffle=False)

images, labels = [], []
for img, label, _ in data_loader:
    images.append(img.view(img.size(0), -1).type(torch.FloatTensor))
    labels.append(label.type(torch.LongTensor))

images = torch.cat(images)
labels = torch.cat(labels)

# 50/50 train/test split
X_train = images[:len(images)//2].numpy()
X_test = images[len(images)//2:].numpy()
y_train = labels[:len(labels)//2].numpy()
y_test = labels[len(labels)//2:].numpy()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# reducing dimensionality with PCA for speed/efficiency
pca = PCA(n_components=150)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# list of kernels that I am trying out
kernels = ['linear', 'rbf', 'poly']
results = {}

for kernel in kernels:
    print(f"\n🔹 Training SVM with {kernel} kernel...")
    model = svm.SVC(kernel=kernel, C=10, gamma='scale', class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[kernel] = {
        "accuracy": model.score(X_test, y_test),
        "report": classification_report(y_test, y_pred, zero_division=0, output_dict=False),
        "y_pred": y_pred
    }
    
    # classification report
    print(f"Kernel: {kernel}")
    print("Accuracy:", results[kernel]["accuracy"])
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    # confusion matrix creation
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, cmap="Blues", annot=False, cbar=True)
    plt.title(f"Confusion Matrix - {kernel} kernel")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

print("\n Kernel Comparison Results:")
for k, v in results.items():
    print(f"{k} kernel -> Accuracy: {v['accuracy']:.4f}")
