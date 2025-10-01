
# Face Classification with Support Vector Machines on the LFW Dataset

## Overview
This project explores the application of Support Vector Machines (SVMs) to face classification using the LFW (Labeled Faces in the Wild) Deep Funneled dataset.  
The goal is to evaluate how different SVM kernels (linear, RBF, polynomial) perform on a high-dimensional facial recognition task after dimensionality reduction.

---

## Dataset
The dataset used is the **LFW Deep Funneled** collection, which contains aligned and standardized facial images of public figures.  
It is accessed via the Activeloop DeepLake hub:

```python
lfw_dataset = deeplake.load('hub://activeloop/lfw-deep-funneled')
````

Images are flattened into feature vectors, and labels represent different individuals.

---

## Methodology

1. **Data Preparation**

   * Flattened each image into a feature vector.
   * Standardized all features with z-score normalization.
   * Applied Principal Component Analysis (PCA) to reduce dimensionality to 150 components for computational efficiency.
   * Split data 50/50 into training and test sets.

2. **Model Training**

   * Trained SVM classifiers with three kernels:

     * Linear
     * Radial Basis Function (RBF)
     * Polynomial

   Each model was trained using `C=10`, `gamma='scale'`, and class balancing enabled.

3. **Evaluation Metrics**

   * Overall accuracy on the test set.
   * **Classification Report**: Precision, Recall, F1-score per class.
   * **Confusion Matrix Heatmaps** to visualize prediction performance.

---

## Results

The following results summarize kernel performance:

* **Linear Kernel**

  * Provides a baseline with reasonable accuracy.
  * Efficient in training time.

* **RBF Kernel**

  * Achieves the best overall accuracy.
  * Captures nonlinear relationships in the feature space.

* **Polynomial Kernel**

  * Shows more variability in accuracy.
  * Computationally slower than linear and RBF kernels.

Heatmaps of confusion matrices are generated for each kernel, showing how well the classifier distinguishes between different individuals.

---

## Conclusion

This experiment demonstrates that SVMs can be effectively applied to face classification tasks when combined with preprocessing techniques like PCA.
Among the tested kernels, the **RBF kernel consistently outperformed the others**, suggesting that nonlinear decision boundaries are better suited for the complexity of facial recognition data.

Future extensions could involve:

* Hyperparameter optimization (grid search or randomized search).
* Comparing SVMs with other classifiers such as Random Forests, Logistic Regression, or Deep Neural Networks.
* Expanding the evaluation to include larger subsets of the dataset.

