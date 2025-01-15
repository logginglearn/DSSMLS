# DSSMLS: Dynamic Semi-Supervised Meta-Learning with SHAP for Lithology Classification

## Overview

Lithology classification is a crucial task in geological surveys, particularly in the interpretation of well-logging data. However, annotating well-logging data is labor-intensive and costly, leading to a significant shortage of labeled samples. Standard supervised learning techniques often fail to perform well under such data-scarce conditions.

To tackle this issue, we propose a novel approach for **few-shot lithology classification** called **Dynamic Semi-Supervised Meta-Learning with SHAP (DSSMLS)**. This method addresses the challenges of limited labeled data and class imbalance by introducing the following key innovations:

1. **Dynamic Prototype Adjustment**: We employ a dynamic mechanism to adaptively modify the distances between data samples and their respective class prototypes using a trainable distance metric. This enhances classification performance, especially for complex and non-linear data distributions.
   
2. **Pseudo-Labeling Strategy**: A pseudo-labeling technique is used to refine the prototypes by leveraging unlabeled data, thus improving the model's generalization capability in conditions where labeled data is scarce.

3. **Feature Extraction with Attention Mechanism**: The model integrates an attention-based convolutional kernel extractor to capture both global and local features from the well-logging data, preserving crucial local patterns and ensuring robust feature representation.

Experimental results demonstrate that **DSSMLS** outperforms traditional lithology classification models, significantly improving accuracy and stability, even under limited data and imbalanced class distributions.

---

## Key Features

- **Dynamic Prototype Adjustment**: Adaptive distance metrics to refine prototypes for better classification.
- **Pseudo-Labeling**: Utilizes unlabeled data to further optimize prototype positions.
- **Attention Mechanism**: Combines convolutional kernel extraction and attention to capture both global and local features.
- **Meta-Learning**: Few-shot learning capabilities for effective lithology classification with limited labeled data.

---

## Requirements

- Python >= 3.7
- PyTorch >= 1.8
- NumPy >= 1.18
- SciPy >= 1.5
- Matplotlib (for visualization)

Install required dependencies:

 
