# Leukemia Detection using Deep Learning

**Description:**

This project aims to detect Leukemia using deep learning techniques. It explores various approaches including convolutional neural networks (CNNs) with data augmentation, transfer learning (using DenseNet and ResNet architectures), and fine-tuning pre-trained models.

**Dataset:**

The dataset used in this project consists of microscopic blood smear images categorized into six classes:

1. ALL (Acute Lymphoblastic Leukemia)
2. AML (Acute Myeloid Leukemia)
3. CLL (Chronic Lymphocytic Leukemia)
4. MM (Multiple Myeloma)
5. Healthy
6. CML (Chronic Myeloid Leukemia)

You can find the dataset [here (insert link to dataset if publicly available)].

**Data Splitting:**

The dataset is split into three subsets:

* Train: 60% of the data used for training the models.
* Validation: 20% of the data used for evaluating model performance during training.
* Test: 20% of the data used for final model evaluation.

**Models:**

The following models were implemented:

* **CNN with Data Augmentation:** A CNN model with data augmentation techniques like random flipping, rotation, and zooming to enhance model generalization.
* **Transfer Learning with DenseNet:** A pre-trained DenseNet121 model was used as a feature extractor, with a custom classification head added for Leukemia detection. Two variations were explored: one without fully connected layers and one with fully connected layers.
* **Transfer Learning with ResNet:** Similar to DenseNet, a pre-trained ResNet50 model was used with a custom classification head. 
* **Fine-tuning:** Fine-tuning was applied to both DenseNet and ResNet models by unfreezing some layers of the base models and training with a lower learning rate to further improve performance.

**Results:**

[Include a brief summary of the results achieved by each model, such as accuracy, precision, recall, F1-score, and potentially a confusion matrix.]

**Usage:**

1. Clone this repository: `git clone [repository URL]`
2. Download the dataset and place it in the appropriate directory.
3. Install the required libraries:


4. Run the Jupyter Notebook `leukemia_detection.ipynb` to train and evaluate the models.

**Dependencies:**

* Python 3.x
* TensorFlow 2.x
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn


**Author:**

Rahul Drabit Chowdhury
