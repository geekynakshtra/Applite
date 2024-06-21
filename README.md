# Applite
# Few-Shot Learning on CIFAR-100 

This project demonstrates a few-shot learning approach using a pre-trained ResNet50 model on the CIFAR-100 dataset. Few-shot learning aims to train a model with a limited number of samples per class. This project includes data preparation, model training, evaluation, and computing various performance metrics such as accuracy, precision, recall, and F1 score.

# Requirements
Python 3.7+
PyTorch
torchvision
numpy
scikit-learn

# Code Explanation
1. Data Preparation
Transforms: Applied various data augmentations for the training set and normalization for both training and test sets.
Few-Shot Dataset: Created a subset of the training dataset with a limited number of samples per class.
Train/Validation Split: Split the few-shot training dataset into training and validation subsets.
2. Model Selection
Pre-trained ResNet50: Used a pre-trained ResNet50 model from torchvision and modified the final fully connected layer to output 100 classes for CIFAR-100.
3. Training
Criterion: Used CrossEntropyLoss.
Optimizer: Used SGD with momentum and weight decay.
Scheduler: Used a learning rate scheduler to reduce the learning rate at certain epochs.
Training Loop: Included both training and validation within each epoch.
4. Evaluation
Evaluated the model on the test dataset and computed metrics such as accuracy, precision, recall, and F1 score using precision_recall_fscore_support from sklearn.



# Future Work
1.Hyperparameter Tuning: Experiment with different hyperparameters to improve model performance.
2.Data Augmentation: Explore additional data augmentation techniques to enhance the model's robustness.
3.Advanced Models: Test other pre-trained models and compare their performance with ResNet50.
4.Visualization: Implement visualizations for data augmentation effects and training progress.


