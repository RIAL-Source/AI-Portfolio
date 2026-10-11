# Heart Disease Classification Project

This project uses Machine Learning algorithms to predict the presence of heart disease based on various clinical parameters.

## Dataset

The dataset used is the [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) available on Kaggle. It contains 918 samples and 12 features, including:

- Age
- Sex 
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting electrocardiogram results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression induced by exercise 
- Slope of the peak exercise ST segment
- Number of major vessels colored by fluoroscopy
- Thalassemia
- Diagnosis of heart disease (target)

## Models Used

The project compares the performance of three classification models:

1. **Decision Tree**: decision tree with max depth 4 and min samples split of 50.

2. **Random Forest**: ensemble of 100 trees with max depth 4 and min samples split of 50. 

3. **XGBoost**: extreme gradient boosting classifier with 500 estimators and learning rate 0.1.

## Model Evaluation

The models are evaluated on the Accuracy metric. The dataset is split into training set (80%) and validation set (20%).

For XGBoost, the training set is further split into train (80%) and eval (20%) for monitoring performance during training. 

## Results

- Decision Tree:
  - Train Accuracy:
  - Test Accuracy: 
- Random Forest:
  - Train Accuracy:
  - Test Accuracy:
- XGBoost: 
  - Train Accuracy: 0.9973
  - Test Accuracy: 0.8852

## Next Steps

- Hyperparameter tuning with Grid Search
- Try other algorithms (SVM, Neural Networks) 
- Feature selection to identify most important predictors
- Analyze misclassified samples to improve the model