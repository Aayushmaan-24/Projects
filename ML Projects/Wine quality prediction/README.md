# Wine Quality Prediction Project Summary

## Project Goal
The goal of this project is to predict the quality of red wine based on its chemical properties.

## Data Source
The dataset used is the Red Wine Quality dataset, which contains information about the physicochemical properties of red wine and its quality score.

## Data Exploration and Visualization
- The dataset contains 1599 samples and 12 features (11 input features and 1 target variable 'quality').
- There are no missing values in the dataset.
- The data types are primarily float64, with the 'quality' column as int64.
- The 'quality' variable is an integer ranging from 3 to 8. The distribution shows that most wines are rated between 5 and 6.
- Wines were classified into three categories based on quality: poor (quality < 5), average (quality 5-6), and finest (quality >= 7). The counts for each category are:
    - Total samples: 1599
    - Finest quality wines: 217
    - Average quality wines: 1319
    - Poor quality wines: 63
- Visualizations (scatter matrix, heatmap, joint plots, bar plots) were used to explore relationships between features and the target variable.
    - A strong negative correlation was observed between fixed acidity and pH.
    - A strong positive correlation was observed between fixed acidity and citric acid, and fixed acidity and density.
    - Volatile acidity shows a negative correlation with quality.
    - Alcohol content shows a positive correlation with quality.
- Outlier detection using the Interquartile Range (IQR) revealed outliers in most features.

## Data Preparation
- The 'quality' column was categorized into three classes: 0 (poor), 1 (average), and 2 (finest).
- The data was split into training and testing sets (80% train, 20% test).

## Model Training and Evaluation
- Four models were trained and evaluated: Gaussian Naive Bayes, Decision Tree Classifier, Random Forest Classifier, and Logistic Regression.
- The models were trained on different sample sizes (1%, 10%, and 100%) of the training data.
- The performance of the models was evaluated using accuracy and F-beta score (with beta=0.5).
- The Random Forest Classifier consistently performed well across different sample sizes, showing good accuracy and F-beta scores on the test data.

## Feature Importance
- Feature importance was determined using the trained Random Forest Classifier.
- The most important features for predicting wine quality were found to be: alcohol, volatile acidity, sulphates, density, and total sulfur dioxide.

## Hyperparameter Tuning
- Grid Search with cross-validation was performed on the Random Forest Classifier to find optimal hyperparameters (`n_estimators` and `max_features`).
- The optimized model achieved a slightly lower accuracy and F-score on the testing data compared to the unoptimized model, suggesting that the default parameters or other hyperparameter combinations might be more suitable. The best parameters found were `max_features=4` and `n_estimators=30`.

## Predictions
- The optimized Random Forest model was used to make predictions on new wine data.

## Conclusion
The project successfully explored the wine quality dataset, identified key features influencing wine quality, and trained several machine learning models to predict quality. The Random Forest Classifier showed promising results, and further hyperparameter tuning or exploring other models could potentially improve performance.
