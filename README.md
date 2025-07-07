# Taxi Guru: Predicting Taxi Fare with Machine Learning

**Taxi Guru** is a machine learning project aimed at predicting taxi fare amounts using real-world trip data. Built as part of an applied data science challenge, the project explores the relationship between geographical, temporal, and engineered features in determining the cost of a ride.

The core objective was not only to build a high-performing regression model but to gain practical experience in the end-to-end machine learning pipeline — from data cleaning and preprocessing to model comparison, hyperparameter tuning, and evaluation.

---

This project was developed independently and submitted to a Kaggle-hosted prediction challenge. The dataset consisted of thousands of NYC taxi rides, including pickup/drop-off coordinates, timestamps, and passenger counts. The goal was to predict the final fare amount for each trip.

To build this system, I implemented and compared a variety of machine learning algorithms including:

- **Linear Regression** – as a baseline model  
- **Logistic Regression** – for classification tasks during data validation  
- **K-Nearest Neighbors (KNN)** – to explore locality-based predictions  
- **Decision Tree & Random Forest** – for non-linear regression and feature importance analysis  
- **Multi-Layer Perceptron (MLP)** – to test a neural network approach to tabular data

Through extensive **exploratory data analysis (EDA)** and visualization, I examined trends in fare distribution, mapped outliers using geolocation clusters, and identified key features like trip duration, distance (via Haversine formula), and rush hour indicators.

A significant part of the effort was directed toward **feature engineering**. I created additional fields including:
- Day of the week
- Hour of the day
- Estimated distance
- Geospatial interaction terms

This helped uncover relationships that were not explicit in the original dataset, enabling better performance across all models.

After comparing models on both training and validation sets, Random Forest and MLP yielded the most consistent results with minimal overfitting. Hyperparameters were optimized using GridSearchCV for tree-based models and manual tuning for MLP due to training complexity.

The final notebook includes:
- Cleaned and documented code
- Visual EDA and correlation analysis
- Evaluation metrics: MAE, RMSE, R²
- Cross-validation performance
- Final predictions

This project demonstrates my ability to not only work with structured data at scale but also to think critically about the modeling process, challenge assumptions, and optimize for both performance and interpretability.

