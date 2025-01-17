# Supervised Learning for Tourism Dataset: Revenue Maximization via Activity Ranking

## Overview
This project evaluates supervised learning algorithms to optimize revenue in the tourism sector. Using the provided dataset, we rank six activity categories (`Nature`, `Historical`, `Cultural`, `Beach`, `Adventure`, `Urban`) to maximize revenue (`Revenue`) and/or revenue per visitor (`Revenue/Visitors`) for a country-specific context.

## Dataset
The dataset is sourced from Kaggle: [Tourism Dataset](https://www.kaggle.com/datasets/umeradnaan/tourism-dataset). It contains information on tourism activities across seven countries (`India`, `USA`, `Brazil`, `France`, `Egypt`, `China`, `Australia`) and includes:

- **Category**: Tourism activity type (e.g., Nature, Historical).
- **Country**: Country of the activity.
- **Revenue**: Total revenue generated.
- **Visitors**: Total visitors for the activity.
- **Rating**: Visitor ratings of activities.
- Additional engineered features: `Log_Visitors`, `Revenue_per_Visitor`, `Log_Revenue`.

## Objectives
1. Preprocess the dataset to extract meaningful patterns and correlations.
2. Build supervised learning models to rank activity categories for maximizing revenue.
3. Compare model performance based on accuracy and meaningfulness of revenue hierarchies.

## Algorithms Implemented

### Classification-Based Approaches
1. **ID3 Decision Tree**
   - Predicts activity categories using optimized quantile binning for numerical features.

2. **Naive Bayes**
   - Evaluated with two feature sets:
     - Basic: `Visitors`, `Rating`, `Accommodation Available`.
     - Enhanced: Includes `Revenue_per_Visitor`, `Log_Visitors`, and `Log_Revenue`.

3. **AdaBoost**
   - Implemented using:
     - Custom AdaBoost with decision stumps.
     - Scikit-learn’s `AdaBoostClassifier`.

4. **k-Nearest Neighbors (kNN)**
   - Treated as a classifier for activity categories.
   - Tuned with `GridSearchCV` for parameters like `n_neighbors`, `weights`, and distance metrics (`Euclidean`, `Manhattan`, `Minkowski`).

### Regression-Based Approaches
1. **Linear Regression**
   - Predicts revenue directly to rank activities based on profitability.
   - Raw and log-transformed features/targets were tested for improved performance.

2. **Logistic Regression**
   - Predicts activity categories with features like `Log_Visitors` and `Revenue_per_Visitor`.
   - Applied class balancing to handle imbalanced datasets and capture linear trends effectively.

## Methodology
1. **Preprocessing**:
   - Filtered dataset by country for localized insights.
   - Engineered features: `Log_Visitors`, `Log_Revenue`, `Revenue_per_Visitor`.
   - Encoded categorical variables using `LabelEncoder`.
   - Normalized features for kNN using `StandardScaler`.
2. **Model Training and Evaluation**:
   - Split data into training/testing sets (80%-20%), ensuring stratified sampling for balanced categories.
   - Evaluation metrics:
     - Classification: Accuracy.
     - Regression: MSE, MAE, R² score.
3. **Revenue Hierarchy Calculation**:
   - For classification models, categories were ranked based on predicted average revenue.
   - Regression models ranked activities based on direct revenue predictions.

## Results

### Classification-Based Models
- **ID3 Decision Tree**:
  - Achieved the highest accuracy among classification models (e.g., 25.29% in the USA).
  - Enhanced preprocessing (quantile-based binning) significantly improved results.

- **Naive Bayes**:
  - Enhanced features outperformed the basic set, with accuracy up to 22.21% in France.
  - Results highlighted the algorithm's simplicity and limitations with interdependent features.

- **AdaBoost**:
  - Scikit-learn’s implementation slightly outperformed the custom version (e.g., 19.88% in Australia).
  - Consistently identified `Cultural` and `Beach` as top categories across countries.

- **k-Nearest Neighbors (kNN)**:
  - Tuned as a classifier, achieving moderate accuracy with hyperparameter optimization.
  - Effective in countries with distinct category patterns, but overall performance was limited by high-dimensional data.

### Regression-Based Models
- **Linear Regression**:
  - Log-transformed targets improved R² scores (e.g., 0.80 in India).
  - Raw features/targets yielded poor predictions and less reliable revenue hierarchies.

- **Logistic Regression**:
  - Provided a robust baseline for category classification with accuracy ranging from 12.78% (India) to 19.88% (Australia).
  - Captured linear trends but struggled with the dataset’s non-linear complexities.

## Visualizations
- Accuracy comparisons across models and countries.
- Revenue hierarchies for classification and regression models.
- Error metrics (MSE, MAE, R²) for regression-based approaches.

## Code Structure

```
ml-algorithms-tourism-analysis/
├── algorithms/
│   ├── adaboost/
│   │   ├── adaboost_comparison.py     # Compares Scikit-learn and custom AdaBoost
│   │   └── custom_adaboost.py         # Custom AdaBoost implementation
│   ├── naive_bayes/
│   │   ├── bayes_naive_complex.py     # Naive Bayes with advanced features
│   │   └── bayes_naive_small.py       # Basic Naive Bayes implementation
│   ├── regression/
│   │   ├── linear_regression.py       # Linear Regression for revenue prediction
│   │   └── logistic_regression.py     # Logistic Regression for classification
│   ├── id3.py                         # ID3 Decision Tree for activity classification
│   ├── knn.py                         # k-Nearest Neighbors for classification
├── data/
│   └── tourism_dataset.csv            # Kaggle tourism dataset
├── documentation/
├── scripts/
│   ├── data_analysis.py               # Exploratory data analysis
│   └── download_data.py               # Dataset download script
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run individual scripts for specific models or all models for comprehensive results:
   ```bash
   python adaboost_comparison.py
   python knn.py
   ```

## References
- Scikit-learn Documentation
- Kaggle Dataset: [Tourism Dataset](https://www.kaggle.com/datasets/umeradnaan/tourism-dataset)
