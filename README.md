
# Feature Selection on California Housing Price Prediction 🏠

using feature selection techniques to analyze what would be it's effects on R2 Score & Mean Square Error. Comaparing diferent Feature selection method like PCA, Mutual_Inforemation(SelectKBased) &  Recursive Feature Elimination (RFE).


## Data 
The California housing dataset consists of `20640` data points, with each datapoint having `8 features`. This dataset was obtained from the StatLib repository -
[Dataset_Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)  

This dataset was derived from the 1990 U.S. census, using one row per census
block group. A block group is the smallest geographical unit for which the U.S.
Census Bureau publishes sample data (a block group typically has a population
of 600 to 3,000 people).

A household is a group of people residing within a home. The average
number of rooms and bedrooms in this dataset are provided per household.


**Features Description**

1. `MedInc`     : median income in block group
2. `HouseAge`   : median house age in block group
3. `AveRooms`   : average number of rooms per household
4. `AveBedrms`  : average number of bedrooms per household
5. `Population` : block group population
6. `AveOccup`   : average number of household members
7. `Latitude`   : block group latitude
8. `Longitude`  : block group longitude

**Target** : `Median house value` - for California districts, expressed in hundreds of thousands of dollars ($100,000).

[California Housing Dataset in Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

## Libraries Used 

**Language:** Python
**Packages:** Sklearn, Matplotlib, Pandas


## Method Details 📜

### Filter Based Feature Selection Methods

#### Mutual Information Regression

#### Using SelectKBest

Here we make use of `sklearn.feature_selection.SelectKBest`, a filter-based feature selection method and pass parameters like  `mutual_info_regression` (a score function that computes the mutual information between variables), `k` value (numerical value that depicts top number of features to be picked.). The output obtained here are `k` best features (x_new).

Now perform the train test split on the `x_new` and `y`, then perform general Regression implementation and calculate the r2_score.


### Wrapper Based Methods for Feature Selection

#### Recursive Feature Elimination (RFE)

We make use of `sklearn.feature_selection.RFE` module for performing recursive feature elimination. `RFE` method has `estimator`, `n_features_to_select`, `step` as input parameters.   

As a estimator we can use either `Linear Regressor` or `Lasso Regularization`. Here we have decided to use `Lasso`. `n_features_to_select` takes integer value, basically number of features and `step` - corresponds to the number of features to remove at each iteration.

Finally, generate train test split on `x_new`, `y` and proceed with Regressor implementation, r2_score calculation.


### Dimensionality Reduction 

#### Principal Component Analysis

As part of `PCA` implementation, firstly import `sklearn.decomposition.PCA` class. Initialize `PCA` by passing the parameters like `n_components` (number of components to keep) and `svd_solver` is set to `full`.

Apply fit_transform(), resultant `x_new` will contain  4 components (as defined in the PCA params).

As Final step generate train test split on `x_new`, `y` and proceed with Regressor implementation, r2_score calculation.





## Evaluation On vanilla Model 🔍

| Metric        | Value         |
| ------------- | ------------- |
| R2 Score      | 0.60          |
| MSE		| 0.54
The above table gives the R2_score & MSE of the Simple Regression model (with all features included).

Below table gives `r2_scores` obtained using various feature selection Methods

## Evaluation result of different technique

| Technique Used | K value | R2 Score | MSE   |
|----------------|---------|---------|-------|
| `SelectKBest`  | 3       |   0.59  | 0.55 |
| `Recursive Feature Elimination (RFE)`    | 5      | 0.24    | 0.96 |
| `Principal Component Analysis`    | 4       | 0.45    | 0.69 |



### 🔑 Evaluation Conclusions 💡

1. `SelectKBest`, `SelectPercentile` reduced the model features to `half (4 features)` while maintaining the r2_score.
2. `RFE` method has obtained `3 features` but r2_score was significantly reduced i;e `0.24`. 
5. `PCA` has reduced features to 4, its respective r2_score is `0.45`.

## ⚡Key Takeaways⚡

Good understanding of how various feature selection methods work. 

Reducing the complexity of the models.

## Acknowledgements 🙌

[SelectKBest](https://medium.com/@Kavya2099/optimizing-performance-selectkbest-for-efficient-feature-selection-in-machine-learning-3b635905ed48)

[Recursive- Feature - elimination](https://www.analyticsvidhya.com/blog/2023/05/recursive-feature-elimination/)

[Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
 
