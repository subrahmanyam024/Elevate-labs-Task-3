# Elevate-labs-Task-3

Housing Price Prediction with Linear Regression
Overview
This project implements simple and multiple linear regression models to predict house prices using the Housing.csv dataset. The dataset contains 545 entries with features like area, bedrooms, bathrooms, stories, parking, and categorical variables (mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus). The code preprocesses the data, trains models, evaluates performance with MAE, MSE, and R², visualizes results, and interprets feature impacts.
Dataset

File: Housing.csv
Rows: 545
Columns: 13
Numerical: price (target, int64), area, bedrooms, bathrooms, stories, parking (int64)
Categorical: mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus (object)


No Missing Values: All columns have 545 non-null entries.

Preprocessing

One-Hot Encoding:

Categorical columns are converted to binary columns using pd.get_dummies with drop_first=True to avoid multicollinearity.
Example: mainroad ('yes', 'no') becomes mainroad_yes (True/False).
Results in 8 boolean columns: mainroad_yes, guestroom_yes, basement_yes, hotwaterheating_yes, airconditioning_yes, prefarea_yes, furnishingstatus_semi-furnished, furnishingstatus_unfurnished.
Note: Boolean (bool) dtype is used for efficiency (1 byte vs. 8 bytes for int64) and is compatible with scikit-learn’s LinearRegression. No conversion to numerical (int64/float64) is needed.


Scaling:

Numerical features (area, bedrooms, bathrooms, stories, parking) are standardized using StandardScaler (mean=0, std=1) to improve model performance.


Features and Target:

Features (X): 13 columns (5 scaled numerical + 8 boolean).
Target (y): price (int64).



Models

Simple Linear Regression:

Uses only the area feature (scaled) to predict price.
Metrics:
MAE: 1,474,748.13 (~$1.47M average error)
MSE: 3,675,286,604,768.19 (~3.68T, large due to price scale)
R²: 0.27 (explains 27% of price variance, indicating poor fit)


Interpretation: area alone is a weak predictor, as other features (e.g., bathrooms, airconditioning) are significant.


Multiple Linear Regression:

Uses all 13 features to predict price.
Metrics:
MAE: 970,043.40 (~$970K, better than simple regression)
MSE: 1,754,318,687,330.67 (~1.75T, lower than simple regression)
R²: 0.65 (explains 65% of price variance, decent fit)


Interpretation: Including all features improves accuracy, but some error remains due to data variability or unmodeled factors.



Code Structure
The code (housing_regression_complete.py) performs the following:

Imports:

Libraries: pandas, numpy, sklearn (train_test_split, LinearRegression, StandardScaler, metrics), matplotlib.pyplot.


Preprocessing:

Loads Housing.csv.
Applies one-hot encoding to categorical columns (bool dtype).
Scales numerical features.


Data Splitting:

Splits data into 80% training and 20% testing sets (random_state=42).


Simple Linear Regression:

Trains model on area vs. price.
Evaluates with MAE, MSE, R².
Plots scatter of scaled area vs. price with regression line.


Multiple Linear Regression:

Trains model on all features.
Evaluates with MAE, MSE, R².
Plots actual vs. predicted prices (points near diagonal indicate better fit).
Displays coefficients to show feature impacts.


Coefficients:

Example output:Feature                           Coefficient
airconditioning_yes               791426.74
hotwaterheating_yes              684649.89
prefarea_yes                     629890.57
bathrooms                        549420.50
area                             511615.56
basement_yes                     390251.18
mainroad_yes                     367919.95
stories                          353158.43
guestroom_yes                    231610.04
parking                          193542.78
bedrooms                          56615.57
furnishingstatus_semi-furnished -126881.82
furnishingstatus_unfurnished    -413645.06


Interpretation:
Positive coefficients (e.g., airconditioning_yes: +$791K) increase price.
Negative coefficients (e.g., furnishingstatus_unfurnished: -$413K) decrease price.
Larger coefficients (e.g., airconditioning_yes, bathrooms) indicate stronger impact.





Usage

Requirements:

Python 3.x
Libraries: pandas, numpy, scikit-learn, matplotlib
Install: pip install pandas numpy scikit-learn matplotlib


Running the Code:

Place Housing.csv in the same directory as housing_regression_complete.py.
Run: python housing_regression_complete.py
Outputs:
Console: MAE, MSE, R² for both models, coefficient table.
Plots: Simple regression line, multiple regression actual vs. predicted scatter.



Results

Simple Linear Regression:
Weak performance (R² = 0.27) due to using only area.
Large errors (MAE: ~$1.47M, MSE: ~3.68T).


Multiple Linear Regression:
Better performance (R² = 0.65, MAE: ~$970K, MSE: ~1.75T).
Captures more price variance using all features.


Plots:
Simple regression: Shows linear fit of area vs. price (scattered points indicate poor fit).
Multiple regression: Points closer to diagonal line suggest better predictions.



Recommendations

Improve Simple Regression:

R² (0.27) is low. Use multiple regression for predictions.
Try other single features (e.g., bathrooms) or non-linear models.


Enhance Multiple Regression:

Add interaction terms: X['area_bathrooms'] = X['area'] * X['bathrooms']
Use RandomForestRegressor for non-linear relationships:from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
print(f"Random Forest R²: {r2_score(y_test, rf_model.predict(X_test)):.4f}")


Apply cross-validation:from sklearn.model_selection import cross_val_score
scores = cross_val_score(multi_model, X, y, cv=5, scoring='r2')
print(f"Cross-validated R²: {scores.mean():.4f} (±{scores.std():.4f})")




Outlier Handling:

Large MSE suggests outliers. Filter extreme values:from scipy.stats import zscore
df = df[(zscore(df['price']) < 3) & (zscore(df['area']) < 3)]




Prediction Function:

Add a function to predict new house prices (example in code comments):def predict_house_price(area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, 
                       basement, hotwaterheating, airconditioning, prefarea, furnishingstatus,
                       model, scaler, columns, categorical_cols):
    input_data = pd.DataFrame({...})
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True, dtype=bool)
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])
    for col in columns:
        if col not in input_encoded:
            input_encoded[col] = False
    input_encoded = input_encoded[columns]
    price = model.predict(input_encoded)[0]
    print(f"Predicted price: ${price:,.2f}")





Notes

Bool Dtypes: One-hot encoded columns are bool (e.g., mainroad_yes), which is efficient and compatible with LinearRegression. No need to convert to int64/float64.
Price Scale: Prices are in large units (e.g., Indian Rupees, ~$1.75M–13.3M), leading to large MSE values.

Example Output
Simple Linear Regression (area vs price):
MAE: 1474748.13
MSE: 3675286604768.19
R²: 0.27

Multiple Linear Regression (all features):
MAE: 970043.40
MSE: 1754318687330.67
R²: 0.65

Multiple Linear Regression Coefficients:
                            Feature    Coefficient
airconditioning_yes              791426.74
hotwaterheating_yes              684649.89
...
furnishingstatus_unfurnished    -413645.06


