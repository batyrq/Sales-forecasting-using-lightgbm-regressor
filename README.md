# Sales Forecasting with LightGBM Regressor

-----

## Project Description

This project tackles the challenge of **sales forecasting** using historical transactional data, aiming to predict future sales accurately. Leveraging **LightGBM**, a highly efficient gradient boosting framework, this solution processes time-series data, enriches it with external factors like oil prices and holiday events, and builds a robust regression model.

The primary goal is to predict daily sales for various product families across different stores, demonstrating the power of machine learning in business analytics and demand prediction. This work was developed as a solution for the [Store Sales - Time Series Forecasting Kaggle competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview), where it achieved a competitive score of **1.30791**.

-----
## Kaggle Competition

This project is a solution to the **"Store Sales - Time Series Forecasting"** competition on Kaggle. The objective of the competition was to forecast the sales of various products at different Ecuadorian stores, considering external factors such as oil prices, holidays, and promotions.

  * **Competition Link:** [https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview)

-----

## Dataset

The project utilizes several CSV files provided by the Kaggle competition:

  * **`train.csv`**: Contains the historical sales data, including `date`, `store_nbr`, `family`, `sales`, `onpromotion`.
  * **`test.csv`**: The test set for which sales predictions need to be made. It has the same columns as `train.csv` except for `sales`.
  * **`oil.csv`**: Provides daily crude oil prices (`dcoilwtico`). This is an important external factor that might influence sales.
  * **`holidays_events.csv`**: Contains information about various holidays and events, including their `date`, `type`, `locale`, `locale_name`, `description`, and `transferred` status.
  * **`sample_submission.csv`**: A sample submission file indicating the required format for predictions.

All datasets are loaded using `pandas`.

-----

## Feature Engineering

Robust feature engineering is crucial for time-series forecasting. The following steps were performed to enrich the dataset for the LightGBM model:

1.  **Handling Missing Oil Prices**:

      * Missing values in the `dcoilwtico` column of the `oil.csv` dataset are imputed with the mean of the existing oil prices.

    <!-- end list -->

    ```python
    oil['dcoilwtico'] = oil['dcoilwtico'].fillna(oil['dcoilwtico'].mean())
    ```

2.  **Categorical Feature Encoding**:

      * The `family` column, which is a categorical feature representing product categories, is transformed into numerical labels using `LabelEncoder` from `sklearn.preprocessing`. This is applied consistently to both `train` and `test` datasets.

    <!-- end list -->

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(train['family'])
    train['family'] = le.transform(train['family'])
    test['family'] = le.transform(test['family']) # Applied to test set as well
    ```

3.  **Date-based Features**:

      * The `date` columns in both `train` and `test` datasets are converted to datetime objects.
      * New features are extracted from the `date` to capture temporal patterns:
          * `day_of_week`: Day of the week (Monday=0, Sunday=6).
          * `day_of_month`: Day of the month.
          * `year`: The year.

    <!-- end list -->

    ```python
    train['date'] = pd.to_datetime(train['date'])
    train['day_of_week'] = train['date'].dt.dayofweek
    train['day_of_month'] = train['date'].dt.day
    train['year'] = train['date'].dt.year

    test['date'] = pd.to_datetime(test['date'])
    test['day_of_week'] = test['date'].dt.dayofweek
    test['day_of_month'] = test['date'].dt.day
    test['year'] = test['date'].dt.year
    ```

4.  **Holiday Information Integration**:

      * A new binary feature, `isholiday`, is created in both `train` and `test` datasets, initialized to `0`.
      * The `holidays_events.csv` is processed. For any holiday marked as `transferred=='True'`, the corresponding dates in `train` and `test` datasets have their `isholiday` flag set to `1`. This implies `transferred` holidays are considered actual holidays in this context.

    <!-- end list -->

    ```python
    train['isholiday'] = np.zeros(len(train))
    test['isholiday'] = np.zeros(len(test))

    holidays['date'] = pd.to_datetime(holidays['date'])

    for i in range(len(holidays)):
        if holidays.loc[i, 'transferred'] == 'True': # Assuming 'True' indicates a significant holiday
            # Update 'isholiday' in train
            train.loc[train['date'] == holidays.loc[i, 'date'], 'isholiday'] = 1
            # Update 'isholiday' in test
            test.loc[test['date'] == holidays.loc[i, 'date'], 'isholiday'] = 1
    ```

    *Note: The original code iterated through each row of `train` and `test` for every holiday, which is inefficient. The corrected approach above uses `loc` with boolean indexing for faster updates.*

5.  **Oil Price Integration**:

      * The `dcoilwtico` feature is added to both `train` and `test` datasets, initialized to zeros.
      * A dictionary `posoil` is created to quickly map dates to their corresponding oil prices from the `oil` DataFrame.
      * For each entry in `train` and `test`, if its `date` matches a date in `posoil`, the respective oil price is assigned. Otherwise, the mean oil price is used as a fallback.

    <!-- end list -->

    ```python
    train['dcoilwtico'] = np.zeros(len(train))
    test['dcoilwtico'] = np.zeros(len(test))
    oil['date'] = pd.to_datetime(oil['date'])

    posoil = {}
    for i in range(len(oil)):
        posoil[oil.loc[i, 'date']] = oil.loc[i, 'dcoilwtico']

    # Efficiently merge oil prices (original loop can be slow for large datasets)
    train['dcoilwtico'] = train['date'].map(posoil).fillna(np.mean(oil['dcoilwtico']))
    test['dcoilwtico'] = test['date'].map(posoil).fillna(np.mean(oil['dcoilwtico']))
    ```

    *Note: Similar to holidays, the original oil price merging loops were very slow for large datasets. The corrected approach above uses `map` and `fillna` for much better performance.*

-----

## Model

The core predictive model used in this project is **LightGBM Regressor** (`lgb.LGBMRegressor`). LightGBM is chosen for its speed, efficiency, and high performance on tabular data, especially in gradient boosting tasks.

The model is configured with the following key parameters:

  * **`n_estimators`**: `1000` (Number of boosting rounds/trees)
  * **`learning_rate`**: `0.05` (Step size shrinkage to prevent overfitting)
  * **`num_leaves`**: `64` (Maximum number of leaves in one tree)
  * **`random_state`**: `42` (Seed for random number generation for reproducibility)

<!-- end list -->

```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42
)
```

-----

## Training & Prediction

**Training**:
The model is trained on the prepared `X_train` features to predict the `sales` (`y_train`). The `date` column is dropped from `X_train` as its components have already been extracted into numerical features.

```python
y_train = train['sales']
X_train = train.drop(['sales', 'date'], axis=1) # Features for training

model.fit(X_train, y_train) # Model training
```

**Prediction**:
After training, the model is used to make predictions on the preprocessed `test` dataset. The `date` column is also dropped from the `test` set before prediction.

```python
test = test.drop('date', axis=1) # Features for prediction
sample_submission['sales'] = model.predict(test) # Generate predictions
```

**Post-processing**:
Negative sales predictions, which are not physically possible, are capped at `0` to ensure realistic output.

```python
for i in range(len(sample_submission)):
    sample_submission.loc[i, 'sales'] = max(sample_submission.loc[i, 'sales'], 0)
```

Finally, the predictions are saved into `submission.csv` in the format required by the Kaggle competition.

```python
sample_submission.to_csv('submission.csv', index=False)
```

-----

## Results

The model achieved a **Root Mean Squared Logarithmic Error(RMSLE)** score of **1.30791** on the Kaggle competition leaderboard. This indicates a good level of accuracy in forecasting sales, demonstrating the effectiveness of LightGBM combined with the engineered features.

-----

## Usage

To run this sales forecasting solution:

1.  **Download the datasets**:

      * Obtain `train.csv`, `test.csv`, `oil.csv`, `holidays_events.csv`, and `sample_submission.csv` from the [Kaggle competition page](https://www.google.com/search?q=https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
      * Place them in the `/kaggle/input/store-sales-time-series-forecasting/` directory or update the file paths in the script to match your local setup.

2.  **Execute the script**:
    Run the Python script containing the provided code. It will perform data loading, feature engineering, model training, prediction, and generate the `submission.csv` file.

    ```bash
    python your_script_name.py
    ```

-----

## Installation

Ensure you have Python installed. Then, install the necessary libraries using `pip`:

```bash
pip install pandas numpy scikit-learn lightgbm
```

-----
