import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load data
df = pd.read_csv('train.csv')

# 2. Feature selection
# We'll use:
# - GrLivArea (square footage)
# - BedroomAbvGr (number of bedrooms)
# - Bathrooms: FullBath + 0.5*HalfBath

df['Bathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']

features = df[['GrLivArea', 'BedroomAbvGr', 'Bathrooms']]
target = df['SalePrice']

# 3. Handle missing values, if any
features = features.fillna(0)
target = target.fillna(target.mean())

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# 5. Model training
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test Mean Squared Error: {mse:.2f}")

# 7. Model usage example
example = pd.DataFrame({
    'GrLivArea': [2000],
    'BedroomAbvGr': [3],
    'Bathrooms': [2.5]
})
predicted_price = model.predict(example)[0]
print(f"Predicted price for 2000 sqft, 3 beds, 2.5 baths: ${predicted_price:,.0f}")

# Optional: Check model coefficients
print("Coefficients:", dict(zip(features.columns, model.coef_)))
print("Intercept:", model.intercept_)
