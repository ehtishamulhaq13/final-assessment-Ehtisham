import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


df = pd.read_csv('Adobe (ADBE) From 1986 To Dec-2024 (1).csv')
print("Shape:", df.shape)
print(df.head())
print(df.isnull().sum())

df['Prev_Open'] = df['Open'].shift(1)
df['Prev_High'] = df['High'].shift(1)
df['Prev_Low'] = df['Low'].shift(1)
df['Prev_Close'] = df['Close'].shift(1)
df['Prev_Volume'] = df['Volume'].shift(1)
df = df.dropna().reset_index(drop=True)

feature_cols = ['Prev_Open', 'Prev_High', 'Prev_Low', 'Prev_Close', 'Prev_Volume']
target_col = 'Close'
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
results = []

def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name:<20} -> R2 (accuracy): {r2:.4f}   RMSE: {rmse:.2f}   MAE: {mae:.2f}")
    results.append((name, r2, rmse, mae))

print("\n== MACHINE LEARNING ==")

lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
lin_preds = lin_model.predict(X_test_scaled)
evaluate("Linear Regression", y_test, lin_preds)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
ridge_preds = ridge_model.predict(X_test_scaled)
evaluate("Ridge Regression", y_test, ridge_preds)

lasso_model = Lasso(alpha=1.0, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)
lasso_preds = lasso_model.predict(X_test_scaled)
evaluate("Lasso Regression", y_test, lasso_preds)

sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
model_preds = {
    "Linear Regression": lin_preds,
    "Ridge Regression": ridge_preds,
    "Lasso Regression": lasso_preds
}

for ax, (name, preds) in zip(axes, model_preds.items()):
    ax.plot(y_test.values, label="Actual", color="#1f77b4", linewidth=1.2)
    ax.plot(preds, label="Predicted", color="#ff7f0e", linewidth=1.2, alpha=0.8)
    ax.set_title(f"{name}: Actual vs Predicted")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Close Price")
    ax.legend()

plt.tight_layout()
plt.savefig("linear_ridge_lasso_actual_vs_predicted.png", dpi=120)
plt.show()

# Scatter plot version (Actual vs Predicted correlation)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, (name, preds) in zip(axes, model_preds.items()):
    sns.scatterplot(x=y_test.values, y=preds, ax=ax, alpha=0.5, color="#2ca02c")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            color="red", linestyle="--", label="Perfect Prediction Line")
    ax.set_title(f"{name}: Correlation Plot")
    ax.set_xlabel("Actual Close")
    ax.set_ylabel("Predicted Close")
    ax.legend()

plt.tight_layout()
plt.savefig("linear_ridge_lasso_scatter.png", dpi=120)
plt.show()

print("\n== DEEP LEARNING: Dense Neural Network ==")

dense_model = Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])
dense_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
dense_model.fit(X_train_scaled, y_train, validation_split=0.1,
                 epochs=100, batch_size=32, verbose=0)
dense_preds = dense_model.predict(X_test_scaled, verbose=0).flatten()
evaluate("Dense NN", y_test, dense_preds)

print("\n== FINAL COMPARISON (higher R2 / lower RMSE & MAE = better) ==")
results_df = pd.DataFrame(results, columns=["Model", "R2_Accuracy", "RMSE", "MAE"])
results_df = results_df.sort_values("R2_Accuracy", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))
results_df.to_csv("stock_model_comparison.csv", index=False)