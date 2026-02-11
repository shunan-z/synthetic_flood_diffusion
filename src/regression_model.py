import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib  
import os      

class FloodXGBModel:
    def __init__(self, params=None):
        """
        Initialize the XGBoost Regressor with optional parameters.
        """
        self.default_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1  # Use all cores
        }
        if params:
            self.default_params.update(params)
            
        self.model = xgb.XGBRegressor(**self.default_params)
        self.feature_cols = [
            "elev_mean", "elev_var", "Category", "Speed", 
            "Tide", "Direction_sin", "Direction_cos", 
            "latitude", "longitude"
        ]

    def train(self, df):
        """
        Prepares data, trains the model, and prints metrics.
        Returns: trained model, metrics dictionary
        """
        print("\n--- Training XGBoost Baseline ---")
        
        # 1. Prepare Data
        # Ensure we only use rows with valid targets
        data = df[self.feature_cols + ["target_value"]].dropna().copy()
        X = data[self.feature_cols]
        y = data["target_value"]
        
        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 3. Fit
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
        self.model.fit(X_train, y_train)
        
        # 4. Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        train_rmse = train_mse ** 0.5
        test_rmse = test_mse ** 0.5
        
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Test RMSE:  {test_rmse:.4f}")
        print(f"Test R²:    {test_r2:.4f}")
        
        return {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "test_r2": test_r2
        }

    def plot_importance(self):
        """
        Plots Gain and Weight feature importance side-by-side.
        """
        if not hasattr(self.model, "feature_importances_"):
            print("Model not trained yet!")
            return

        # (a) Gain-based importance (default)
        gain_imp = pd.Series(
            self.model.feature_importances_, 
            index=self.feature_cols
        ).sort_values(ascending=False)

        # (b) Weight-based importance (frequency)
        booster = self.model.get_booster()
        # get_score returns dict like {'f0': 5}, map back to names
        weight_scores = booster.get_score(importance_type="weight")
        weight_imp = pd.Series(dtype=float)
        
        for feat in self.feature_cols:
            weight_imp[feat] = weight_scores.get(feat, 0)
        
        weight_imp = weight_imp.sort_values(ascending=False)

        # Plotting
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        gain_imp.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("Feature Importance (Gain / Information)")
        plt.ylabel("Relative Gain")
        
        plt.subplot(1, 2, 2)
        weight_imp.plot(kind="bar", color="salmon", edgecolor="black")
        plt.title("Feature Importance (Weight / Frequency)")
        plt.ylabel("Number of Splits")
        
        plt.tight_layout()
        plt.show()

    def predict(self, X_new):
        """Wrapper for prediction"""
        return self.model.predict(X_new[self.feature_cols])

    # --- ADD THIS NEW METHOD ---
    def save(self, filepath):
        """
        Saves the entire class instance (model + config) to a file.
        """
        # Ensure the folder exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the whole object using joblib
        joblib.dump(self, filepath)
        print(f"✅ Model saved successfully to: {filepath}")

    @staticmethod
    def load(filepath):
        """
        Static method to load a saved model.
        Usage: model = FloodXGBModel.load("path/to/model.pkl")
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        print(f"✅ Model loaded from: {filepath}")
        return model