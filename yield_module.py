# Auto-converted from notebook: /mnt/data/yield_prediction.ipynb

# Interactive Crop Yield Prediction System
# Takes user input and predicts yield using CSV data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import pickle
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

class CropYieldSystem:
    """Complete interactive crop yield prediction system"""

    def __init__(self, csv_path: str = r"D:\New folder\Desktop\agri\Indian_Crop_Fusion_Master_CLEANED.csv"):
        self.csv_path = csv_path
        self.df = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.target_column = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.is_trained = False
        self.column_info = {}

    def load_and_prepare_data(self):
        """Load CSV and prepare data"""
        print("🌾 CROP YIELD PREDICTION SYSTEM")
        print("=" * 60)
        print(f"\n📂 Loading data from: {self.csv_path}")

        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    self.df = pd.read_csv(self.csv_path, encoding=encoding)
                    print(f"✅ Successfully loaded with {encoding} encoding")
                    break
                except (UnicodeDecodeError, FileNotFoundError):
                    continue

            if self.df is None:
                raise FileNotFoundError(f"Could not load {self.csv_path}")

            print(f"📊 Dataset shape: {self.df.shape}")
            print(f"📋 Columns found: {list(self.df.columns)}")

            # Clean column names
            self.df.columns = self.df.columns.str.strip()

            # Analyze columns
            self._analyze_columns()

            # Identify target variable
            self._identify_target()

            return True

        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            print("Creating sample dataset for demonstration...")
            self._create_sample_data()
            return True

    def _analyze_columns(self):
        """Analyze dataset columns"""
        print("\n📈 Column Analysis:")
        print("-" * 60)

        for col in self.df.columns:
            dtype = self.df[col].dtype
            nulls = self.df[col].isnull().sum()
            unique = self.df[col].nunique()

            self.column_info[col] = {
                'dtype': str(dtype),
                'nulls': nulls,
                'unique': unique,
                'sample': self.df[col].dropna().iloc[0] if len(self.df[col].dropna()) > 0 else None
            }

            if dtype in ['int64', 'float64']:
                self.numerical_columns.append(col)
                if len(self.df[col].dropna()) > 0:
                    stats = f"Range: [{self.df[col].min():.2f}, {self.df[col].max():.2f}]"
                else:
                    stats = "No data"
            else:
                self.categorical_columns.append(col)
                stats = f"Categories: {unique}"

            print(f"  {col:25s} | {str(dtype):10s} | {stats}")

    def _identify_target(self):
        """Identify target variable (yield/production)"""
        yield_keywords = ['yield', 'production', 'output', 'harvest', 'produce', 'tonnes', 'tons']

        # First priority: columns with yield keywords
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in yield_keywords):
                if self.df[col].dtype in ['int64', 'float64']:
                    self.target_column = col
                    print(f"\n🎯 Target variable identified: {self.target_column}")
                    # Remove target from numerical columns if present
                    if col in self.numerical_columns:
                        self.numerical_columns.remove(col)
                    return

        # Second priority: use last numeric column
        if not self.target_column and self.numerical_columns:
            self.target_column = self.numerical_columns[-1]
            self.numerical_columns.remove(self.target_column)
            print(f"\n🎯 Using target variable: {self.target_column}")

    def _create_sample_data(self):
        """Create sample data if CSV not available"""
        np.random.seed(42)
        n = 1000

        self.df = pd.DataFrame({
            'N': np.random.randint(0, 150, n),
            'P': np.random.randint(0, 150, n),
            'K': np.random.randint(0, 210, n),
            'temperature': np.random.uniform(10, 45, n),
            'humidity': np.random.uniform(10, 100, n),
            'ph': np.random.uniform(3.5, 9.5, n),
            'rainfall': np.random.uniform(20, 300, n),
            'crop': np.random.choice(['rice', 'wheat', 'maize', 'cotton'], n)
        })

        # Create synthetic yield
        self.df['yield'] = (
            self.df['N'] * 2 +
            self.df['P'] * 3 +
            self.df['K'] * 1.5 +
            self.df['rainfall'] * 5 +
            (25 - abs(self.df['temperature'] - 25)) * 10 +
            self.df['humidity'] * 2 +
            np.random.normal(0, 100, n)
        ).clip(100, 5000)

        self.target_column = 'yield'
        self.numerical_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.categorical_columns = ['crop']

        print("✅ Sample dataset created")

    def prepare_features(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare features for modeling"""
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()

        # Handle missing values
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown')

        # Encode categorical variables
        for col in self.categorical_columns:
            if col in df.columns and col != self.target_column:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    try:
                        df[col] = self.encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # Handle unknown categories
                        df[col] = 0

        # Create feature engineering
        df = self._create_features(df)

        # Select features (exclude target)
        feature_cols = [col for col in df.columns if col != self.target_column]

        # Only use columns that exist in the dataframe
        self.feature_columns = [col for col in feature_cols if col in df.columns]

        X = df[self.feature_columns]
        y = df[self.target_column] if self.target_column in df.columns else None

        return X, y

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        try:
            # NPK ratios
            if all(col in df.columns for col in ['N', 'P', 'K']):
                df['NPK_total'] = df['N'] + df['P'] + df['K']
                df['NP_ratio'] = df['N'] / (df['P'] + 1)
                df['NK_ratio'] = df['N'] / (df['K'] + 1)
                df['PK_ratio'] = df['P'] / (df['K'] + 1)

            # Temperature stress
            if 'temperature' in df.columns:
                df['temp_stress'] = (df['temperature'] - 25).abs()

            # Moisture index
            if 'rainfall' in df.columns and 'humidity' in df.columns:
                df['moisture_index'] = df['rainfall'] * df['humidity'] / 100

            # pH balance
            if 'ph' in df.columns:
                df['ph_balance'] = (df['ph'] - 7).abs()
        except Exception as e:
            print(f"⚠️  Feature engineering warning: {str(e)}")

        return df

    def train_models(self):
        """Train multiple ML models"""
        print("\n🚀 TRAINING PREDICTION MODELS")
        print("=" * 60)

        X, y = self.prepare_features()

        if y is None or len(y) == 0:
            print("❌ No target variable found. Cannot train models.")
            return {}

        print(f"📊 Training samples: {len(X)}")
        print(f"📊 Features: {len(self.feature_columns)}")
        print(f"🎯 Target: {self.target_column}")
        print(f"📈 Target range: [{y.min():.2f}, {y.max():.2f}]")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)

        # Define models
        models_config = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'Ridge Regression': Ridge(alpha=1.0)
        }

        print("\n🔄 Training in progress...")
        results = {}

        for name, model in models_config.items():
            try:
                if name == 'Ridge Regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                # Metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100

                results[name] = {
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape
                }

                self.models[name] = model

                print(f"\n✅ {name}")
                print(f"   R² Score: {r2:.4f}")
                print(f"   RMSE: {rmse:.2f}")
                print(f"   MAE: {mae:.2f}")
                print(f"   MAPE: {mape:.2f}%")

            except Exception as e:
                print(f"\n❌ {name} failed: {str(e)}")

        self.is_trained = True if self.models else False

        if self.is_trained:
            print("\n✅ Training completed successfully!")
        else:
            print("\n❌ Training failed!")

        return results

    def get_user_input(self) -> Dict:
        """Interactive user input collection"""
        print("\n" + "=" * 60)
        print("📝 ENTER CROP CONDITIONS FOR PREDICTION")
        print("=" * 60)

        user_data = {}

        # Get original feature columns (before engineering)
        original_features = [col for col in self.df.columns
                           if col != self.target_column and col in (self.numerical_columns + self.categorical_columns)]

        # Collect input for each original feature
        for col in original_features:
            if col in self.column_info:
                info = self.column_info[col]
                sample = info.get('sample', '')

                if col in self.numerical_columns:
                    while True:
                        try:
                            prompt = f"\n{col} (e.g., {sample}): "
                            value = input(prompt).strip()

                            if value == '':
                                # Use median as default
                                value = self.df[col].median()
                                print(f"  Using default: {value:.2f}")
                            else:
                                value = float(value)

                            user_data[col] = value
                            break
                        except ValueError:
                            print("  ⚠️  Please enter a valid number")

                elif col in self.categorical_columns:
                    unique_values = list(self.df[col].unique())[:10]
                    print(f"\n{col} options: {', '.join(map(str, unique_values))}")

                    value = input(f"{col} (e.g., {sample}): ").strip()

                    if value == '':
                        value = str(sample)
                        print(f"  Using default: {value}")

                    user_data[col] = value

        return user_data

    def predict_yield(self, input_data: Dict) -> Dict:
        """Predict yield for given conditions"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_models() first.")

        # Create DataFrame
        df_input = pd.DataFrame([input_data])

        # Ensure all original feature columns exist
        for col in self.df.columns:
            if col != self.target_column and col not in df_input.columns:
                if col in self.numerical_columns:
                    df_input[col] = self.df[col].median()
                elif col in self.categorical_columns:
                    df_input[col] = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'unknown'

        # Prepare features
        X_input, _ = self.prepare_features(df_input)

        # Ensure X_input has all required features
        for col in self.feature_columns:
            if col not in X_input.columns:
                X_input[col] = 0

        # Reorder columns to match training data
        X_input = X_input[self.feature_columns]

        # Get predictions from all models
        predictions = {}

        for name, model in self.models.items():
            try:
                if name == 'Ridge Regression':
                    X_scaled = self.scalers['standard'].transform(X_input)
                    pred = model.predict(X_scaled)[0]
                else:
                    pred = model.predict(X_input)[0]

                predictions[name] = max(0, pred)
            except Exception as e:
                print(f"⚠️  Warning: {name} prediction failed: {str(e)}")
                continue

        # Ensemble prediction (average)
        if predictions:
            predictions['Ensemble (Average)'] = np.mean(list(predictions.values()))

        return predictions

    def display_prediction(self, predictions: Dict, input_data: Dict):
        """Display prediction results"""
        print("\n" + "=" * 60)
        print("🎯 YIELD PREDICTION RESULTS")
        print("=" * 60)

        print("\n📊 Input Conditions:")
        for key, value in input_data.items():
            if isinstance(value, (int, float)):
                print(f"  {key:20s}: {value:.2f}")
            else:
                print(f"  {key:20s}: {value}")

        if not predictions:
            print("\n❌ No predictions available")
            return

        print(f"\n🌾 Predicted {self.target_column}:")
        print("-" * 60)

        max_pred = max(predictions.values()) if predictions else 1

        for model, prediction in predictions.items():
            bar_length = int((prediction / max_pred) * 40) if max_pred > 0 else 0
            bar = "█" * bar_length
            print(f"  {model:25s}: {prediction:8.2f} {bar}")

        # Best and worst predictions
        if len(predictions) > 1:
            best_model = max(predictions.items(), key=lambda x: x[1])
            worst_model = min(predictions.items(), key=lambda x: x[1])

            print(f"\n📈 Highest Prediction: {best_model[0]} = {best_model[1]:.2f}")
            print(f"📉 Lowest Prediction:  {worst_model[0]} = {worst_model[1]:.2f}")
            print(f"📊 Prediction Range: ±{abs(best_model[1] - worst_model[1])/2:.2f}")

        # Recommendations
        self._provide_recommendations(input_data, predictions)

    def _provide_recommendations(self, input_data: Dict, predictions: Dict):
        """Provide optimization recommendations"""
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 60)

        recommendations = []

        # NPK recommendations
        if all(k in input_data for k in ['N', 'P', 'K']):
            n_val = input_data['N']
            p_val = input_data['P']
            k_val = input_data['K']

            if n_val < 50:
                recommendations.append("⚠️  Nitrogen (N) is low. Consider increasing to 80-100")
            elif n_val > 120:
                recommendations.append("⚠️  Nitrogen (N) is high. Reduce to avoid nutrient burn")

            if p_val < 30:
                recommendations.append("⚠️  Phosphorus (P) is low. Consider increasing to 50-70")

            if k_val < 30:
                recommendations.append("⚠️  Potassium (K) is low. Consider increasing to 50-80")

        # Temperature recommendations
        if 'temperature' in input_data:
            temp = input_data['temperature']
            if temp < 15:
                recommendations.append("🌡️  Temperature is low. Consider using protective covering")
            elif temp > 35:
                recommendations.append("🌡️  Temperature is high. Ensure adequate irrigation")

        # Rainfall/humidity recommendations
        if 'rainfall' in input_data:
            rain = input_data['rainfall']
            if rain < 50:
                recommendations.append("💧 Rainfall is low. Supplementary irrigation recommended")
            elif rain > 250:
                recommendations.append("💧 Rainfall is high. Ensure proper drainage")

        # pH recommendations
        if 'ph' in input_data:
            ph = input_data['ph']
            if ph < 6.0:
                recommendations.append("🧪 Soil is acidic. Consider adding lime")
            elif ph > 8.0:
                recommendations.append("🧪 Soil is alkaline. Consider adding sulfur")

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("  ✅ Current conditions are optimal!")

    def save_model(self, filename: str = 'crop_yield_model.pkl'):
        """Save trained model"""
        if not self.is_trained:
            print("❌ No trained model to save")
            return

        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'column_info': self.column_info
        }

        try:
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✅ Model saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save model: {str(e)}")

    def run_interactive_session(self):
        """Run complete interactive prediction session"""
        # Load data
        if not self.load_and_prepare_data():
            print("❌ Failed to load data")
            return

        # Train models
        results = self.train_models()

        if not self.is_trained:
            print("❌ Training failed. Cannot proceed with predictions.")
            return

        # Interactive prediction loop
        while True:
            print("\n" + "=" * 60)
            choice = input("\n🌾 Enter '1' to make prediction, '2' to exit: ").strip()

            if choice == '1':
                try:
                    # Get user input
                    user_data = self.get_user_input()

                    # Make prediction
                    predictions = self.predict_yield(user_data)

                    # Display results
                    self.display_prediction(predictions, user_data)

                except KeyboardInterrupt:
                    print("\n\n⚠️  Prediction cancelled by user.")
                    continue
                except Exception as e:
                    print(f"\n❌ Prediction error: {str(e)}")
                    continue

            elif choice == '2':
                print("\n👋 Thank you for using Crop Yield Prediction System!")
                break
            else:
                print("⚠️  Invalid choice. Please enter 1 or 2")

        # Save model
        try:
            save_choice = input("\n💾 Save model? (y/n): ").strip().lower()
            if save_choice == 'y':
                self.save_model()
        except KeyboardInterrupt:
            print("\n\nExiting without saving...")

# Main execution
