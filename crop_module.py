# Auto-converted from notebook: /mnt/data/crop_prediction.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ============================================
# 1. DATA LOADING
# ============================================
def load_crop_data(file_path):
    """Load crop recommendation data from CSV file"""
    print("="*70)
    print(" "*15 + "LOADING CROP RECOMMENDATION DATA")
    print("="*70)

    try:
        df = pd.read_csv(file_path)
        print(f"\n✅ Successfully loaded data from file")
        print(f"📊 Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Display column information
        print("\n📋 Column Information:")
        print(f"   Total columns: {len(df.columns)}")

        # Categorize columns
        soil_cols = ['Soilcolor', 'Ph', 'K', 'P', 'N', 'Zn', 'S']
        weather_cols = [col for col in df.columns if any(x in col for x in ['QV2M', 'T2M', 'PRECTOTCORR'])]
        other_cols = ['WD10M', 'GWETTOP', 'CLOUD_AMT', 'WS2M_RANGE', 'PS']
        target_col = 'label'

        print(f"\n   🌱 Soil Properties ({len([c for c in soil_cols if c in df.columns])}):")
        for col in soil_cols:
            if col in df.columns:
                print(f"      • {col}")

        print(f"\n   🌤️  Weather Variables ({len([c for c in weather_cols if c in df.columns])}):")
        for col in weather_cols:
            if col in df.columns:
                season = col.split('-')[-1] if '-' in col else ''
                print(f"      • {col}")

        print(f"\n   🌍 Other Environmental Factors ({len([c for c in other_cols if c in df.columns])}):")
        for col in other_cols:
            if col in df.columns:
                print(f"      • {col}")

        print(f"\n   🎯 Target: {target_col}")

        return df

    except FileNotFoundError:
        print(f"\n❌ Error: File not found at {file_path}")
        print("   Please check the file path and try again.")
        return None

    except Exception as e:
        print(f"\n❌ Error loading data: {str(e)}")
        return None

# ============================================
# 2. DATA EXPLORATION
# ============================================
def explore_data(df):
    """Perform comprehensive exploratory data analysis"""
    print("\n" + "="*70)
    print(" "*20 + "DATA EXPLORATION")
    print("="*70)

    # Basic information
    print("\n📊 Dataset Overview:")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")

    # Display first few rows
    print("\n📝 Sample Data (First 5 rows):")
    print(df.head())

    # Missing values check
    print("\n🔍 Data Quality Check:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No missing values found!")
    else:
        print("   ⚠️  Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"      • {col}: {count} ({count/len(df)*100:.2f}%)")

    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\n   Duplicate rows: {duplicates}")

    # Target variable analysis
    print("\n🌾 Crop Distribution:")
    crop_counts = df['label'].value_counts()
    print(f"   Total unique crops: {df['label'].nunique()}")
    print("\n   Crop counts:")
    for crop, count in crop_counts.items():
        print(f"      • {crop}: {count} ({count/len(df)*100:.1f}%)")

    # Statistical summary for soil properties
    print("\n📈 Soil Properties - Statistical Summary:")
    soil_cols = ['Ph', 'K', 'P', 'N', 'Zn', 'S']
    soil_stats = df[soil_cols].describe()
    print(soil_stats.round(2))

    # Visualizations
    create_eda_visualizations(df)

    return df

def create_eda_visualizations(df):
    """Create comprehensive EDA visualizations"""

    # 1. Crop Distribution
    plt.figure(figsize=(14, 6))
    crop_counts = df['label'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(crop_counts)))
    bars = plt.bar(range(len(crop_counts)), crop_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    plt.xticks(range(len(crop_counts)), crop_counts.index, rotation=45, ha='right')
    plt.xlabel('Crop Type', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Distribution of Crops in Dataset', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('01_crop_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: 01_crop_distribution.png")

    # 2. Soil Properties Distribution
    soil_cols = ['Ph', 'K', 'P', 'N', 'Zn', 'S']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, col in enumerate(soil_cols):
        axes[idx].hist(df[col], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col} Distribution', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(col, fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].grid(alpha=0.3)

        # Add statistics text
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        axes[idx].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('02_soil_properties_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 02_soil_properties_distribution.png")

    # 3. Correlation Heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(16, 14))
    correlation = df[numeric_cols].corr()

    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=False, fmt='.2f',
                cmap='coolwarm', center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation"})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 03_correlation_heatmap.png")

    # 4. Soil Properties by Crop (Box plots)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, col in enumerate(soil_cols):
        df.boxplot(column=col, by='label', ax=axes[idx], patch_artist=True)
        axes[idx].set_title(f'{col} by Crop Type', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Crop', fontsize=10)
        axes[idx].set_ylabel(col, fontsize=10)
        plt.sca(axes[idx])
        plt.xticks(rotation=45, ha='right', fontsize=8)
        axes[idx].get_figure().suptitle('')  # Remove default title

    plt.tight_layout()
    plt.savefig('04_soil_properties_by_crop.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 04_soil_properties_by_crop.png")

    # 5. Soil Color Distribution
    if 'Soilcolor' in df.columns:
        plt.figure(figsize=(12, 6))
        soil_color_counts = df['Soilcolor'].value_counts()
        plt.barh(range(len(soil_color_counts)), soil_color_counts.values,
                color='sienna', edgecolor='black', alpha=0.7)
        plt.yticks(range(len(soil_color_counts)), soil_color_counts.index)
        plt.xlabel('Count', fontsize=12, fontweight='bold')
        plt.ylabel('Soil Color', fontsize=12, fontweight='bold')
        plt.title('Distribution of Soil Colors', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        for i, v in enumerate(soil_color_counts.values):
            plt.text(v + 5, i, str(v), va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('05_soil_color_distribution.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 05_soil_color_distribution.png")

# ============================================
# 3. DATA PREPROCESSING
# ============================================
def preprocess_data(df):
    """Preprocess the data for machine learning"""
    print("\n" + "="*70)
    print(" "*22 + "DATA PREPROCESSING")
    print("="*70)

    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']

    print(f"\n📊 Initial shapes:")
    print(f"   Features (X): {X.shape}")
    print(f"   Target (y): {y.shape}")

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    label_encoders = {}
    if len(categorical_cols) > 0:
        print(f"\n🔄 Encoding categorical features:")
        for col in categorical_cols:
            print(f"   • {col}: {X[col].nunique()} unique values")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Encode target variable
    print(f"\n🎯 Encoding target variable (label):")
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    print(f"   Unique crops: {len(le_target.classes_)}")
    for i, crop in enumerate(le_target.classes_):
        count = np.sum(y_encoded == i)
        print(f"   {i+1:2d}. {crop:20s}: {count:4d} samples")

    # Split the data
    print(f"\n✂️  Splitting data (80% train, 20% test):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"   Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"   Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    # Feature scaling
    print(f"\n⚖️  Applying Standard Scaling to features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("   ✅ Scaling completed")
    print("\n✅ Preprocessing completed successfully!")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le_target, X.columns, label_encoders

# ============================================
# 4. MODEL TRAINING
# ============================================
def train_models(X_train, y_train):
    """Train multiple ML models"""
    print("\n" + "="*70)
    print(" "*22 + "MODEL TRAINING")
    print("="*70)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42)
    }

    trained_models = {}

    print(f"\n🤖 Training {len(models)} machine learning models...\n")

    for name, model in models.items():
        print(f"{'─'*70}")
        print(f"🔄 Training: {name}")

        # Train model
        model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

        trained_models[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }

        print(f"   ✅ Training completed")
        print(f"   📊 Cross-validation scores: {cv_scores}")
        print(f"   📈 Mean Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    print(f"\n{'─'*70}")
    print("✅ All models trained successfully!")

    return trained_models

# ============================================
# 5. MODEL EVALUATION
# ============================================
def evaluate_models(trained_models, X_test, y_test, le):
    """Evaluate all trained models"""
    print("\n" + "="*70)
    print(" "*22 + "MODEL EVALUATION")
    print("="*70)

    results = []

    for name, model_dict in trained_models.items():
        model = model_dict['model']

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'CV Mean': model_dict['cv_mean'],
            'CV Std': model_dict['cv_std']
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    print("\n" + "="*70)
    print(" "*18 + "MODEL COMPARISON SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))

    # Find best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]['model']

    print(f"\n{'='*70}")
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"{'='*70}")
    print(f"   Accuracy:  {results_df.iloc[0]['Accuracy']:.4f}")
    print(f"   Precision: {results_df.iloc[0]['Precision']:.4f}")
    print(f"   Recall:    {results_df.iloc[0]['Recall']:.4f}")
    print(f"   F1-Score:  {results_df.iloc[0]['F1-Score']:.4f}")

    # Detailed classification report for best model
    y_pred_best = best_model.predict(X_test)
    print(f"\n📋 Detailed Classification Report - {best_model_name}:")
    print("="*70)
    print(classification_report(y_test, y_pred_best, target_names=le.classes_, zero_division=0))

    # Create visualizations
    create_evaluation_visualizations(results_df, best_model, best_model_name, X_test, y_test, le)

    return results_df, best_model_name, best_model

def create_evaluation_visualizations(results_df, best_model, best_model_name, X_test, y_test, le):
    """Create evaluation visualizations"""

    # 1. Model Comparison
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2, idx % 2]
        bars = ax.barh(results_df['Model'], results_df[metric], color=color, edgecolor='black', alpha=0.8)
        ax.set_xlabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, results_df[metric])):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('06_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: 06_model_comparison.png")

    # 2. Confusion Matrix
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=le.classes_, yticklabels=le.classes_,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Crop', fontsize=13, fontweight='bold')
    plt.ylabel('Actual Crop', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('07_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 07_confusion_matrix.png")

# ============================================
# 6. FEATURE IMPORTANCE
# ============================================
def analyze_feature_importance(best_model, best_model_name, feature_names):
    """Analyze and visualize feature importance"""
    print("\n" + "="*70)
    print(" "*18 + "FEATURE IMPORTANCE ANALYSIS")
    print("="*70)

    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print(f"\n📊 Top 15 Most Important Features ({best_model_name}):\n")
        for i, idx in enumerate(indices[:15], 1):
            print(f"   {i:2d}. {feature_names[idx]:25s}: {importances[idx]:.4f}")

        # Visualization
        plt.figure(figsize=(14, 10))
        top_n = min(20, len(importances))
        top_indices = indices[:top_n]

        plt.barh(range(top_n), importances[top_indices],
                color='steelblue', edgecolor='black', alpha=0.8)
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Feature Importance - {best_model_name}',
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, v in enumerate(importances[top_indices]):
            plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig('08_feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✅ Saved: 08_feature_importance.png")
    else:
        print(f"\n⚠️  {best_model_name} does not support feature importance analysis")

# ============================================
# 7. PREDICTION DEMO
# ============================================
def prediction_demo(best_model, scaler, le, feature_names, X_test, y_test):
    """Demonstrate crop prediction with sample data"""
    print("\n" + "="*70)
    print(" "*20 + "PREDICTION DEMONSTRATION")
    print("="*70)

    # Random sample from test set
    n_samples = 3
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

    for i, idx in enumerate(sample_indices, 1):
        print(f"\n{'─'*70}")
        print(f"📋 Sample #{i}:")
        print(f"{'─'*70}")

        sample = X_test[idx:idx+1]
        actual_crop = le.inverse_transform([y_test[idx]])[0]

        # Show key features
        print("\n   Key Input Features:")
        key_features = ['Ph', 'K', 'P', 'N', 'Zn', 'S']
        for feat in key_features:
            if feat in feature_names:
                feat_idx = list(feature_names).index(feat)
                print(f"      • {feat:10s}: {sample[0][feat_idx]:8.2f}")

        # Predict
        prediction = best_model.predict(sample)
        predicted_crop = le.inverse_transform(prediction)[0]

        print(f"\n   ✅ Actual Crop:    {actual_crop}")
        print(f"   🔮 Predicted Crop: {predicted_crop}")

        # Probabilities
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(sample)[0]
            top_n = min(5, len(probabilities))
            top_idx = np.argsort(probabilities)[::-1][:top_n]

            print(f"\n   📊 Top {top_n} Predictions:")
            for j, crop_idx in enumerate(top_idx, 1):
                crop = le.classes_[crop_idx]
                prob = probabilities[crop_idx] * 100
                marker = "✓" if crop == predicted_crop else " "
                print(f"      {marker} {j}. {crop:20s}: {prob:5.2f}%")

        # Result
        if predicted_crop == actual_crop:
            print(f"\n   🎯 Result: ✅ CORRECT PREDICTION!")
        else:
            print(f"\n   🎯 Result: ❌ Incorrect prediction")

# ============================================
# 8. USER INPUT PREDICTION
# ============================================
def predict_from_user_input(best_model, scaler, le, feature_names, categorical_encoders, df_original):
    """Get user input and predict crop recommendation"""
    print("\n" + "="*70)
    print(" "*15 + "🌾 CROP RECOMMENDATION SYSTEM 🌾")
    print("="*70)
    print("\nEnter the following details to get crop recommendation:\n")

    user_input = {}

    try:
        # Get soil color
        print("─"*70)
        print("SOIL PROPERTIES")
        print("─"*70)

        if 'Soilcolor' in df_original.columns:
            unique_colors = df_original['Soilcolor'].unique()
            print(f"\nAvailable Soil Colors: {', '.join(unique_colors)}")
            soil_color = input("Enter Soil Color: ").strip()
            user_input['Soilcolor'] = soil_color

        # Get soil properties
        user_input['Ph'] = float(input("Enter pH value (e.g., 5.5 to 8.5): "))
        user_input['K'] = float(input("Enter Potassium (K) content (e.g., 100 to 800): "))
        user_input['P'] = float(input("Enter Phosphorus (P) content (e.g., 5 to 150): "))
        user_input['N'] = float(input("Enter Nitrogen (N) content (e.g., 0 to 150): "))
        user_input['Zn'] = float(input("Enter Zinc (Zn) content (e.g., 0 to 10): "))
        user_input['S'] = float(input("Enter Sulfur (S) content (e.g., 10 to 50): "))

        # Get weather data for each season
        print("\n" + "─"*70)
        print("WEATHER DATA (Enter data for each season)")
        print("─"*70)

        seasons = ['W', 'Sp', 'Su', 'Au']  # Winter, Spring, Summer, Autumn
        season_names = {'W': 'Winter', 'Sp': 'Spring', 'Su': 'Summer', 'Au': 'Autumn'}

        for season in seasons:
            print(f"\n{season_names[season]}:")
            user_input[f'QV2M-{season}'] = float(input(f"  Humidity (QV2M-{season}, g/kg, e.g., 5-20): "))
            user_input[f'T2M_MAX-{season}'] = float(input(f"  Max Temperature (T2M_MAX-{season}, °C, e.g., 15-45): "))
            user_input[f'T2M_MIN-{season}'] = float(input(f"  Min Temperature (T2M_MIN-{season}, °C, e.g., 0-30): "))
            user_input[f'PRECTOTCORR-{season}'] = float(input(f"  Rainfall (PRECTOTCORR-{season}, mm, e.g., 0-300): "))

        # Get other environmental factors
        print("\n" + "─"*70)
        print("OTHER ENVIRONMENTAL FACTORS")
        print("─"*70)

        user_input['WD10M'] = float(input("Enter Wind Direction at 10m (WD10M, degrees, 0-360): "))
        user_input['GWETTOP'] = float(input("Enter Surface Soil Wetness (GWETTOP, 0-1): "))
        user_input['CLOUD_AMT'] = float(input("Enter Cloud Amount (CLOUD_AMT, 0-100%): "))
        user_input['WS2M_RANGE'] = float(input("Enter Wind Speed Range at 2m (WS2M_RANGE, m/s, 0-10): "))
        user_input['PS'] = float(input("Enter Surface Pressure (PS, kPa, 85-105): "))

        # Create DataFrame from user input
        input_df = pd.DataFrame([user_input])

        # Ensure all columns are in the correct order
        missing_cols = set(feature_names) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0  # Add missing columns with default value

        input_df = input_df[feature_names]  # Reorder columns to match training data

        # Encode categorical features if any
        if 'Soilcolor' in input_df.columns and 'Soilcolor' in categorical_encoders:
            try:
                input_df['Soilcolor'] = categorical_encoders['Soilcolor'].transform([user_input['Soilcolor']])[0]
            except:
                print(f"\n⚠️  Warning: Unknown soil color. Using default encoding.")
                input_df['Soilcolor'] = 0

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = best_model.predict(input_scaled)
        predicted_crop = le.inverse_transform(prediction)[0]

        # Display results
        print("\n" + "="*70)
        print(" "*20 + "🎯 PREDICTION RESULTS 🎯")
        print("="*70)

        print(f"\n✅ RECOMMENDED CROP: {predicted_crop}")

        # Show probabilities if available
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(input_scaled)[0]
            top_n = min(5, len(probabilities))
            top_idx = np.argsort(probabilities)[::-1][:top_n]

            print(f"\n📊 Top {top_n} Crop Recommendations (with confidence):\n")
            for i, crop_idx in enumerate(top_idx, 1):
                crop = le.classes_[crop_idx]
                prob = probabilities[crop_idx] * 100
                bar_length = int(prob / 2)
                bar = "█" * bar_length
                print(f"   {i}. {crop:20s} [{bar:<50}] {prob:5.2f}%")

        print("\n" + "="*70)

        # Ask if user wants to make another prediction
        print("\n")
        another = input("Would you like to make another prediction? (yes/no): ").strip().lower()
        if another in ['yes', 'y']:
            predict_from_user_input(best_model, scaler, le, feature_names, categorical_encoders, df_original)

    except ValueError as e:
        print(f"\n❌ Error: Invalid input. Please enter numeric values where required.")
        print(f"   Details: {str(e)}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Prediction cancelled by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")

# ============================================
# 8. MAIN EXECUTION
# ============================================
def main():
    print("\n" + "="*70)
    print(" "*10 + "CROP RECOMMENDATION SYSTEM USING MACHINE LEARNING")
    print("="*70)

    # File path
    file_path = r"D:\New folder\Desktop\agri\Crop Recommendation using Soil Properties and Weather Prediction.csv"

    # Step 1: Load Data
    print("\n🔄 STEP 1: Loading Dataset...")
    df = load_crop_data(file_path)

    if df is None:
        return

    # Step 2: Explore Data
    print("\n🔄 STEP 2: Exploratory Data Analysis...")
    df_explored = explore_data(df)

    # Step 3: Preprocess Data
    print("\n🔄 STEP 3: Data Preprocessing...")
    X_train, X_test, y_train, y_test, scaler, le, feature_names, categorical_encoders = preprocess_data(df)

    # Step 4: Train Models
    print("\n🔄 STEP 4: Training Machine Learning Models...")
    trained_models = train_models(X_train, y_train)

    # Step 5: Evaluate Models
    print("\n🔄 STEP 5: Evaluating Models...")
    results_df, best_model_name, best_model = evaluate_models(trained_models, X_test, y_test, le)

    # Step 6: Feature Importance
    print("\n🔄 STEP 6: Analyzing Feature Importance...")
    analyze_feature_importance(best_model, best_model_name, feature_names)

    # Step 7: Prediction Demo
    print("\n🔄 STEP 7: Prediction Demonstration...")
    prediction_demo(best_model, scaler, le, feature_names, X_test, y_test)

    # Summary
    print("\n" + "="*70)
    print(" "*25 + "✅ TRAINING COMPLETED!")
    print("="*70)

    print("\n📁 Generated Files:")
    files = [
        "01_crop_distribution.png",
        "02_soil_properties_distribution.png",
        "03_correlation_heatmap.png",
        "04_soil_properties_by_crop.png",
        "05_soil_color_distribution.png",
        "06_model_comparison.png",
        "07_confusion_matrix.png",
        "08_feature_importance.png"
    ]

    for i, file in enumerate(files, 1):
        print(f"   {i}. ✓ {file}")

    print("\n" + "="*70)
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📊 Test Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
    print(f"📈 Total Crops: {len(le.classes_)}")
    print(f"📋 Total Features: {len(feature_names)}")
    print("="*70)

    # Step 8: User Input Prediction
    print("\n" + "="*70)
    print(" "*15 + "🔄 STEP 8: USER INPUT PREDICTION")
    print("="*70)

    user_choice = input("\nWould you like to predict a crop based on your input? (yes/no): ").strip().lower()

    if user_choice in ['yes', 'y']:
        predict_from_user_input(best_model, scaler, le, feature_names, categorical_encoders, df)
    else:
        print("\n✅ System ready for predictions. You can call predict_from_user_input() anytime!")

    print("\n" + "="*70)
    print(" "*20 + "🌾 THANK YOU FOR USING 🌾")
    print("="*70)

    # Return objects for future use
    return best_model, scaler, le, feature_names, categorical_encoders, df

