# Auto-converted from notebook: /mnt/data/irrigation.ipynb

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. FERTILIZER OPTIMIZATION MODEL
# ============================================================================

class FertilizerOptimizationModel:
    """Predicts optimal N-P-K fertilizer ratios"""
    
    def __init__(self):
        self.n_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.p_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.k_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.crop_encoder = LabelEncoder()
        
    def generate_training_data(self, n_samples=1000):
        """Generate synthetic training data"""
        np.random.seed(42)
        
        data = {
            'soil_N': np.random.uniform(20, 100, n_samples),
            'soil_P': np.random.uniform(10, 80, n_samples),
            'soil_K': np.random.uniform(15, 90, n_samples),
            'pH': np.random.uniform(5.5, 8.0, n_samples),
            'organic_matter': np.random.uniform(1.0, 5.0, n_samples),
            'soil_moisture': np.random.uniform(20, 80, n_samples),
            'previous_yield': np.random.uniform(2.0, 6.0, n_samples),
            'crop_type': np.random.choice(['wheat', 'rice', 'maize', 'cotton'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        crop_requirements = {
            'wheat': {'N': 120, 'P': 60, 'K': 40},
            'rice': {'N': 100, 'P': 50, 'K': 50},
            'maize': {'N': 150, 'P': 70, 'K': 60},
            'cotton': {'N': 130, 'P': 65, 'K': 55}
        }
        
        df['required_N'] = df.apply(
            lambda row: max(0, crop_requirements[row['crop_type']]['N'] - row['soil_N']) * 
            (row['previous_yield'] / 4.0) * (1.1 if abs(row['pH'] - 6.5) > 0.5 else 1.0),
            axis=1
        )
        
        df['required_P'] = df.apply(
            lambda row: max(0, crop_requirements[row['crop_type']]['P'] - row['soil_P']) * 
            (row['previous_yield'] / 4.0) * (1.15 if row['organic_matter'] < 2.5 else 1.0),
            axis=1
        )
        
        df['required_K'] = df.apply(
            lambda row: max(0, crop_requirements[row['crop_type']]['K'] - row['soil_K']) * 
            (row['previous_yield'] / 4.0) * (1.1 if row['soil_moisture'] < 40 else 1.0),
            axis=1
        )
        
        return df
    
    def train(self):
        """Train the models"""
        df = self.generate_training_data(1000)
        df['crop_encoded'] = self.crop_encoder.fit_transform(df['crop_type'])
        
        X = df[['soil_N', 'soil_P', 'soil_K', 'pH', 'organic_matter', 
                'soil_moisture', 'previous_yield', 'crop_encoded']]
        X_scaled = self.scaler.fit_transform(X)
        
        self.n_model.fit(X_scaled, df['required_N'])
        self.p_model.fit(X_scaled, df['required_P'])
        self.k_model.fit(X_scaled, df['required_K'])
        
    def predict(self, soil_N, soil_P, soil_K, pH, organic_matter, 
                soil_moisture, previous_yield, crop_type):
        """Predict fertilizer requirements"""
        
        crop_encoded = self.crop_encoder.transform([crop_type])[0]
        features = np.array([[soil_N, soil_P, soil_K, pH, organic_matter, 
                             soil_moisture, previous_yield, crop_encoded]])
        features_scaled = self.scaler.transform(features)
        
        n_required = max(0, self.n_model.predict(features_scaled)[0])
        p_required = max(0, self.p_model.predict(features_scaled)[0])
        k_required = max(0, self.k_model.predict(features_scaled)[0])
        
        organic_recommendation = organic_matter < 3.0
        
        return {
            'N_kg_per_ha': round(n_required, 1),
            'P_kg_per_ha': round(p_required, 1),
            'K_kg_per_ha': round(k_required, 1),
            'NPK_ratio': f"{round(n_required)}:{round(p_required)}:{round(k_required)}",
            'recommend_organic': organic_recommendation,
            'estimated_cost_INR': round((n_required * 20 + p_required * 30 + k_required * 25)),
            'organic_alternatives': [
                'Vermicompost (NPK: 1.5-1.0-1.5)',
                'Cow Manure (NPK: 0.5-0.2-0.5)',
                'Neem Cake (NPK: 5.0-1.0-1.0)',
                'Bone Meal (NPK: 3.0-15.0-0)'
            ] if organic_recommendation else []
        }


# ============================================================================
# 2. IRRIGATION SCHEDULING MODEL
# ============================================================================

class IrrigationSchedulingModel:
    """Recommends irrigation timing and water amount"""
    
    def __init__(self):
        self.water_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.schedule_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.crop_encoder = LabelEncoder()
        
    def generate_training_data(self, n_samples=1000):
        """Generate synthetic training data"""
        np.random.seed(42)
        
        data = {
            'soil_moisture': np.random.uniform(15, 85, n_samples),
            'temperature': np.random.uniform(15, 40, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'rainfall_forecast': np.random.uniform(0, 50, n_samples),
            'wind_speed': np.random.uniform(0, 20, n_samples),
            'crop_type': np.random.choice(['wheat', 'rice', 'maize', 'cotton'], n_samples),
            'growth_stage': np.random.choice(['vegetative', 'flowering', 'maturity'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        optimal_moisture = {'wheat': 70, 'rice': 85, 'maize': 75, 'cotton': 65}
        
        df['water_deficit'] = df.apply(
            lambda row: max(0, optimal_moisture[row['crop_type']] - row['soil_moisture']),
            axis=1
        )
        
        df['ET'] = (df['temperature'] * 0.5 + (100 - df['humidity']) * 0.3 - 
                   df['wind_speed'] * 0.2)
        
        df['water_amount'] = (df['water_deficit'] * 2.5 + df['ET'] - 
                             df['rainfall_forecast'] * 0.8).clip(lower=0)
        
        df['irrigation_schedule'] = df.apply(
            lambda row: 24 if row['soil_moisture'] < 40 
            else 48 if row['soil_moisture'] < 60 
            else 72,
            axis=1
        )
        
        return df
    
    def train(self):
        """Train the models"""
        df = self.generate_training_data(1000)
        df['crop_encoded'] = self.crop_encoder.fit_transform(df['crop_type'])
        
        growth_map = {'vegetative': 0, 'flowering': 1, 'maturity': 2}
        df['growth_encoded'] = df['growth_stage'].map(growth_map)
        
        X = df[['soil_moisture', 'temperature', 'humidity', 'rainfall_forecast',
               'wind_speed', 'crop_encoded', 'growth_encoded']]
        X_scaled = self.scaler.fit_transform(X)
        
        self.water_model.fit(X_scaled, df['water_amount'])
        self.schedule_model.fit(X_scaled, df['irrigation_schedule'])
        
    def predict(self, soil_moisture, temperature, humidity, rainfall_forecast,
               wind_speed, crop_type, growth_stage='vegetative'):
        """Predict irrigation requirements"""
        
        crop_encoded = self.crop_encoder.transform([crop_type])[0]
        growth_map = {'vegetative': 0, 'flowering': 1, 'maturity': 2}
        growth_encoded = growth_map.get(growth_stage, 0)
        
        features = np.array([[soil_moisture, temperature, humidity, 
                            rainfall_forecast, wind_speed, crop_encoded, growth_encoded]])
        features_scaled = self.scaler.transform(features)
        
        water_amount = max(0, self.water_model.predict(features_scaled)[0])
        schedule_hours = max(24, self.schedule_model.predict(features_scaled)[0])
        
        if soil_moisture < 40:
            urgency = 'HIGH'
        elif soil_moisture < 60:
            urgency = 'MEDIUM'
        else:
            urgency = 'LOW'
        
        if water_amount > 30:
            method = 'Drip Irrigation'
        elif water_amount > 15:
            method = 'Sprinkler Irrigation'
        else:
            method = 'Light Irrigation'
        
        return {
            'water_amount_mm': round(water_amount, 1),
            'water_amount_liters_per_m2': round(water_amount, 1),
            'next_irrigation_hours': round(schedule_hours),
            'next_irrigation_days': round(schedule_hours / 24, 1),
            'urgency': urgency,
            'recommended_method': method,
            'water_saved_from_rainfall_mm': round(rainfall_forecast * 0.8, 1),
            'total_water_savings_percent': round((rainfall_forecast * 0.8 / max(water_amount, 1)) * 100, 1)
        }


# ============================================================================
# 3. CROP ROTATION RECOMMENDATION MODEL
# ============================================================================

class CropRotationModel:
    """Suggests optimal next-season crops"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.crop_encoder = LabelEncoder()
        self.target_encoder = LabelEncoder()
        
    def generate_training_data(self, n_samples=1000):
        """Generate synthetic training data"""
        np.random.seed(42)
        
        data = {
            'previous_crop': np.random.choice(['wheat', 'rice', 'maize', 'cotton', 
                                              'soybean', 'chickpea'], n_samples),
            'soil_N': np.random.uniform(30, 100, n_samples),
            'soil_P': np.random.uniform(20, 80, n_samples),
            'soil_K': np.random.uniform(25, 90, n_samples),
            'soil_health_score': np.random.uniform(40, 95, n_samples),
            'previous_yield': np.random.uniform(2.0, 6.0, n_samples),
            'seasons_same_crop': np.random.randint(1, 5, n_samples),
            'organic_matter': np.random.uniform(1.5, 5.0, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        rotation_rules = {
            'wheat': ['soybean', 'chickpea', 'soybean'],
            'rice': ['wheat', 'maize', 'chickpea'],
            'maize': ['wheat', 'soybean', 'chickpea'],
            'cotton': ['wheat', 'soybean', 'chickpea'],
            'soybean': ['wheat', 'rice', 'maize'],
            'chickpea': ['wheat', 'rice', 'cotton']
        }
        
        df['next_crop'] = df.apply(
            lambda row: np.random.choice(rotation_rules[row['previous_crop']]),
            axis=1
        )
        
        df.loc[(df['soil_N'] < 50) & (df['previous_crop'].isin(['wheat', 'rice', 'maize'])), 
               'next_crop'] = 'soybean'
        
        return df
    
    def train(self):
        """Train the model"""
        df = self.generate_training_data(1000)
        df['prev_crop_encoded'] = self.crop_encoder.fit_transform(df['previous_crop'])
        df['next_crop_encoded'] = self.target_encoder.fit_transform(df['next_crop'])
        
        X = df[['prev_crop_encoded', 'soil_N', 'soil_P', 'soil_K', 
               'soil_health_score', 'previous_yield', 'seasons_same_crop', 
               'organic_matter']]
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, df['next_crop_encoded'])
        
    def predict(self, previous_crop, soil_N, soil_P, soil_K, soil_health_score,
               previous_yield, seasons_same_crop, organic_matter):
        """Predict optimal next crop"""
        
        prev_crop_encoded = self.crop_encoder.transform([previous_crop])[0]
        features = np.array([[prev_crop_encoded, soil_N, soil_P, soil_K,
                            soil_health_score, previous_yield, seasons_same_crop,
                            organic_matter]])
        features_scaled = self.scaler.transform(features)
        
        probabilities = self.model.predict_proba(features_scaled)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_crops = self.target_encoder.inverse_transform(top_3_indices)
        top_3_probs = probabilities[top_3_indices]
        
        rotation_benefits = {
            'wheat': 'Restores soil structure, reduces pest pressure',
            'rice': 'Breaks disease cycles, manages weeds effectively',
            'maize': 'Deep roots improve soil structure',
            'cotton': 'Long growing season, different nutrient pattern',
            'soybean': 'Fixes 30-40 kg N/ha, improves soil fertility',
            'chickpea': 'Fixes 40-50 kg N/ha, drought resistant'
        }
        
        n_improvement = 40 if top_3_crops[0] in ['soybean', 'chickpea'] else 0
        yield_increase = 15 if seasons_same_crop >= 2 else 10
        cost_reduction = 30 if top_3_crops[0] in ['soybean', 'chickpea'] else 20
        
        return {
            'recommended_crops': [
                {
                    'crop': crop,
                    'confidence': round(prob * 100, 1),
                    'benefit': rotation_benefits.get(crop, 'Diversifies cropping system')
                }
                for crop, prob in zip(top_3_crops, top_3_probs)
            ],
            'nitrogen_restoration_kg_per_ha': n_improvement,
            'expected_yield_increase_percent': yield_increase,
            'cost_reduction_percent': cost_reduction,
            'sustainability_score': round(soil_health_score + n_improvement * 0.5),
            'rotation_urgency': 'HIGH' if seasons_same_crop >= 3 else 'MEDIUM' if seasons_same_crop >= 2 else 'LOW',
            'key_recommendation': f"Rotate from {previous_crop} to {top_3_crops[0]} to restore soil health"
        }


# ============================================================================
# INTERACTIVE USER INPUT SYSTEM
# ============================================================================

def display_banner():
    print("\n" + "="*80)
    print(" "*20 + "🌾 SMART AGRICULTURE ML PREDICTION SYSTEM 🌾")
    print("="*80)
    print("Get AI-powered recommendations for:")
    print("  1. Fertilizer Optimization (N-P-K)")
    print("  2. Irrigation Scheduling")
    print("  3. Crop Rotation Planning")
    print("="*80 + "\n")


def get_user_input(prompt, input_type=float, valid_range=None, options=None):
    """Helper function to get validated user input"""
    while True:
        try:
            if options:
                print(f"\n{prompt}")
                for i, option in enumerate(options, 1):
                    print(f"  {i}. {option}")
                choice = int(input("Enter choice (number): "))
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                else:
                    print(f"❌ Please enter a number between 1 and {len(options)}")
            else:
                value = input_type(input(f"{prompt}: "))
                if valid_range:
                    min_val, max_val = valid_range
                    if min_val <= value <= max_val:
                        return value
                    else:
                        print(f"❌ Please enter a value between {min_val} and {max_val}")
                else:
                    return value
        except ValueError:
            print("❌ Invalid input. Please try again.")


def fertilizer_prediction_interface(model):
    """Interactive interface for fertilizer prediction"""
    print("\n" + "="*80)
    print("🌱 FERTILIZER OPTIMIZATION")
    print("="*80)
    
    soil_N = get_user_input("Soil Nitrogen (N) level [ppm] (20-120)", float, (20, 120))
    soil_P = get_user_input("Soil Phosphorus (P) level [ppm] (10-100)", float, (10, 100))
    soil_K = get_user_input("Soil Potassium (K) level [ppm] (15-120)", float, (15, 120))
    pH = get_user_input("Soil pH (5.0-8.5)", float, (5.0, 8.5))
    organic_matter = get_user_input("Organic Matter [%] (1.0-6.0)", float, (1.0, 6.0))
    soil_moisture = get_user_input("Soil Moisture [%] (20-90)", float, (20, 90))
    previous_yield = get_user_input("Previous Yield [ton/ha] (1.0-8.0)", float, (1.0, 8.0))
    crop_type = get_user_input(
        "Select Crop Type",
        str,
        options=['wheat', 'rice', 'maize', 'cotton']
    )
    
    print("\n⏳ Analyzing soil data and generating recommendations...")
    result = model.predict(soil_N, soil_P, soil_K, pH, organic_matter, 
                          soil_moisture, previous_yield, crop_type)
    
    print("\n" + "─"*80)
    print("📊 FERTILIZER RECOMMENDATIONS")
    print("─"*80)
    print(f"✓ Recommended NPK Ratio: {result['NPK_ratio']}")
    print(f"  • Nitrogen (N):    {result['N_kg_per_ha']} kg/ha")
    print(f"  • Phosphorus (P):  {result['P_kg_per_ha']} kg/ha")
    print(f"  • Potassium (K):   {result['K_kg_per_ha']} kg/ha")
    print(f"\n💰 Estimated Cost: ₹{result['estimated_cost_INR']}/ha")
    
    if result['recommend_organic']:
        print(f"\n💡 ORGANIC ALTERNATIVES RECOMMENDED (Organic matter is low)")
        print("   Consider these cost-effective options:")
        for alt in result['organic_alternatives']:
            print(f"   • {alt}")
        print("   Benefits: Reduces cost by 30-40%, improves soil health")
    
    print("─"*80)


def irrigation_prediction_interface(model):
    """Interactive interface for irrigation prediction"""
    print("\n" + "="*80)
    print("💧 IRRIGATION SCHEDULING")
    print("="*80)
    
    soil_moisture = get_user_input("Current Soil Moisture [%] (10-90)", float, (10, 90))
    temperature = get_user_input("Temperature [°C] (10-45)", float, (10, 45))
    humidity = get_user_input("Humidity [%] (20-100)", float, (20, 100))
    rainfall_forecast = get_user_input("Expected Rainfall [mm] (0-100)", float, (0, 100))
    wind_speed = get_user_input("Wind Speed [km/h] (0-30)", float, (0, 30))
    crop_type = get_user_input(
        "Select Crop Type",
        str,
        options=['wheat', 'rice', 'maize', 'cotton']
    )
    growth_stage = get_user_input(
        "Select Growth Stage",
        str,
        options=['vegetative', 'flowering', 'maturity']
    )
    
    print("\n⏳ Calculating irrigation requirements...")
    result = model.predict(soil_moisture, temperature, humidity, rainfall_forecast,
                          wind_speed, crop_type, growth_stage)
    
    print("\n" + "─"*80)
    print("📊 IRRIGATION RECOMMENDATIONS")
    print("─"*80)
    print(f"💧 Water Amount Required: {result['water_amount_mm']} mm")
    print(f"                          ({result['water_amount_liters_per_m2']} liters per m²)")
    print(f"\n⏰ Next Irrigation: {result['next_irrigation_hours']} hours ({result['next_irrigation_days']} days)")
    print(f"🚨 Priority Level: {result['urgency']}")
    print(f"🔧 Recommended Method: {result['recommended_method']}")
    
    if rainfall_forecast > 0:
        print(f"\n🌧️ RAINFALL ADJUSTMENT:")
        print(f"   Water Saved: {result['water_saved_from_rainfall_mm']} mm")
        print(f"   Total Savings: {result['total_water_savings_percent']}%")
        print(f"   ✓ Recommendation adjusted to prevent over-watering")
    
    print("─"*80)


def crop_rotation_interface(model):
    """Interactive interface for crop rotation prediction"""
    print("\n" + "="*80)
    print("🔄 CROP ROTATION PLANNING")
    print("="*80)
    
    previous_crop = get_user_input(
        "Previous Crop Grown",
        str,
        options=['wheat', 'rice', 'maize', 'cotton', 'soybean', 'chickpea']
    )
    soil_N = get_user_input("Soil Nitrogen (N) [ppm] (20-120)", float, (20, 120))
    soil_P = get_user_input("Soil Phosphorus (P) [ppm] (10-100)", float, (10, 100))
    soil_K = get_user_input("Soil Potassium (K) [ppm] (15-120)", float, (15, 120))
    soil_health_score = get_user_input("Soil Health Score (40-100)", float, (40, 100))
    previous_yield = get_user_input("Previous Yield [ton/ha] (1.0-8.0)", float, (1.0, 8.0))
    seasons_same_crop = int(get_user_input("Seasons Growing Same Crop (1-6)", float, (1, 6)))
    organic_matter = get_user_input("Organic Matter [%] (1.0-6.0)", float, (1.0, 6.0))
    
    print("\n⏳ Analyzing crop history and soil health...")
    result = model.predict(previous_crop, soil_N, soil_P, soil_K, soil_health_score,
                          previous_yield, seasons_same_crop, organic_matter)
    
    print("\n" + "─"*80)
    print("📊 CROP ROTATION RECOMMENDATIONS")
    print("─"*80)
    print(f"🎯 Key Recommendation: {result['key_recommendation']}")
    print(f"\n🌱 TOP RECOMMENDED CROPS:")
    for i, crop_info in enumerate(result['recommended_crops'], 1):
        print(f"\n   {i}. {crop_info['crop'].upper()} (Confidence: {crop_info['confidence']}%)")
        print(f"      └─ Benefit: {crop_info['benefit']}")
    
    print(f"\n📈 EXPECTED BENEFITS:")
    print(f"   • Nitrogen Restoration: +{result['nitrogen_restoration_kg_per_ha']} kg/ha")
    print(f"   • Yield Increase: +{result['expected_yield_increase_percent']}%")
    print(f"   • Cost Reduction: -{result['cost_reduction_percent']}%")
    print(f"   • Sustainability Score: {result['sustainability_score']}/100")
    
    print(f"\n⚠️  Rotation Urgency: {result['rotation_urgency']}")
    if seasons_same_crop >= 3:
        print(f"   WARNING: You've grown the same crop for {seasons_same_crop} seasons!")
        print(f"   Immediate rotation is critical to prevent soil depletion.")
    
    print("─"*80)


def main():
    """Main interactive system"""
    display_banner()
    
    print("⏳ Initializing ML models...")
    fert_model = FertilizerOptimizationModel()
    fert_model.train()
    print("✓ Fertilizer model ready")
    
    irr_model = IrrigationSchedulingModel()
    irr_model.train()
    print("✓ Irrigation model ready")
    
    rotation_model = CropRotationModel()
    rotation_model.train()
    print("✓ Crop rotation model ready")
    
    print("\n✅ All models trained and ready!")
    
    while True:
        print("\n" + "="*80)
        print("MAIN MENU")
        print("="*80)
        choice = get_user_input(
            "Select Prediction Type",
            str,
            options=[
                'Fertilizer Optimization',
                'Irrigation Scheduling',
                'Crop Rotation Planning',
                'Exit'
            ]
        )
        
        if choice == 'Fertilizer Optimization':
            fertilizer_prediction_interface(fert_model)
        elif choice == 'Irrigation Scheduling':
            irrigation_prediction_interface(irr_model)
        elif choice == 'Crop Rotation Planning':
            crop_rotation_interface(rotation_model)
        elif choice == 'Exit':
            print("\n" + "="*80)
            print("Thank you for using Smart Agriculture ML System! 🌾")
            print("="*80)
            break
        
        input("\n📝 Press Enter to continue...")


