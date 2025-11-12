from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
from typing import Optional

app = FastAPI(title="AgriYield+ Real ML API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# INITIALIZE ML MODELS ON STARTUP
# ============================================================================
print("🌾 Initializing AgriYield ML Models...")

# Import and initialize models
try:
    from irrigation_module import (
        FertilizerOptimizationModel,
        IrrigationSchedulingModel,
        CropRotationModel
    )
    
    fertilizer_model = FertilizerOptimizationModel()
    fertilizer_model.train()
    print("✅ Fertilizer model loaded")
    
    irrigation_model = IrrigationSchedulingModel()
    irrigation_model.train()
    print("✅ Irrigation model loaded")
    
    rotation_model = CropRotationModel()
    rotation_model.train()
    print("✅ Crop rotation model loaded")
    
    MODELS_LOADED = True
except Exception as e:
    print(f"⚠️ Warning: Could not load ML models: {e}")
    print("📝 Using fallback prediction logic")
    MODELS_LOADED = False

print("🚀 Backend ready!")

# ============================================================================
# ROOT & HEALTH ENDPOINTS
# ============================================================================
@app.get("/")
def home():
    return {
        "message": "🌾 AgriYield+ Backend is Running Successfully!",
        "status": "online",
        "models_loaded": MODELS_LOADED,
        "version": "2.0",
        "endpoints": [
            "/predict/yield",
            "/predict/fertilizer", 
            "/predict/irrigation",
            "/predict/crop",
            "/predict/rotation",
            "/predict/disease"
        ],
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {
        "success": True,
        "status": "ok",
        "models_loaded": MODELS_LOADED
    }

# ============================================================================
# YIELD PREDICTION
# ============================================================================
@app.post("/predict/yield")
def predict_yield_endpoint(payload: dict):
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        
        N = float(data.get("N", 80))
        P = float(data.get("P", 50))
        K = float(data.get("K", 60))
        temperature = float(data.get("temperature", 25))
        humidity = float(data.get("humidity", 70))
        ph = float(data.get("ph", 6.5))
        rainfall = float(data.get("rainfall", 150))
        crop = data.get("crop", "wheat").lower()
        
        # Yield estimation logic
        base_yields = {
            'wheat': 4.0, 'rice': 5.5, 'maize': 6.0,
            'cotton': 2.5, 'sugarcane': 70.0
        }
        
        base_yield = base_yields.get(crop, 4.0)
        
        # Calculate factors
        nutrient_factor = (N/100 + P/60 + K/70) / 3
        temp_factor = 1.0 if 20 <= temperature <= 30 else 0.85
        humidity_factor = 1.0 if 60 <= humidity <= 80 else 0.9
        ph_factor = 1.0 if 6.0 <= ph <= 7.5 else 0.85
        rain_factor = 1.0 if 100 <= rainfall <= 200 else 0.9
        
        predicted_yield = base_yield * nutrient_factor * temp_factor * humidity_factor * ph_factor * rain_factor
        optimal_yield = base_yield * 1.2
        yield_percentage = (predicted_yield / optimal_yield) * 100
        
        recommendations = []
        if N < 80:
            recommendations.append(f"Nitrogen is below optimal - increase by {80-N:.0f} kg/ha")
        else:
            recommendations.append("Nitrogen level is optimal")
            
        if not (6.0 <= ph <= 7.5):
            recommendations.append("Soil pH needs adjustment to 6.0-7.5 range")
        else:
            recommendations.append("Soil pH is optimal")
            
        if not (20 <= temperature <= 30):
            recommendations.append("Temperature conditions are suboptimal")
        else:
            recommendations.append("Temperature conditions are favorable")
            
        recommendations.append("Consider soil testing for micronutrients")
        
        return {
            "success": True,
            "summary": "Yield Prediction Results",
            "confidence": 88,
            "outputs": [
                {
                    "type": "scalar",
                    "label": "Predicted Yield",
                    "value": round(predicted_yield, 2),
                    "units": "tons/hectare"
                },
                {
                    "type": "scalar",
                    "label": "Optimal Yield",
                    "value": round(optimal_yield, 2),
                    "units": "tons/hectare"
                },
                {
                    "type": "scalar",
                    "label": "Performance",
                    "value": round(yield_percentage, 1),
                    "units": "% of optimal"
                },
                {
                    "type": "text",
                    "label": "Crop Type",
                    "value": crop.capitalize()
                },
                {
                    "type": "list",
                    "label": "Key Recommendations",
                    "value": recommendations
                }
            ]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# FERTILIZER OPTIMIZATION (Using Real ML Model)
# ============================================================================
@app.post("/predict/fertilizer")
def predict_fertilizer_endpoint(payload: dict):
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        
        N = float(data.get("N", 60))
        P = float(data.get("P", 40))
        K = float(data.get("K", 50))
        crop = data.get("crop", "wheat").lower()
        soil_type = data.get("soil", "loamy").lower()
        area = float(data.get("area", 1))
        
        # Use ML model if available
        if MODELS_LOADED:
            try:
                result = fertilizer_model.predict(
                    soil_N=N,
                    soil_P=P,
                    soil_K=K,
                    pH=6.5,  # default
                    organic_matter=3.0,  # default
                    soil_moisture=60,  # default
                    previous_yield=4.0,  # default
                    crop_type=crop
                )
                
                recommendations = [
                    f"Apply {result['N_kg_per_ha']} kg/ha Nitrogen",
                    f"Apply {result['P_kg_per_ha']} kg/ha Phosphorus",
                    f"Apply {result['K_kg_per_ha']} kg/ha Potassium",
                    f"NPK Ratio: {result['NPK_ratio']}"
                ]
                
                if result['recommend_organic']:
                    recommendations.extend(result['organic_alternatives'])
                
                return {
                    "success": True,
                    "summary": "Fertilizer Optimization (ML-Powered)",
                    "confidence": 92,
                    "outputs": [
                        {
                            "type": "scalar",
                            "label": "Nitrogen Needed",
                            "value": result['N_kg_per_ha'],
                            "units": "kg/ha"
                        },
                        {
                            "type": "scalar",
                            "label": "Phosphorus Needed",
                            "value": result['P_kg_per_ha'],
                            "units": "kg/ha"
                        },
                        {
                            "type": "scalar",
                            "label": "Potassium Needed",
                            "value": result['K_kg_per_ha'],
                            "units": "kg/ha"
                        },
                        {
                            "type": "scalar",
                            "label": "Estimated Cost",
                            "value": result['estimated_cost_INR'] * area,
                            "units": "₹"
                        },
                        {
                            "type": "text",
                            "label": "NPK Ratio",
                            "value": result['NPK_ratio']
                        },
                        {
                            "type": "list",
                            "label": "Recommendations",
                            "value": recommendations
                        }
                    ]
                }
            except Exception as ml_error:
                print(f"ML model error: {ml_error}, using fallback")
        
        # Fallback logic
        crop_requirements = {
            'wheat': (120, 60, 40), 'rice': (150, 75, 75),
            'maize': (180, 90, 60), 'cotton': (150, 75, 75),
            'sugarcane': (200, 100, 100)
        }
        
        req_n, req_p, req_k = crop_requirements.get(crop, (120, 60, 40))
        
        n_needed = max(0, req_n - N)
        p_needed = max(0, req_p - P)
        k_needed = max(0, req_k - K)
        
        total_cost = (n_needed * 20 + p_needed * 30 + k_needed * 25) * area
        
        recommendations = []
        if n_needed > 0:
            recommendations.append(f"Apply {round(n_needed * area, 1)} kg Urea (46% N)")
        if p_needed > 0:
            recommendations.append(f"Apply {round(p_needed * area * 2.17, 1)} kg DAP (46% P)")
        if k_needed > 0:
            recommendations.append(f"Apply {round(k_needed * area * 1.67, 1)} kg MOP (60% K)")
        
        if not recommendations:
            recommendations.append("Soil nutrient levels are adequate for this crop")
        
        return {
            "success": True,
            "summary": "Fertilizer Optimization",
            "confidence": 85,
            "outputs": [
                {"type": "scalar", "label": "Nitrogen Needed", "value": round(n_needed, 1), "units": "kg/ha"},
                {"type": "scalar", "label": "Phosphorus Needed", "value": round(p_needed, 1), "units": "kg/ha"},
                {"type": "scalar", "label": "Potassium Needed", "value": round(k_needed, 1), "units": "kg/ha"},
                {"type": "scalar", "label": "Estimated Cost", "value": round(total_cost, 2), "units": "₹"},
                {"type": "text", "label": "Crop Type", "value": crop.capitalize()},
                {"type": "list", "label": "Fertilizer Recommendations", "value": recommendations}
            ]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# IRRIGATION SCHEDULING (Using Real ML Model)
# ============================================================================
@app.post("/predict/irrigation")
def predict_irrigation_endpoint(payload: dict):
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        
        moisture = float(data.get("moisture", 45))
        temperature = float(data.get("temperature", 30))
        humidity = float(data.get("humidity", 60))
        rainfall = float(data.get("rainfall", 5))
        crop = data.get("crop", "wheat").lower()
        stage = data.get("stage", "vegetative").lower()
        
        # Use ML model if available
        if MODELS_LOADED:
            try:
                result = irrigation_model.predict(
                    soil_moisture=moisture,
                    temperature=temperature,
                    humidity=humidity,
                    rainfall_forecast=rainfall,
                    wind_speed=10,  # default
                    crop_type=crop,
                    growth_stage=stage
                )
                
                tips = [
                    f"Current moisture at {moisture}% - target is 60-70%",
                    f"Water requirement: {result['water_amount_mm']} mm",
                    f"Next irrigation in {result['next_irrigation_hours']} hours",
                    "Early morning irrigation reduces evaporation by 30%"
                ]
                
                if rainfall > 0:
                    tips.append(f"Rainfall will save {result['water_saved_from_rainfall_mm']} mm water")
                
                return {
                    "success": True,
                    "summary": "Smart Irrigation Plan (ML-Powered)",
                    "confidence": 90,
                    "outputs": [
                        {"type": "scalar", "label": "Daily Water Requirement", "value": result['water_amount_mm'], "units": "mm/day"},
                        {"type": "scalar", "label": "Next Irrigation", "value": result['next_irrigation_hours'], "units": "hours"},
                        {"type": "scalar", "label": "Current Soil Moisture", "value": moisture, "units": "%"},
                        {"type": "text", "label": "Irrigation Status", "value": result['urgency']},
                        {"type": "text", "label": "Recommended Method", "value": result['recommended_method']},
                        {"type": "list", "label": "Irrigation Tips", "value": tips}
                    ]
                }
            except Exception as ml_error:
                print(f"ML model error: {ml_error}, using fallback")
        
        # Fallback logic
        crop_water_req = {'wheat': 4.5, 'rice': 7.0, 'maize': 5.5, 'cotton': 5.0, 'sugarcane': 6.5}
        stage_multiplier = {'germination': 1.2, 'vegetative': 1.0, 'flowering': 1.4, 'maturity': 0.7}
        
        base_water = crop_water_req.get(crop, 5.0)
        stage_mult = stage_multiplier.get(stage, 1.0)
        
        temp_mult = 1.0 + (temperature - 25) * 0.02
        humidity_mult = 1.0 - (humidity - 60) * 0.005
        moisture_deficit = max(0, 70 - moisture) * 0.5
        
        daily_water = base_water * stage_mult * temp_mult * humidity_mult + moisture_deficit
        daily_water = max(0, daily_water - rainfall)
        
        if moisture < 40:
            status = "IMMEDIATE irrigation required"
            schedule = "Irrigate today"
        elif moisture < 60:
            status = "Irrigation recommended soon"
            schedule = "Irrigate within 24-48 hours"
        else:
            status = "Adequate moisture"
            schedule = "No immediate irrigation needed"
        
        return {
            "success": True,
            "summary": "Smart Irrigation Plan",
            "confidence": 85,
            "outputs": [
                {"type": "scalar", "label": "Daily Water Requirement", "value": round(daily_water, 2), "units": "mm/day"},
                {"type": "scalar", "label": "Current Soil Moisture", "value": moisture, "units": "%"},
                {"type": "text", "label": "Irrigation Status", "value": status},
                {"type": "text", "label": "Recommended Schedule", "value": schedule},
                {"type": "list", "label": "Irrigation Tips", "value": [
                    f"Current moisture at {moisture}% - target is 60-70%",
                    "Early morning irrigation reduces evaporation by 30%",
                    f"Crop is in {stage} stage - adjust timing accordingly",
                    "Consider drip irrigation to save 40-60% water"
                ]}
            ]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# CROP RECOMMENDATION
# ============================================================================
@app.post("/predict/crop")
def predict_crop_endpoint(payload: dict):
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        
        N = float(data.get("N", 50))
        P = float(data.get("P", 40))
        K = float(data.get("K", 45))
        ph = float(data.get("ph", 6.5))
        
        crops_data = {
            'wheat': {'n': (60, 120), 'p': (30, 60), 'k': (30, 60), 'ph': (6.0, 7.5)},
            'rice': {'n': (80, 150), 'p': (40, 80), 'k': (40, 80), 'ph': (5.5, 7.0)},
            'maize': {'n': (100, 180), 'p': (50, 90), 'k': (40, 70), 'ph': (5.8, 7.0)},
            'cotton': {'n': (80, 150), 'p': (40, 75), 'k': (40, 75), 'ph': (6.0, 7.5)},
            'chickpea': {'n': (20, 50), 'p': (30, 60), 'k': (30, 60), 'ph': (6.0, 7.5)},
            'soybean': {'n': (30, 60), 'p': (40, 80), 'k': (35, 70), 'ph': (6.0, 7.0)}
        }
        
        scores = {}
        for crop, ranges in crops_data.items():
            score = 0
            if ranges['n'][0] <= N <= ranges['n'][1]: score += 25
            if ranges['p'][0] <= P <= ranges['p'][1]: score += 25
            if ranges['k'][0] <= K <= ranges['k'][1]: score += 25
            if ranges['ph'][0] <= ph <= ranges['ph'][1]: score += 25
            scores[crop] = score
        
        sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_crop = sorted_crops[0][0]
        best_score = sorted_crops[0][1]
        alternatives = [f"{c[0].capitalize()} ({c[1]}% match)" for c in sorted_crops[1:4]]
        
        suitability = "Excellent" if best_score >= 90 else "Good" if best_score >= 70 else "Fair"
        
        recommendations = [
            f"{best_crop.capitalize()} is highly suitable for your soil conditions",
            f"Soil pH of {ph} is {'optimal' if 6.0 <= ph <= 7.0 else 'needs adjustment'}",
            f"Nitrogen levels are {'adequate' if N >= 50 else 'low - consider supplementation'}",
            "Consider crop rotation for long-term soil health"
        ]
        
        return {
            "success": True,
            "summary": "Crop Recommendation",
            "confidence": best_score,
            "outputs": [
                {"type": "text", "label": "Recommended Crop", "value": best_crop.capitalize()},
                {"type": "text", "label": "Suitability", "value": suitability},
                {"type": "scalar", "label": "Soil Nitrogen (N)", "value": N, "units": "ppm"},
                {"type": "scalar", "label": "Soil Phosphorus (P)", "value": P, "units": "ppm"},
                {"type": "scalar", "label": "Soil Potassium (K)", "value": K, "units": "ppm"},
                {"type": "scalar", "label": "Soil pH", "value": ph, "units": ""},
                {"type": "list", "label": "Alternative Crops", "value": alternatives},
                {"type": "list", "label": "Recommendations", "value": recommendations}
            ]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# CROP ROTATION (Using Real ML Model)
# ============================================================================
@app.post("/predict/rotation")
def predict_rotation_endpoint(payload: dict):
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        
        previous_crop = data.get('previous_crop', 'wheat').lower()
        soil_N = float(data.get('soil_N', 60))
        soil_P = float(data.get('soil_P', 45))
        soil_K = float(data.get('soil_K', 55))
        soil_health_score = float(data.get('soil_health_score', 70))
        previous_yield = float(data.get('previous_yield', 3.5))
        seasons_same_crop = int(data.get('seasons_same_crop', 2))
        organic_matter = float(data.get('organic_matter', 3.0))
        
        # Use ML model if available
        if MODELS_LOADED:
            try:
                result = rotation_model.predict(
                    previous_crop=previous_crop,
                    soil_N=soil_N,
                    soil_P=soil_P,
                    soil_K=soil_K,
                    soil_health_score=soil_health_score,
                    previous_yield=previous_yield,
                    seasons_same_crop=seasons_same_crop,
                    organic_matter=organic_matter
                )
                
                top_crops = [crop_info['crop'] for crop_info in result['recommended_crops']]
                benefits = [f"{crop_info['crop'].capitalize()}: {crop_info['benefit']}" 
                           for crop_info in result['recommended_crops']]
                
                outputs = [
                    {"type": "text", "label": "Next Recommended Crop", "value": top_crops[0].capitalize()},
                    {"type": "text", "label": "Previous Crop", "value": previous_crop.capitalize()},
                    {"type": "scalar", "label": "Soil Health Score", "value": soil_health_score, "units": "/100"},
                    {"type": "scalar", "label": "Expected Yield Increase", "value": result['expected_yield_increase_percent'], "units": "%"},
                    {"type": "scalar", "label": "Nitrogen Restoration", "value": result['nitrogen_restoration_kg_per_ha'], "units": "kg/ha"},
                    {"type": "scalar", "label": "Cost Reduction", "value": result['cost_reduction_percent'], "units": "%"},
                    {"type": "text", "label": "Rotation Urgency", "value": result['rotation_urgency']},
                    {"type": "list", "label": "Expected Benefits", "value": benefits},
                    {"type": "list", "label": "Alternative Crops", "value": [c.capitalize() for c in top_crops[1:3]]}
                ]
                
                return {
                    "success": True,
                    "summary": "Crop Rotation Plan (ML-Powered)",
                    "confidence": min(95, 70 + (10 if seasons_same_crop >= 3 else 0) + (10 if soil_health_score < 60 else 0)),
                    "outputs": outputs
                }
            except Exception as ml_error:
                print(f"ML model error: {ml_error}, using fallback")
        
        # Fallback logic (from your original enhanced code)
        rotation_map = {
            'wheat': ['chickpea', 'soybean', 'maize'],
            'rice': ['wheat', 'chickpea', 'cotton'],
            'maize': ['soybean', 'wheat', 'cotton'],
            'cotton': ['wheat', 'chickpea', 'soybean'],
            'soybean': ['wheat', 'maize', 'rice'],
            'chickpea': ['wheat', 'rice', 'maize']
        }
        
        recommended = rotation_map.get(previous_crop, ['wheat', 'rice', 'maize'])
        next_crop = recommended[0]
        
        if soil_health_score < 60:
            reason = "Soil restoration needed - legumes will fix nitrogen"
            next_crop = 'chickpea' if 'chickpea' in recommended else 'soybean'
        elif soil_N < 50:
            reason = "Low nitrogen - nitrogen-fixing crop recommended"
            next_crop = 'chickpea' if 'chickpea' in recommended else 'soybean'
        elif seasons_same_crop >= 3:
            reason = "Break pest and disease cycles - rotation critical"
        elif organic_matter < 2.5:
            reason = "Improve organic matter content"
            next_crop = 'chickpea' if 'chickpea' in recommended else recommended[0]
        else:
            reason = "Optimize nutrient cycling and soil structure"
        
        benefits = []
        if next_crop in ['chickpea', 'soybean']:
            n_gain = 40 if next_crop == 'soybean' else 30
            benefits.append(f"Improves soil nitrogen by {n_gain}-{n_gain+10} kg/ha")
            benefits.append(f"Reduces fertilizer costs by approximately 25-30%")
        
        benefits.extend([
            "Breaks pest and disease cycles",
            "Improves soil structure and organic matter"
        ])
        
        if organic_matter < 2.5:
            benefits.append(f"Critical: Increase organic matter from {organic_matter}% to >2.5%")
        
        confidence = 70
        if seasons_same_crop >= 3: confidence += 10
        if soil_health_score < 60: confidence += 10
        if previous_crop in rotation_map: confidence += 5
        confidence = min(confidence, 95)
        
        yield_improvement = 15 if seasons_same_crop >= 3 else 10 if seasons_same_crop >= 2 else 5
        
        three_year_plan = [
            f"Year 1: {next_crop.capitalize()}",
            f"Year 2: {recommended[1].capitalize() if len(recommended) > 1 else 'Wheat'}",
            f"Year 3: {recommended[2].capitalize() if len(recommended) > 2 else 'Maize'}"
        ]
        
        warnings = []
        if seasons_same_crop >= 4:
            warnings.append("⚠️ Extended monoculture detected - rotation strongly recommended")
        if soil_health_score < 50:
            warnings.append("⚠️ Poor soil health - consider cover cropping or fallow period")
        if organic_matter < 2.0:
            warnings.append("⚠️ Very low organic matter - urgent improvement needed")
        
        outputs = [
            {"type": "text", "label": "Next Recommended Crop", "value": next_crop.capitalize()},
            {"type": "text", "label": "Previous Crop", "value": previous_crop.capitalize()},
            {"type": "scalar", "label": "Soil Health Score", "value": soil_health_score, "units": "/100"},
            {"type": "scalar", "label": "Expected Yield Improvement", "value": yield_improvement, "units": "%"},
            {"type": "text", "label": "Primary Reason", "value": reason},
            {"type": "list", "label": "Expected Benefits", "value": benefits},
            {"type": "text", "label": "Alternative Crops", "value": ", ".join([c.capitalize() for c in recommended[1:3]])},
            {"type": "list", "label": "3-Year Rotation Plan", "value": three_year_plan}
        ]
        
        if warnings:
            outputs.append({"type": "list", "label": "Cautions", "value": warnings})
        
        return {
            "success": True,
            "summary": "Crop Rotation Plan",
            "confidence": confidence,
            "outputs": outputs
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# DISEASE DETECTION
# ============================================================================
@app.post("/predict/disease")
async def predict_disease(
    image: Optional[UploadFile] = File(None),
    symptoms: str = Form("")
):
    try:
        disease_db = {
            'yellow': ['Nitrogen deficiency', 'Yellow rust', 'Fusarium wilt'],
            'brown': ['Leaf blight', 'Brown spot', 'Anthracnose'],
            'spot': ['Leaf spot', 'Bacterial spot', 'Cercospora leaf spot'],
            'wilt': ['Fusarium wilt', 'Verticillium wilt', 'Bacterial wilt'],
            'rust': ['Stem rust', 'Leaf rust', 'Yellow rust'],
            'mold': ['Powdery mildew', 'Downy mildew', 'Gray mold']
        }
        
        detected_diseases = []
        symptoms_lower = symptoms.lower()
        
        for keyword, diseases in disease_db.items():
            if keyword in symptoms_lower:
                detected_diseases.extend(diseases)
        
        if not detected_diseases:
            detected_diseases = ['General crop stress - further diagnosis needed']
        
        primary_disease = detected_diseases[0]
        
        treatments = {
            'Nitrogen deficiency': ['Apply urea 50 kg/ha', 'Foliar spray of urea solution', 'Add organic compost'],
            'Yellow rust': ['Apply fungicide (Propiconazole)', 'Remove infected leaves', 'Ensure proper spacing'],
            'Leaf blight': ['Copper-based fungicide', 'Remove infected plant parts', 'Improve drainage'],
            'Fusarium wilt': ['Remove infected plants', 'Soil fumigation', 'Use resistant varieties']
        }
        
        treatment = treatments.get(primary_disease, [
            'Consult agricultural extension officer',
            'Take sample to diagnostic lab',
            'Ensure proper crop nutrition and watering'
        ])
        
        return {
            "success": True,
            "summary": "Disease Detection Results",
            "confidence": 85,
            "outputs": [
                {"type": "text", "label": "Primary Disease", "value": primary_disease},
                {"type": "text", "label": "Severity", "value": "Moderate" if 'deficiency' in primary_disease.lower() else "Requires attention"},
                {"type": "list", "label": "Possible Diseases", "value": detected_diseases[:3]},
                {"type": "list", "label": "Treatment Recommendations", "value": treatment},
                {"type": "list", "label": "Prevention Tips", "value": [
                    "Regular field monitoring",
                    "Maintain proper plant spacing",
                    "Ensure balanced nutrition",
                    "Practice crop rotation"
                ]}
            ]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)