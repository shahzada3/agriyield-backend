from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, json, os
app = FastAPI(title="AgriYield Real ML API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
@app.get("/")
def home():
    """
    Root route for AgriYield Backend
    Shows a friendly status message when opened in a browser.
    """
    return {
        "message": "🌾 AgriYield Backend is Running Successfully!",
        "status": "online",
        "docs": "https://agriyield-backend.onrender.com/docs",
        "health": "https://agriyield-backend.onrender.com/health"
    }


# Import user notebook modules
from yield_module import *  # noqa
from irrigation_module import *  # noqa
from crop_module import *  # noqa

@app.get("/health")
def health():
    return {"success": True, "status": "ok"}

@app.post("/predict/yield")
def predict_yield(payload: dict):
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        # try to map expected names
        N = float(data.get("N", data.get("n", 0)))
        P = float(data.get("P", data.get("p", 0)))
        K = float(data.get("K", data.get("k", 0)))
        temperature = float(data.get("temperature", data.get("temp", 25)))
        humidity = float(data.get("humidity", 50))
        ph = float(data.get("ph", 6.5))
        rainfall = float(data.get("rainfall", 0))
        # call notebook's predict_yield if exists
        if 'predict_yield' in globals():
            out = predict_yield(N=N, P=P, K=K, temperature=temperature, humidity=humidity, ph=ph, rainfall=rainfall)
            # normalize output
            if isinstance(out, dict) and 'pred' in out:
                val = float(out['pred'])
            elif isinstance(out, (list, tuple)):
                val = float(out[0])
            else:
                val = float(out)
            return {"success": True, "summary": "Yield Prediction", "confidence": 85, "outputs":[{"label":"Predicted Yield","value":round(val,3),"type":"scalar","units":"ton/ha"}]}
        else:
            return {"success": False, "error":"predict_yield not found in module"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/predict/irrigation")
def predict_irrigation(payload: dict):
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        moisture = float(data.get("moisture", data.get("soil_moisture", 50)))
        temperature = float(data.get("temperature", data.get("temp",25)))
        humidity = float(data.get("humidity", 50))
        rainfall = float(data.get("rainfall", 0))
        crop = data.get("crop", "wheat")
        stage = data.get("stage", "vegetative")
        if 'predict_irrigation' in globals():
            res = predict_irrigation(moisture=moisture, temperature=temperature, humidity=humidity, rainfall=rainfall, crop=crop, stage=stage)
            # Expecting either dict with keys or list of outputs
            outputs = []
            if isinstance(res, dict):
                # try to extract keys water_needed, status, recommendation
                water = res.get('water_needed') or res.get('water') or res.get('water_needed_mm') or res.get('water_needed_mm', None)
                status = res.get('status') or res.get('irrigation_status') or res.get('status_text')
                rec = res.get('recommendation') or res.get('advice') or res.get('recommendation_text')
                if water is not None:
                    outputs.append({"label":"Water Needed","value":round(float(water),2),"type":"scalar","units":"mm"})
                if status is not None:
                    outputs.append({"label":"Irrigation Status","value":str(status),"type":"text"})
                if rec is not None:
                    outputs.append({"label":"Recommendation","value":str(rec),"type":"text"})
            elif isinstance(res, (list, tuple)) and len(res)>=3:
                outputs = [
                    {"label":"Water Needed","value":round(float(res[0]),2),"type":"scalar","units":"mm"},
                    {"label":"Irrigation Status","value":str(res[1]),"type":"text"},
                    {"label":"Recommendation","value":str(res[2]),"type":"text"}
                ]
            else:
                # fallback: single value
                outputs = [{"label":"Water Needed","value":round(float(res),2),"type":"scalar","units":"mm"}]
            return {"success":True,"summary":"Irrigation Plan","confidence":80,"outputs":outputs}
        else:
            return {"success":False,"error":"predict_irrigation not found in module"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/predict/crop")
def predict_crop_endpoint(payload: dict):
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        N = float(data.get("N", 20))
        P = float(data.get("P", 10))
        K = float(data.get("K", 10))
        ph = float(data.get("ph", 6.5))
        
        # Simple rule-based crop recommendation
        # You can enhance this with your actual ML model later
        crop_scores = {}
        
        # Rice: High N (80-120), moderate P (35-60), moderate K (35-60), pH 5.5-7.0
        rice_score = 0
        if 80 <= N <= 120: rice_score += 30
        elif N > 60: rice_score += 20
        if 35 <= P <= 60: rice_score += 25
        if 35 <= K <= 60: rice_score += 25
        if 5.5 <= ph <= 7.0: rice_score += 20
        crop_scores['rice'] = rice_score
        
        # Wheat: Moderate N (60-100), moderate P (30-50), moderate K (30-50), pH 6.0-7.5
        wheat_score = 0
        if 60 <= N <= 100: wheat_score += 30
        elif N > 40: wheat_score += 20
        if 30 <= P <= 50: wheat_score += 25
        if 30 <= K <= 50: wheat_score += 25
        if 6.0 <= ph <= 7.5: wheat_score += 20
        crop_scores['wheat'] = wheat_score
        
        # Maize: High N (100-150), high P (50-80), high K (50-80), pH 5.8-7.0
        maize_score = 0
        if 100 <= N <= 150: maize_score += 30
        elif N > 80: maize_score += 20
        if 50 <= P <= 80: maize_score += 25
        if 50 <= K <= 80: maize_score += 25
        if 5.8 <= ph <= 7.0: maize_score += 20
        crop_scores['maize'] = maize_score
        
        # Cotton: High N (100-140), moderate P (40-70), high K (60-90), pH 6.0-8.0
        cotton_score = 0
        if 100 <= N <= 140: cotton_score += 30
        elif N > 80: cotton_score += 20
        if 40 <= P <= 70: cotton_score += 25
        if 60 <= K <= 90: cotton_score += 25
        if 6.0 <= ph <= 8.0: cotton_score += 20
        crop_scores['cotton'] = cotton_score
        
        # Chickpea: Low N (20-40), moderate P (30-50), moderate K (30-50), pH 6.0-7.5
        chickpea_score = 0
        if 20 <= N <= 40: chickpea_score += 30
        elif N < 60: chickpea_score += 20
        if 30 <= P <= 50: chickpea_score += 25
        if 30 <= K <= 50: chickpea_score += 25
        if 6.0 <= ph <= 7.5: chickpea_score += 20
        crop_scores['chickpea'] = chickpea_score
        
        # Soybean: Low-moderate N (30-60), moderate P (35-55), moderate K (35-55), pH 6.0-7.0
        soybean_score = 0
        if 30 <= N <= 60: soybean_score += 30
        elif N < 80: soybean_score += 20
        if 35 <= P <= 55: soybean_score += 25
        if 35 <= K <= 55: soybean_score += 25
        if 6.0 <= ph <= 7.0: soybean_score += 20
        crop_scores['soybean'] = soybean_score
        
        # Get top 3 crops
        sorted_crops = sorted(crop_scores.items(), key=lambda x: x[1], reverse=True)
        top_crop = sorted_crops[0][0]
        confidence = min(95, sorted_crops[0][1])
        
        # Create recommendations
        recommendations = []
        for crop, score in sorted_crops[:3]:
            if score > 50:
                recommendations.append(f"{crop.capitalize()} (Match: {score}%)")
        
        return {
            "success": True,
            "summary": "Crop Recommendation",
            "confidence": confidence,
            "outputs": [
                {"label": "Recommended Crop", "value": top_crop.capitalize(), "type": "text"},
                {"label": "Soil Nitrogen (N)", "value": N, "type": "scalar", "units": "ppm"},
                {"label": "Soil Phosphorus (P)", "value": P, "type": "scalar", "units": "ppm"},
                {"label": "Soil Potassium (K)", "value": K, "type": "scalar", "units": "ppm"},
                {"label": "Soil pH", "value": ph, "type": "scalar", "units": "pH"},
                {"label": "Alternative Crops", "value": recommendations, "type": "list"}
            ]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/predict/fertilizer")
def predict_fertilizer(payload: dict):
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        N = float(data.get("N", data.get("n",50)))
        P = float(data.get("P", data.get("p",30)))
        K = float(data.get("K", data.get("k",40)))
        crop = data.get("crop","wheat")
        soil_type = data.get("soil","loamy")
        if 'predict_fertilizer' in globals():
            out = predict_fertilizer(N=N,P=P,K=K,crop=crop,soil=soil_type)
            # Expect three outputs N_recommend, P_recommend, K_recommend
            if isinstance(out, dict):
                return {"success":True,"summary":"Fertilizer Optimization","confidence":80,"outputs":[
                    {"label":"Recommended N","value":out.get("N_recommend", out.get("N_rec", N)),"type":"scalar","units":"kg/ha"},
                    {"label":"Recommended P","value":out.get("P_recommend", out.get("P_rec", P)),"type":"scalar","units":"kg/ha"},
                    {"label":"Recommended K","value":out.get("K_recommend", out.get("K_rec", K)),"type":"scalar","units":"kg/ha"}
                ]}
            elif isinstance(out,(list,tuple)) and len(out)>=3:
                return {"success":True,"summary":"Fertilizer Optimization","confidence":80,"outputs":[
                    {"label":"Recommended N","value":out[0],"type":"scalar","units":"kg/ha"},
                    {"label":"Recommended P","value":out[1],"type":"scalar","units":"kg/ha"},
                    {"label":"Recommended K","value":out[2],"type":"scalar","units":"kg/ha"}
                ]}
            else:
                return {"success":False,"error":"Unexpected fertilizer output format"}
        else:
            return {"success":False,"error":"predict_fertilizer not found in module"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/predict/rotation")
def predict_rotation(payload: dict):
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        # Use existing rotation logic if present
        if 'predict_rotation' in globals():
            out = predict_rotation(**data)
            # If returns dict/structured output, map to outputs
            if isinstance(out, dict):
                outputs = []
                for k,v in out.items():
                    outputs.append({"label":k,"value":str(v),"type":"text"})
                return {"success":True,"summary":"Rotation Plan","confidence":78,"outputs":outputs}
        # fallback heuristic (simple)
        prev_crop = data.get('previous_crop','wheat')
        soil_health = float(data.get('soil_health_score',70))
        suggestion = "Legume (Chickpea/Soybean)" if soil_health < 60 else "Cereal (Wheat/Maize)"
        return {"success":True,"summary":"Rotation Plan","confidence":70,"outputs":[{"label":"Next Suggested Crop","value":suggestion,"type":"text"}]}
    except Exception as e:
        return {"success": False, "error": str(e)}