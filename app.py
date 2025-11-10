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
def predict_crop(payload: dict):
    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        N = float(data.get("N", 20)); P = float(data.get("P",10)); K = float(data.get("K",10)); ph = float(data.get("ph",6.5))
        if 'predict_crop' in globals():
            out = predict_crop(N=N,P=P,K=K,ph=ph)
            if isinstance(out, dict) and 'crop' in out:
                cropname = out['crop']
            elif isinstance(out, (list,tuple)):
                cropname = out[0]
            else:
                cropname = out
            return {"success":True,"summary":"Crop Prediction","confidence":75,"outputs":[{"label":"Predicted Crop","value":str(cropname),"type":"text"}]}
        else:
            return {"success":False,"error":"predict_crop not found in module"}
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