from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cyber Breach Forecaster API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
models = {}
scaler = None
label_encoders = {}
feature_columns = []
attack_scenarios = {}
model_metrics = {}

class PredictionInput(BaseModel):
    """Input schema for breach prediction."""
    source_ip: str = Field(..., description="Source IP address")
    dest_ip: str = Field(..., description="Destination IP address")
    duration: float = Field(..., ge=0, description="Connection duration in seconds")
    protocol: str = Field(..., description="Protocol (tcp, udp, icmp)")
    service: str = Field(..., description="Service name (http, ssh, ftp, etc.)")
    state: str = Field(..., description="Connection state (FIN, INT, REQ, etc.)")
    source_packets: int = Field(..., ge=0, description="Source packets count")
    dest_packets: int = Field(..., ge=0, description="Destination packets count") 
    source_bytes: int = Field(..., ge=0, description="Source bytes")
    dest_bytes: int = Field(..., ge=0, description="Destination bytes")
    packet_rate: float = Field(..., ge=0, description="Packets per second")
    connection_attempts: int = Field(..., ge=0, description="Connection attempts")
    login_failures: int = Field(0, ge=0, description="Failed login attempts")

class PredictionResponse(BaseModel):
    """Response schema for breach prediction."""
    prediction: str  # "Safe" or "Potential Breach"
    probability: float  # 0.0 to 1.0
    risk_level: str  # "Low", "Medium", "High"
    confidence: float  # 0.0 to 1.0
    detected_patterns: List[str]
    timestamp: str

class AttackScenarioResponse(BaseModel):
    """Response schema for attack scenarios."""
    scenario_name: str
    description: str
    features: Dict[str, Any]

class MetricsResponse(BaseModel):
    """Response schema for model metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    training_date: str

@app.on_event("startup")
async def load_models():
    """Load trained models on server startup."""
    global models, scaler, label_encoders, feature_columns, attack_scenarios, model_metrics
    
    models_dir = '/app/backend/models'
    
    try:
        # Load models
        models['rf'] = joblib.load(f'{models_dir}/random_forest_model.pkl')
        models['xgb'] = joblib.load(f'{models_dir}/xgboost_model.pkl')
        scaler = joblib.load(f'{models_dir}/scaler.pkl')
        label_encoders = joblib.load(f'{models_dir}/label_encoders.pkl')
        
        # Load metadata
        with open(f'{models_dir}/model_metadata.json', 'r') as f:
            metadata = json.load(f)
            feature_columns = metadata['feature_columns']
            attack_scenarios = metadata['attack_scenarios']
            model_metrics = metadata['metrics']
        
        logger.info("✅ Models loaded successfully!")
        logger.info(f"📊 Model F1-Score: {model_metrics.get('f1_score', 'N/A')}")
        
    except FileNotFoundError:
        logger.error("❌ Models not found! Please run train_model.py first.")
        models = {}
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        models = {}

def convert_input_to_features(input_data: PredictionInput) -> np.ndarray:
    """Convert user input to model features."""
    
    # Create base feature vector with zeros
    features = {col: 0 for col in feature_columns}
    
    # Map input to model features
    features['dur'] = input_data.duration
    features['proto'] = encode_categorical('proto', input_data.protocol.lower())
    features['service'] = encode_categorical('service', input_data.service.lower())
    features['state'] = encode_categorical('state', input_data.state.upper())
    features['spkts'] = input_data.source_packets
    features['dpkts'] = input_data.dest_packets
    features['sbytes'] = input_data.source_bytes
    features['dbytes'] = input_data.dest_bytes
    features['rate'] = input_data.packet_rate
    
    # Calculate derived features
    if input_data.duration > 0:
        features['sload'] = input_data.source_bytes / input_data.duration
        features['dload'] = input_data.dest_bytes / input_data.duration
    
    # Connection pattern features
    features['ct_dst_ltm'] = input_data.connection_attempts
    features['is_ftp_login'] = 1 if input_data.login_failures > 0 else 0
    
    # Statistical features (estimated)
    if input_data.source_packets > 0:
        features['sinpkt'] = input_data.source_bytes / input_data.source_packets
    if input_data.dest_packets > 0:
        features['dinpkt'] = input_data.dest_bytes / input_data.dest_packets
    
    # TTL values (estimated based on protocol)
    features['sttl'] = 64  # Common default
    features['dttl'] = 64 if input_data.dest_packets > 0 else 0
    
    # Convert to numpy array in correct order
    feature_vector = np.array([features[col] for col in feature_columns])
    return feature_vector.reshape(1, -1)

def encode_categorical(column: str, value: str) -> int:
    """Encode categorical values using saved label encoders."""
    if column in label_encoders:
        encoder = label_encoders[column]
        try:
            return encoder.transform([value])[0]
        except ValueError:
            # Handle unseen values
            return 0
    return 0

def detect_attack_patterns(input_data: PredictionInput) -> List[str]:
    """Detect specific attack patterns in the input."""
    patterns = []
    
    # High packet rate indicator
    if input_data.packet_rate > 1000:
        patterns.append("High Traffic Volume (DDoS-like)")
    
    # Port scanning indicators
    if input_data.source_packets < 5 and input_data.dest_packets == 0:
        patterns.append("Port Scanning Behavior")
    
    # Brute force indicators
    if input_data.login_failures > 0:
        patterns.append("Failed Authentication Attempts")
    
    # Large payload indicators
    if input_data.source_bytes > 5000 or input_data.dest_bytes > 10000:
        patterns.append("Large Data Transfer")
    
    # Connection flooding
    if input_data.connection_attempts > 10:
        patterns.append("Multiple Connection Attempts")
    
    return patterns

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Cyber Breach Forecaster API",
        "status": "active",
        "models_loaded": len(models) > 0,
        "version": "1.0.0"
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_breach(input_data: PredictionInput):
    """Predict if network traffic indicates a potential cyber breach."""
    
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert input to features
        features = convert_input_to_features(input_data)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get predictions from both models
        rf_proba = models['rf'].predict_proba(features_scaled)[0][1]
        xgb_proba = models['xgb'].predict_proba(features_scaled)[0][1]
        
        # Ensemble prediction (weighted average)
        ensemble_proba = (rf_proba * 0.6) + (xgb_proba * 0.4)
        prediction = "Potential Breach" if ensemble_proba > 0.5 else "Safe"
        
        # Determine risk level
        if ensemble_proba >= 0.8:
            risk_level = "High"
        elif ensemble_proba >= 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Detect patterns
        patterns = detect_attack_patterns(input_data)
        
        # Calculate confidence (distance from 0.5 threshold)
        confidence = abs(ensemble_proba - 0.5) * 2
        
        return PredictionResponse(
            prediction=prediction,
            probability=float(ensemble_proba),
            risk_level=risk_level,
            confidence=float(confidence),
            detected_patterns=patterns,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/attack-scenarios", response_model=List[AttackScenarioResponse])
async def get_attack_scenarios():
    """Get predefined attack scenarios for testing."""
    
    scenarios = []
    for key, scenario in attack_scenarios.items():
        scenarios.append(AttackScenarioResponse(
            scenario_name=scenario['name'],
            description=scenario['description'],
            features=scenario['features']
        ))
    
    return scenarios

@app.get("/api/attack-scenarios/{scenario_key}")
async def get_attack_scenario(scenario_key: str):
    """Get specific attack scenario features for form auto-fill."""
    
    if scenario_key not in attack_scenarios:
        raise HTTPException(status_code=404, detail="Scenario not found")
    
    scenario = attack_scenarios[scenario_key]
    features = scenario['features']
    
    # Map to API input format
    mapped_features = {
        "source_ip": "192.168.1.100",  # Example IP
        "dest_ip": "10.0.0.50",
        "duration": features.get('dur', 1.0),
        "protocol": features.get('proto', 'tcp'),
        "service": features.get('service', 'http'),
        "state": features.get('state', 'FIN'),
        "source_packets": features.get('spkts', 10),
        "dest_packets": features.get('dpkts', 10),
        "source_bytes": features.get('sbytes', 500),
        "dest_bytes": features.get('dbytes', 500),
        "packet_rate": features.get('rate', 10),
        "connection_attempts": features.get('ct_dst_ltm', 1),
        "login_failures": features.get('is_ftp_login', 0)
    }
    
    return {
        "scenario": scenario['name'],
        "description": scenario['description'],
        "features": mapped_features
    }

@app.get("/api/metrics", response_model=MetricsResponse)
async def get_model_metrics():
    """Get model performance metrics."""
    
    if not model_metrics:
        raise HTTPException(status_code=503, detail="Model metrics not available")
    
    return MetricsResponse(**model_metrics)

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    
    return {
        "status": "healthy",
        "models_loaded": len(models) > 0,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)