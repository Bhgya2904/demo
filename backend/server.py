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
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cyber Breach Forecaster API", version="1.0.0")

# Load environment variables
load_dotenv()

# VirusTotal API configuration
VIRUSTOTAL_API_KEY = os.getenv('VIRUSTOTAL_API_KEY')
VIRUSTOTAL_BASE_URL = "https://www.virustotal.com/vtapi/v2"

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

# Website Security Check Models
class WebsiteCheckRequest(BaseModel):
    """Request schema for website security check."""
    url: str

class WebsiteCheckResponse(BaseModel):
    """Response schema for website security check."""
    url: str
    verdict: str  # "Safe", "Suspicious", "Malicious", "Unknown"
    total_scans: int
    positive_detections: int
    scan_date: str
    permalink: str
    detailed_results: Dict[str, Any]
    threat_explanation: str

# Network Analysis Models (Enhanced UNSW-NB15)
class NetworkTrafficRequest(BaseModel):
    """Request schema for network traffic analysis."""
    dur: float = Field(..., description="Duration of connection")
    proto: str = Field(..., description="Protocol type (tcp/udp/icmp)")
    service: str = Field(..., description="Service type (http/ftp/ssh/etc)")
    state: str = Field(..., description="Connection state")
    spkts: int = Field(..., description="Source to destination packets")
    dpkts: int = Field(..., description="Destination to source packets") 
    sbytes: int = Field(..., description="Source to destination bytes")
    dbytes: int = Field(..., description="Destination to source bytes")
    rate: float = Field(..., description="Connection rate")
    sttl: int = Field(..., description="Source TTL")
    dttl: int = Field(..., description="Destination TTL")
    sload: float = Field(..., description="Source load")
    dload: float = Field(..., description="Destination load")
    sloss: int = Field(0, description="Source packets retransmitted")
    dloss: int = Field(0, description="Destination packets retransmitted")
    sinpkt: float = Field(..., description="Source interpacket arrival time")
    dinpkt: float = Field(..., description="Destination interpacket arrival time")
    sjit: float = Field(0, description="Source jitter")
    djit: float = Field(0, description="Destination jitter")
    swin: int = Field(0, description="Source TCP window advertisement")
    stcpb: int = Field(0, description="Source TCP base sequence number")
    dtcpb: int = Field(0, description="Destination TCP base sequence number")
    dwin: int = Field(0, description="Destination TCP window advertisement")
    tcprtt: float = Field(0, description="TCP round trip time")
    synack: float = Field(0, description="TCP SYN-ACK time")
    ackdat: float = Field(0, description="TCP ACK-DAT time")
    smean: int = Field(0, description="Source packet size mean")
    dmean: int = Field(0, description="Destination packet size mean")
    trans_depth: int = Field(0, description="Transaction depth")
    response_body_len: int = Field(0, description="Response body length")
    ct_srv_src: int = Field(..., description="Connections to same service from source")
    ct_state_ttl: int = Field(..., description="Connections with same state and TTL")
    ct_dst_ltm: int = Field(..., description="Connections to destination in last time")
    ct_src_dport_ltm: int = Field(0, description="Connections from source to dest port in last time")
    ct_dst_sport_ltm: int = Field(0, description="Connections from dest to source port in last time")
    ct_dst_src_ltm: int = Field(0, description="Connections between dest and source in last time")
    is_ftp_login: int = Field(0, description="FTP login attempt (0/1)")
    ct_ftp_cmd: int = Field(0, description="FTP command count")
    ct_flw_http_mthd: int = Field(0, description="HTTP method count in flow")
    ct_src_ltm: int = Field(0, description="Connections from source in last time")
    ct_srv_dst: int = Field(0, description="Connections to same service at destination")
    is_sm_ips_ports: int = Field(0, description="Source and destination IPs/ports are same (0/1)")

class NetworkTrafficResponse(BaseModel):
    """Response schema for network traffic analysis."""
    prediction: str  # "Safe Traffic" or "Attack Predicted"
    confidence: float
    attack_probability: float
    detailed_explanation: str
    feature_importance: Optional[Dict[str, float]] = None

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

@app.post("/api/check-website", response_model=WebsiteCheckResponse)
async def check_website_security(request: WebsiteCheckRequest):
    """Check website security using VirusTotal API."""
    if not VIRUSTOTAL_API_KEY:
        raise HTTPException(status_code=500, detail="VirusTotal API key not configured")
    
    try:
        # Ensure URL has protocol
        url = request.url
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # First, submit URL for scanning
        submit_params = {
            'apikey': VIRUSTOTAL_API_KEY,
            'url': url
        }
        
        submit_response = requests.post(
            f"{VIRUSTOTAL_BASE_URL}/url/scan",
            data=submit_params,
            timeout=30
        )
        submit_response.raise_for_status()
        submit_data = submit_response.json()
        
        if submit_data.get('response_code') != 1:
            raise HTTPException(status_code=400, detail="Failed to submit URL for scanning")
        
        # Get the scan report
        report_params = {
            'apikey': VIRUSTOTAL_API_KEY,
            'resource': url
        }
        
        report_response = requests.get(
            f"{VIRUSTOTAL_BASE_URL}/url/report",
            params=report_params,
            timeout=30
        )
        report_response.raise_for_status()
        report_data = report_response.json()
        
        # Process the results
        if report_data.get('response_code') == 1:
            positives = report_data.get('positives', 0)
            total = report_data.get('total', 0)
            
            # Determine verdict
            if positives == 0:
                verdict = "Safe"
                threat_explanation = "No security engines detected any threats. The URL appears to be safe."
            elif positives <= 2:
                verdict = "Suspicious"
                threat_explanation = f"A small number ({positives}) of security engines flagged this URL. Exercise caution."
            else:
                verdict = "Malicious"
                threat_explanation = f"{positives} out of {total} security engines detected threats. This URL is likely malicious."
            
            return WebsiteCheckResponse(
                url=url,
                verdict=verdict,
                total_scans=total,
                positive_detections=positives,
                scan_date=report_data.get('scan_date', ''),
                permalink=report_data.get('permalink', ''),
                detailed_results=report_data.get('scans', {}),
                threat_explanation=threat_explanation
            )
        else:
            # URL not found in database yet, return pending status
            return WebsiteCheckResponse(
                url=url,
                verdict="Unknown",
                total_scans=0,
                positive_detections=0,
                scan_date='',
                permalink='',
                detailed_results={},
                threat_explanation="URL is being analyzed. Please try again in a few moments."
            )
            
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"VirusTotal API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze-network", response_model=NetworkTrafficResponse)
async def analyze_network_traffic(request: NetworkTrafficRequest):
    """Analyze network traffic for potential intrusions using enhanced UNSW-NB15 model."""
    if not models:
        raise HTTPException(status_code=503, detail="ML models not available")
    
    try:
        # Convert request to DataFrame with proper feature order
        input_data = {}
        
        # Map all request fields to feature columns
        request_dict = request.dict()
        for feature in feature_columns:
            if feature in request_dict:
                input_data[feature] = request_dict[feature]
            else:
                # Set default values for missing features
                input_data[feature] = 0
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Encode categorical features
        categorical_columns = ['proto', 'service', 'state']
        for col in categorical_columns:
            if col in df.columns and col in label_encoders:
                try:
                    # Handle unknown categories by using the most frequent category
                    le = label_encoders[col]
                    if df[col].iloc[0] in le.classes_:
                        df[col] = le.transform([df[col].iloc[0]])
                    else:
                        # Use most frequent class (usually 0 for the first class)
                        df[col] = [0]
                except Exception:
                    df[col] = [0]
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Make predictions
        rf_pred_proba = models['rf'].predict_proba(X_scaled)[0, 1]
        xgb_pred_proba = models['xgb'].predict_proba(X_scaled)[0, 1]
        
        # Ensemble prediction
        ensemble_proba = (rf_pred_proba * 0.6) + (xgb_pred_proba * 0.4)
        prediction = int(ensemble_proba > 0.5)
        
        # Generate response
        if prediction == 0:
            result = "Safe Traffic"
            explanation = "The network traffic pattern appears normal with no signs of malicious activity."
        else:
            result = "Attack Predicted"
            explanation = f"The network traffic shows patterns consistent with malicious activity. Confidence: {ensemble_proba:.2%}. This could indicate various types of network attacks including reconnaissance, exploitation attempts, or data exfiltration."
        
        # Feature importance (simplified)
        feature_importance = {}
        if hasattr(models['rf'], 'feature_importances_'):
            importances = models['rf'].feature_importances_
            for i, feature in enumerate(feature_columns[:10]):  # Top 10 features
                if i < len(importances):
                    feature_importance[feature] = float(importances[i])
        
        return NetworkTrafficResponse(
            prediction=result,
            confidence=float(abs(ensemble_proba - 0.5) * 2),  # Convert to 0-1 confidence
            attack_probability=float(ensemble_proba),
            detailed_explanation=explanation,
            feature_importance=feature_importance
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Network analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)