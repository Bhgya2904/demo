#!/usr/bin/env python3
"""
Cyber Breach Forecaster - ML Model Training Script
Train Random Forest + XGBoost ensemble on UNSW-NB15 dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
import os
import json
from datetime import datetime

# Attack scenario templates based on cybersecurity research
ATTACK_SCENARIOS = {
    "ddos": {
        "name": "DDoS Attack",
        "description": "Distributed Denial of Service with high traffic volume",
        "features": {
            "dur": 0.001,  # Very short duration connections
            "proto": "tcp",
            "service": "http",
            "state": "REQ",
            "spkts": 150,  # High source packets
            "dpkts": 3,    # Low destination packets  
            "sbytes": 8500,  # High source bytes
            "dbytes": 120,   # Low destination bytes
            "rate": 85000,   # Very high packet rate
            "sttl": 64,
            "dttl": 254,
            "sload": 95000,  # Extremely high source load
            "dload": 450,    # Low destination load
            "sinpkt": 0.8,
            "dinpkt": 2.5,
            "ct_srv_src": 25,  # Many connections from same source
            "ct_state_ttl": 15,
            "ct_dst_ltm": 8
        }
    },
    "port_scan": {
        "name": "Port Scanning",
        "description": "Systematic scanning of network ports",
        "features": {
            "dur": 0.05,
            "proto": "tcp",
            "service": "-",
            "state": "INT",
            "spkts": 2,      # Few packets per connection
            "dpkts": 1,
            "sbytes": 48,    # Small packet sizes
            "dbytes": 0,     # Often no response
            "rate": 1200,
            "sttl": 64,
            "dttl": 0,       # No response TTL
            "sload": 950,
            "dload": 0,
            "sinpkt": 1.2,
            "dinpkt": 0,
            "ct_srv_src": 1,   # Each connection to different port
            "ct_state_ttl": 1,
            "ct_dst_ltm": 45   # Many recent connections to same destination
        }
    },
    "brute_force": {
        "name": "Brute Force Login",
        "description": "Repeated login attempts with different credentials",
        "features": {
            "dur": 2.5,
            "proto": "tcp",
            "service": "ssh",  # Common brute force target
            "state": "FIN",
            "spkts": 8,
            "dpkts": 6,
            "sbytes": 580,
            "dbytes": 320,
            "rate": 12,       # Lower rate but persistent
            "sttl": 64,
            "dttl": 64,
            "sload": 232,
            "dload": 128,
            "sinpkt": 1.8,
            "dinpkt": 1.5,
            "ct_srv_src": 1,
            "ct_state_ttl": 3,
            "ct_dst_ltm": 25,  # Multiple attempts to same destination
            "is_ftp_login": 1   # Login attempt indicator
        }
    },
    "sql_injection": {
        "name": "SQL Injection",
        "description": "Database injection attack through web application",
        "features": {
            "dur": 1.8,
            "proto": "tcp", 
            "service": "http",
            "state": "FIN",
            "spkts": 12,
            "dpkts": 15,
            "sbytes": 1850,    # Larger payload for SQL commands
            "dbytes": 2100,    # Server response with data
            "rate": 18,
            "sttl": 64,
            "dttl": 64,
            "sload": 1028,
            "dload": 1166,
            "sinpkt": 2.1,
            "dinpkt": 2.8,
            "ct_srv_src": 3,
            "ct_state_ttl": 5,
            "ct_flw_http_mthd": 3,  # HTTP method variations
            "response_body_len": 2500  # Large response indicating data extraction
        }
    },
    "malware": {
        "name": "Malware Communication",
        "description": "Malware communicating with command & control server",
        "features": {
            "dur": 15.5,      # Longer connection for C&C
            "proto": "tcp",
            "service": "-",   # Unknown service
            "state": "FIN", 
            "spkts": 45,
            "dpkts": 38,
            "sbytes": 2400,
            "dbytes": 8500,   # Downloading commands/updates
            "rate": 5.2,      # Low and steady rate
            "sttl": 64,
            "dttl": 64,
            "sload": 155,
            "dload": 548,
            "sinpkt": 4.8,    # Irregular timing
            "dinpkt": 6.2,
            "ct_srv_src": 1,
            "ct_state_ttl": 8,
            "ct_dst_ltm": 1,  # Usually single destination
            "trans_depth": 0   # No HTTP transaction depth
        }
    },
    "normal": {
        "name": "Normal Safe Traffic",
        "description": "Legitimate network traffic - web browsing, email, etc.",
        "features": {
            "dur": 3.2,
            "proto": "tcp",
            "service": "http",
            "state": "FIN",
            "spkts": 8,       # Balanced traffic
            "dpkts": 12,
            "sbytes": 650,    # Normal web request size
            "dbytes": 1800,   # Normal web response
            "rate": 6.8,      # Normal rate
            "sttl": 64,
            "dttl": 64,
            "sload": 203,
            "dload": 562,
            "sinpkt": 2.2,
            "dinpkt": 2.8,
            "ct_srv_src": 2,
            "ct_state_ttl": 4,
            "ct_flw_http_mthd": 1,
            "response_body_len": 1800,
            "trans_depth": 1   # Normal HTTP transaction
        }
    }
}

def prepare_data():
    """Load and preprocess the UNSW-NB15 dataset."""
    print("Loading UNSW-NB15 dataset...")
    
    # Load dataset
    df = pd.read_csv('/app/unsw_nb15_training.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Remove BOM from first column if present
    if df.columns[0].startswith('\ufeff'):
        df.columns = df.columns.str.replace('\ufeff', '')
    
    # Basic data cleaning
    df = df.dropna()
    
    # Prepare features and target
    # Remove id and attack_cat columns, use label as target
    feature_columns = [col for col in df.columns if col not in ['id', 'attack_cat', 'label']]
    X = df[feature_columns].copy()
    y = df['label'].copy()  # 0 = normal, 1 = attack
    
    print(f"Features: {len(feature_columns)}")
    print(f"Target distribution: Normal={sum(y==0)}, Attack={sum(y==1)}")
    
    # Handle categorical variables
    categorical_columns = ['proto', 'service', 'state']
    label_encoders = {}
    
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    return X, y, feature_columns, label_encoders

def train_ensemble_model(X_train, X_test, y_train, y_test):
    """Train Random Forest + XGBoost ensemble model."""
    print("\nTraining Random Forest + XGBoost ensemble...")
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict_proba(X_test)[:, 1]
    
    # XGBoost
    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    
    # Ensemble prediction (weighted average)
    ensemble_pred_proba = (rf_pred * 0.6) + (xgb_pred * 0.4)
    ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
    
    return rf_model, xgb_model, ensemble_pred, ensemble_pred_proba

def evaluate_model(y_test, ensemble_pred, ensemble_pred_proba):
    """Evaluate model performance and return metrics."""
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, ensemble_pred)
    precision = precision_score(y_test, ensemble_pred)
    recall = recall_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, ensemble_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'training_date': datetime.now().isoformat()
    }
    
    print(f"\n=== MODEL PERFORMANCE METRICS ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"[[{cm[0][0]}, {cm[0][1]}],")
    print(f" [{cm[1][0]}, {cm[1][1]}]]")
    
    return metrics

def save_models_and_metadata(rf_model, xgb_model, scaler, label_encoders, feature_columns, metrics):
    """Save trained models and metadata."""
    
    models_dir = '/app/backend/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save models
    joblib.dump(rf_model, f'{models_dir}/random_forest_model.pkl')
    joblib.dump(xgb_model, f'{models_dir}/xgboost_model.pkl')
    joblib.dump(scaler, f'{models_dir}/scaler.pkl')
    joblib.dump(label_encoders, f'{models_dir}/label_encoders.pkl')
    
    # Save metadata
    metadata = {
        'feature_columns': feature_columns,
        'metrics': metrics,
        'attack_scenarios': ATTACK_SCENARIOS,
        'model_info': {
            'type': 'Random Forest + XGBoost Ensemble',
            'rf_weight': 0.6,
            'xgb_weight': 0.4,
            'threshold': 0.5
        }
    }
    
    with open(f'{models_dir}/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Models and metadata saved to {models_dir}")
    return models_dir

def main():
    """Main training pipeline."""
    print("🚀 Starting Cyber Breach Forecaster Model Training...")
    
    # Load and prepare data
    X, y, feature_columns, label_encoders = prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ensemble model
    rf_model, xgb_model, ensemble_pred, ensemble_pred_proba = train_ensemble_model(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Evaluate model
    metrics = evaluate_model(y_test, ensemble_pred, ensemble_pred_proba)
    
    # Save everything
    models_dir = save_models_and_metadata(
        rf_model, xgb_model, scaler, label_encoders, feature_columns, metrics
    )
    
    print(f"\n🎯 Training completed! F1-Score: {metrics['f1_score']:.4f}")
    print(f"📁 Models saved to: {models_dir}")
    
    return metrics

if __name__ == "__main__":
    main()