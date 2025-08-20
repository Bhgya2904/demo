# Cyber Breach Forecaster - Enhanced Edition

A comprehensive data-driven system that leverages ensemble machine learning models to predict hacking incidents and analyze cybersecurity threats in real-time.

## 🚀 Features

### Core Capabilities
- **Network Traffic Analysis**: Real-time breach prediction using UNSW-NB15 dataset with 95%+ accuracy
- **Website Security Check**: URL threat analysis using VirusTotal API integration
- **Enhanced Network Analysis**: Deep traffic analysis with comprehensive feature support
- **Model Performance Visualization**: Detailed accuracy metrics with confusion matrix and performance graphs

### Technology Stack
- **Frontend**: React.js with Tailwind CSS and Radix UI components
- **Backend**: FastAPI with Python
- **Machine Learning**: Ensemble approach using Random Forest (60%) + XGBoost (40%)
- **Dataset**: UNSW-NB15 network intrusion detection dataset
- **External APIs**: VirusTotal API for URL security analysis

## 🏗️ Project Structure

```
/app/
├── backend/                    # FastAPI backend
│   ├── models/                # Pre-trained ML models
│   ├── server.py              # Main FastAPI application
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/        # UI components
│   │   ├── App.js            # Main React application
│   │   └── index.js          # Entry point
│   ├── package.json          # Node dependencies
│   └── .env                  # Frontend environment
├── train_model.py            # Model training script
├── unsw_nb15_training.csv    # Training dataset
└── README.md                 # This file
```

## 🔧 Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- Yarn package manager

### Environment Setup

1. **Backend Configuration**
   ```bash
   cd /app/backend
   pip install -r requirements.txt
   ```

2. **Frontend Setup**
   ```bash
   cd /app/frontend
   yarn install
   ```

3. **Environment Variables**
   
   Backend (`.env`):
   ```env
   VIRUSTOTAL_API_KEY=your_api_key_here
   MONGO_URL=mongodb://localhost:27017
   DB_NAME=test_database
   CORS_ORIGINS=*
   ```
   
   Frontend (`.env`):
   ```env
   REACT_APP_BACKEND_URL=your_backend_url_here
   ```

### API Key Setup

**VirusTotal API Key** (Required for Website Security Check):
1. Visit https://www.virustotal.com/
2. Create a free account
3. Navigate to your profile → API Key
4. Copy the key to your backend `.env` file

## 🚀 Running the Application

### Using Supervisor (Recommended)
```bash
sudo supervisorctl restart all
sudo supervisorctl status
```

### Manual Start
```bash
# Terminal 1 - Backend
cd /app/backend
uvicorn server:app --host 0.0.0.0 --port 8001

# Terminal 2 - Frontend
cd /app/frontend
yarn start
```

## 📊 API Endpoints

### Network Traffic Prediction
- **POST** `/api/predict` - Predict cyber breach from network features
- **GET** `/api/attack-scenarios` - Get predefined attack scenarios
- **GET** `/api/attack-scenarios/{key}` - Get specific scenario features

### Website Security Analysis  
- **POST** `/api/check-website` - Analyze URL security using VirusTotal

### Enhanced Network Analysis
- **POST** `/api/analyze-network` - Deep network traffic analysis with UNSW-NB15

### Model Performance
- **GET** `/api/metrics` - Get model performance metrics
- **GET** `/api/health` - API health check

## 🔍 Usage Examples

### Network Traffic Prediction
```javascript
const response = await fetch('/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    source_ip: "192.168.1.100",
    dest_ip: "10.0.0.50",
    duration: 1.5,
    protocol: "tcp",
    service: "http",
    state: "FIN",
    source_packets: 10,
    dest_packets: 8,
    source_bytes: 500,
    dest_bytes: 400,
    packet_rate: 6.67,
    connection_attempts: 1,
    login_failures: 0
  })
});
```

### Website Security Check
```javascript
const response = await fetch('/api/check-website', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    url: "https://example.com"
  })
});
```

## 🎯 Model Performance

- **Accuracy**: 95%+ on UNSW-NB15 dataset
- **Precision**: 94%+
- **Recall**: 96%+
- **F1-Score**: 95%+

### Attack Detection Categories
- DDoS Attacks
- Port Scanning
- Brute Force Attacks
- SQL Injection
- Malware Communication
- Reconnaissance
- Exploitation Attempts

## 🖥️ User Interface

### Available Pages
1. **Home** (`/`) - Landing page with feature overview
2. **Network Demo** (`/demo`) - Basic network traffic prediction
3. **Website Check** (`/website-check`) - URL security analysis
4. **Network Analysis** (`/network-analysis`) - Enhanced traffic analysis
5. **Accuracy** (`/accuracy`) - Detailed model performance metrics
6. **About** (`/about`) - Project information
7. **Research** (`/research`) - Dataset and methodology details
8. **Contact** (`/contact`) - Team information

## 🧪 Testing

### Backend API Testing
```bash
# Health check
curl https://your-domain.com/api/health

# Test prediction endpoint
curl -X POST https://your-domain.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "source_ip": "192.168.1.100",
    "dest_ip": "10.0.0.50",
    "duration": 1.5,
    "protocol": "tcp",
    "service": "http",
    "state": "FIN",
    "source_packets": 10,
    "dest_packets": 8,
    "source_bytes": 500,
    "dest_bytes": 400,
    "packet_rate": 6.67,
    "connection_attempts": 1,
    "login_failures": 0
  }'
```

### Frontend Testing
- Access the application through your configured domain
- Test all navigation links
- Try different attack scenarios in the demo
- Verify website security check with known URLs
- Check model accuracy visualizations

## 📈 Machine Learning Pipeline

### Data Preprocessing
1. **Data Cleaning**: Remove duplicates, handle missing values
2. **Feature Engineering**: Calculate derived metrics and patterns
3. **Encoding**: Label encoding for categorical variables
4. **Scaling**: StandardScaler normalization

### Model Architecture
- **Ensemble Approach**: Combines Random Forest (60%) and XGBoost (40%)
- **Feature Space**: 45 network traffic features from UNSW-NB15
- **Training Data**: 175,000+ network flow records
- **Cross-Validation**: K-fold validation for robust performance

### Feature Importance
Top contributing features for breach detection:
- Connection duration and packet rates
- Bytes transferred patterns
- Protocol and service combinations
- Connection state analysis
- Time-based traffic patterns

## 🔒 Security Considerations

- API key management through environment variables
- CORS configuration for secure cross-origin requests
- Input validation and sanitization
- Rate limiting considerations for production deployment
- SSL/TLS encryption for data transmission

## 🚀 Deployment

### Production Checklist
- [ ] Set production environment variables
- [ ] Configure SSL certificates
- [ ] Set up monitoring and logging
- [ ] Implement rate limiting
- [ ] Configure backup strategies
- [ ] Set up error tracking

### Performance Optimization
- Model prediction caching
- API response compression
- Static asset optimization
- Database query optimization

## 👥 Team

**Final Year Project 2024-25**

- **M. Bhagyasri** - Team Member
- **K. Rakesh** - Team Member  
- **G. Ramesh** - Team Member

**Faculty Guide**: Santharaju Sir  
**Head of Department**: Dr. Tamilkodi

## 📄 License

This project is developed as part of an academic final year project. All rights reserved.

## 🤝 Contributing

This is an academic project. For questions or collaboration:
1. Use the contact form in the application
2. Reach out to the project team
3. Contact the faculty guide

## 🔧 Troubleshooting

### Common Issues

**Models Not Loading**:
- Ensure all model files are present in `/app/backend/models/`
- Check Python dependencies are installed
- Verify backend logs for loading errors

**VirusTotal API Issues**:
- Verify API key is correctly set in backend `.env`
- Check API quota limits
- Ensure network connectivity to VirusTotal

**Frontend Build Issues**:
- Run `yarn install` to ensure dependencies
- Check for port conflicts
- Verify backend URL configuration

**Database Connection Issues**:
- Ensure MongoDB is running
- Check connection string in environment variables
- Verify database permissions

### Log Locations
- Backend: `/var/log/supervisor/backend.*.log`
- Frontend: Browser developer console
- System: `/var/log/supervisor/`

## 📚 References

1. UNSW-NB15 Dataset - Australian Centre for Cyber Security
2. VirusTotal API Documentation
3. FastAPI Framework Documentation
4. React.js Documentation
5. Scikit-learn and XGBoost Documentation

---

**Note**: This enhanced edition integrates multiple cybersecurity analysis tools into a unified platform, providing comprehensive threat detection and analysis capabilities for research and educational purposes.