#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for Cyber Breach Forecaster
Tests all API endpoints with various scenarios including attack and normal traffic patterns.
"""

import requests
import json
import sys
from datetime import datetime
from typing import Dict, Any, List

class CyberBreachAPITester:
    def __init__(self, base_url="https://breach-forecast.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name: str, success: bool, details: str = ""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}: PASSED {details}")
        else:
            print(f"❌ {name}: FAILED {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

    def run_test(self, name: str, method: str, endpoint: str, expected_status: int, 
                 data: Dict = None, validate_response: callable = None) -> tuple:
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")

            success = response.status_code == expected_status
            details = f"Status: {response.status_code}"
            
            if success and response.content:
                try:
                    response_data = response.json()
                    if validate_response:
                        validation_result = validate_response(response_data)
                        if not validation_result[0]:
                            success = False
                            details += f" | Validation Failed: {validation_result[1]}"
                        else:
                            details += f" | Response Valid"
                    return success, response_data, details
                except json.JSONDecodeError:
                    details += " | Invalid JSON Response"
                    return False, {}, details
            elif not success:
                details += f" | Expected: {expected_status}"
                try:
                    error_data = response.json()
                    details += f" | Error: {error_data.get('detail', 'Unknown error')}"
                except:
                    details += f" | Response: {response.text[:100]}"
                
            return success, response.json() if success else {}, details

        except requests.exceptions.Timeout:
            details = "Request timeout (30s)"
            return False, {}, details
        except requests.exceptions.ConnectionError:
            details = "Connection error - Backend may be down"
            return False, {}, details
        except Exception as e:
            details = f"Error: {str(e)}"
            return False, {}, details

    def validate_health_response(self, data: Dict) -> tuple:
        """Validate health endpoint response"""
        required_fields = ["status", "models_loaded", "timestamp"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing field: {field}"
        
        if data["status"] != "healthy":
            return False, f"Status is not healthy: {data['status']}"
            
        return True, "Health response valid"

    def validate_metrics_response(self, data: Dict) -> tuple:
        """Validate metrics endpoint response"""
        required_fields = ["accuracy", "precision", "recall", "f1_score", "confusion_matrix", "training_date"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing field: {field}"
        
        # Check if metrics are reasonable (between 0 and 1)
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            if not (0 <= data[metric] <= 1):
                return False, f"{metric} value out of range: {data[metric]}"
                
        return True, "Metrics response valid"

    def validate_scenarios_response(self, data: List) -> tuple:
        """Validate attack scenarios response"""
        if not isinstance(data, list):
            return False, "Response should be a list"
        
        if len(data) == 0:
            return False, "No scenarios returned"
        
        for scenario in data:
            required_fields = ["scenario_name", "description", "features"]
            for field in required_fields:
                if field not in scenario:
                    return False, f"Missing field in scenario: {field}"
                    
        return True, f"Found {len(data)} scenarios"

    def validate_prediction_response(self, data: Dict) -> tuple:
        """Validate prediction endpoint response"""
        required_fields = ["prediction", "probability", "risk_level", "confidence", "detected_patterns", "timestamp"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing field: {field}"
        
        # Validate prediction values
        if data["prediction"] not in ["Safe", "Potential Breach"]:
            return False, f"Invalid prediction value: {data['prediction']}"
        
        if not (0 <= data["probability"] <= 1):
            return False, f"Probability out of range: {data['probability']}"
            
        if data["risk_level"] not in ["Low", "Medium", "High"]:
            return False, f"Invalid risk level: {data['risk_level']}"
            
        if not isinstance(data["detected_patterns"], list):
            return False, "detected_patterns should be a list"
            
        return True, f"Prediction: {data['prediction']}, Risk: {data['risk_level']}"

    def test_health_endpoint(self):
        """Test health check endpoint"""
        success, data, details = self.run_test(
            "Health Check",
            "GET",
            "api/health",
            200,
            validate_response=self.validate_health_response
        )
        self.log_test("Health Endpoint", success, details)
        return success, data

    def test_metrics_endpoint(self):
        """Test model metrics endpoint"""
        success, data, details = self.run_test(
            "Model Metrics",
            "GET", 
            "api/metrics",
            200,
            validate_response=self.validate_metrics_response
        )
        self.log_test("Metrics Endpoint", success, details)
        return success, data

    def test_attack_scenarios_endpoint(self):
        """Test attack scenarios list endpoint"""
        success, data, details = self.run_test(
            "Attack Scenarios List",
            "GET",
            "api/attack-scenarios", 
            200,
            validate_response=self.validate_scenarios_response
        )
        self.log_test("Attack Scenarios Endpoint", success, details)
        return success, data

    def test_individual_scenario(self, scenario_key: str):
        """Test individual attack scenario endpoint"""
        success, data, details = self.run_test(
            f"Individual Scenario ({scenario_key})",
            "GET",
            f"api/attack-scenarios/{scenario_key}",
            200
        )
        
        if success and data:
            # Validate scenario response structure
            required_fields = ["scenario", "description", "features"]
            for field in required_fields:
                if field not in data:
                    success = False
                    details += f" | Missing field: {field}"
                    break
            
            if success and "features" in data:
                features = data["features"]
                required_feature_fields = ["source_ip", "dest_ip", "duration", "protocol", "service"]
                for field in required_feature_fields:
                    if field not in features:
                        success = False
                        details += f" | Missing feature field: {field}"
                        break
        
        self.log_test(f"Individual Scenario ({scenario_key})", success, details)
        return success, data

    def test_ddos_prediction(self):
        """Test DDoS attack prediction"""
        ddos_payload = {
            "source_ip": "192.168.1.100",
            "dest_ip": "10.0.0.50", 
            "duration": 0.001,
            "protocol": "tcp",
            "service": "http",
            "state": "REQ",
            "source_packets": 150,
            "dest_packets": 3,
            "source_bytes": 8500,
            "dest_bytes": 120,
            "packet_rate": 85000,
            "connection_attempts": 8,
            "login_failures": 0
        }
        
        success, data, details = self.run_test(
            "DDoS Attack Prediction",
            "POST",
            "api/predict",
            200,
            data=ddos_payload,
            validate_response=self.validate_prediction_response
        )
        self.log_test("DDoS Attack Prediction", success, details)
        return success, data

    def test_normal_traffic_prediction(self):
        """Test normal traffic prediction"""
        normal_payload = {
            "source_ip": "192.168.1.50",
            "dest_ip": "10.0.0.80",
            "duration": 3.2,
            "protocol": "tcp",
            "service": "http",
            "state": "FIN", 
            "source_packets": 8,
            "dest_packets": 12,
            "source_bytes": 650,
            "dest_bytes": 1800,
            "packet_rate": 6.8,
            "connection_attempts": 2,
            "login_failures": 0
        }
        
        success, data, details = self.run_test(
            "Normal Traffic Prediction",
            "POST",
            "api/predict",
            200,
            data=normal_payload,
            validate_response=self.validate_prediction_response
        )
        self.log_test("Normal Traffic Prediction", success, details)
        return success, data

    def test_invalid_prediction_data(self):
        """Test prediction with invalid data"""
        invalid_payload = {
            "source_ip": "invalid_ip",
            "dest_ip": "10.0.0.50",
            "duration": -1,  # Invalid negative duration
            "protocol": "invalid_protocol",
            "service": "http",
            "state": "REQ",
            "source_packets": -5,  # Invalid negative packets
            "dest_packets": 3,
            "source_bytes": 8500,
            "dest_bytes": 120,
            "packet_rate": 85000,
            "connection_attempts": 8,
            "login_failures": 0
        }
        
        success, data, details = self.run_test(
            "Invalid Prediction Data",
            "POST", 
            "api/predict",
            422,  # Expecting validation error
        )
        self.log_test("Invalid Data Handling", success, details)
        return success, data

    def run_comprehensive_tests(self):
        """Run all backend API tests"""
        print("🚀 Starting Comprehensive Backend API Testing")
        print(f"📡 Testing API at: {self.base_url}")
        print("=" * 60)
        
        # Test basic endpoints
        health_success, health_data = self.test_health_endpoint()
        metrics_success, metrics_data = self.test_metrics_endpoint()
        scenarios_success, scenarios_data = self.test_attack_scenarios_endpoint()
        
        # Test individual scenarios
        scenario_keys = ["ddos", "port_scan", "brute_force", "sql_injection", "malware", "normal"]
        scenario_results = {}
        
        for key in scenario_keys:
            success, data = self.test_individual_scenario(key)
            scenario_results[key] = (success, data)
        
        # Test prediction endpoints
        ddos_success, ddos_data = self.test_ddos_prediction()
        normal_success, normal_data = self.test_normal_traffic_prediction()
        invalid_success, invalid_data = self.test_invalid_prediction_data()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        print(f"✅ Tests Passed: {self.tests_passed}/{self.tests_run}")
        print(f"❌ Tests Failed: {self.tests_run - self.tests_passed}/{self.tests_run}")
        print(f"📈 Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        # Detailed results
        if health_success and health_data:
            print(f"\n🏥 Backend Health: {health_data.get('status', 'Unknown')}")
            print(f"🤖 Models Loaded: {health_data.get('models_loaded', 'Unknown')}")
        
        if metrics_success and metrics_data:
            print(f"\n📊 Model Performance:")
            print(f"   Accuracy: {metrics_data.get('accuracy', 0)*100:.1f}%")
            print(f"   Precision: {metrics_data.get('precision', 0)*100:.1f}%")
            print(f"   Recall: {metrics_data.get('recall', 0)*100:.1f}%")
            print(f"   F1-Score: {metrics_data.get('f1_score', 0)*100:.1f}%")
        
        if ddos_success and ddos_data:
            print(f"\n🔴 DDoS Attack Prediction:")
            print(f"   Result: {ddos_data.get('prediction', 'Unknown')}")
            print(f"   Risk Level: {ddos_data.get('risk_level', 'Unknown')}")
            print(f"   Probability: {ddos_data.get('probability', 0)*100:.1f}%")
        
        if normal_success and normal_data:
            print(f"\n🟢 Normal Traffic Prediction:")
            print(f"   Result: {normal_data.get('prediction', 'Unknown')}")
            print(f"   Risk Level: {normal_data.get('risk_level', 'Unknown')}")
            print(f"   Probability: {normal_data.get('probability', 0)*100:.1f}%")
        
        # Check for critical failures
        critical_failures = []
        if not health_success:
            critical_failures.append("Health endpoint failed")
        if not ddos_success:
            critical_failures.append("DDoS prediction failed")
        if not normal_success:
            critical_failures.append("Normal traffic prediction failed")
            
        if critical_failures:
            print(f"\n🚨 CRITICAL FAILURES:")
            for failure in critical_failures:
                print(f"   • {failure}")
            return False
        
        print(f"\n🎉 Backend API testing completed successfully!")
        return self.tests_passed == self.tests_run

def main():
    """Main test execution"""
    tester = CyberBreachAPITester()
    
    try:
        success = tester.run_comprehensive_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())