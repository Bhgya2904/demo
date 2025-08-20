#!/usr/bin/env python3
"""
Additional Backend API Testing for Website Security Check and Network Analysis
"""

import requests
import json
import sys
from datetime import datetime

class AdditionalAPITester:
    def __init__(self, base_url="https://breach-forecast.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0

    def log_test(self, name: str, success: bool, details: str = ""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}: PASSED {details}")
        else:
            print(f"❌ {name}: FAILED {details}")

    def test_website_security_check(self):
        """Test website security check endpoint"""
        print("\n🔍 Testing Website Security Check Endpoint")
        
        # Test with a safe website
        safe_payload = {"url": "google.com"}
        
        try:
            response = requests.post(
                f"{self.base_url}/api/check-website",
                json=safe_payload,
                headers={'Content-Type': 'application/json'},
                timeout=60  # Longer timeout for VirusTotal API
            )
            
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                required_fields = ["url", "verdict", "total_scans", "positive_detections", "threat_explanation"]
                
                for field in required_fields:
                    if field not in data:
                        success = False
                        details += f" | Missing field: {field}"
                        break
                
                if success:
                    details += f" | Verdict: {data.get('verdict', 'Unknown')}"
                    details += f" | Scans: {data.get('positive_detections', 0)}/{data.get('total_scans', 0)}"
                    
            else:
                try:
                    error_data = response.json()
                    details += f" | Error: {error_data.get('detail', 'Unknown error')}"
                except:
                    details += f" | Response: {response.text[:200]}"
                    
        except requests.exceptions.Timeout:
            success = False
            details = "Request timeout (60s) - VirusTotal API may be slow"
        except Exception as e:
            success = False
            details = f"Error: {str(e)}"
        
        self.log_test("Website Security Check", success, details)
        return success

    def test_network_analysis(self):
        """Test network analysis endpoint"""
        print("\n🌐 Testing Network Analysis Endpoint")
        
        # Test with sample network traffic data
        network_payload = {
            "dur": 1.5,
            "proto": "tcp",
            "service": "http",
            "state": "FIN",
            "spkts": 10,
            "dpkts": 8,
            "sbytes": 500,
            "dbytes": 1200,
            "rate": 6.7,
            "sttl": 64,
            "dttl": 64,
            "sload": 333.3,
            "dload": 800.0,
            "sinpkt": 50.0,
            "dinpkt": 150.0,
            "ct_srv_src": 2,
            "ct_state_ttl": 1,
            "ct_dst_ltm": 3
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/analyze-network",
                json=network_payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                required_fields = ["prediction", "confidence", "attack_probability", "detailed_explanation"]
                
                for field in required_fields:
                    if field not in data:
                        success = False
                        details += f" | Missing field: {field}"
                        break
                
                if success:
                    details += f" | Prediction: {data.get('prediction', 'Unknown')}"
                    details += f" | Confidence: {data.get('confidence', 0)*100:.1f}%"
                    details += f" | Attack Prob: {data.get('attack_probability', 0)*100:.1f}%"
                    
            else:
                try:
                    error_data = response.json()
                    details += f" | Error: {error_data.get('detail', 'Unknown error')}"
                except:
                    details += f" | Response: {response.text[:200]}"
                    
        except Exception as e:
            success = False
            details = f"Error: {str(e)}"
        
        self.log_test("Network Analysis", success, details)
        return success

    def test_root_endpoint(self):
        """Test root endpoint"""
        print("\n🏠 Testing Root Endpoint")
        
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                if "message" in data and "status" in data:
                    details += f" | Message: {data.get('message', '')}"
                    details += f" | Status: {data.get('status', '')}"
                else:
                    success = False
                    details += " | Missing required fields"
            else:
                details += f" | Response: {response.text[:100]}"
                
        except Exception as e:
            success = False
            details = f"Error: {str(e)}"
        
        self.log_test("Root Endpoint", success, details)
        return success

    def run_additional_tests(self):
        """Run additional API tests"""
        print("🔬 Starting Additional Backend API Testing")
        print(f"📡 Testing API at: {self.base_url}")
        print("=" * 60)
        
        # Run tests
        root_success = self.test_root_endpoint()
        website_success = self.test_website_security_check()
        network_success = self.test_network_analysis()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 ADDITIONAL TEST SUMMARY")
        print("=" * 60)
        
        print(f"✅ Tests Passed: {self.tests_passed}/{self.tests_run}")
        print(f"❌ Tests Failed: {self.tests_run - self.tests_passed}/{self.tests_run}")
        print(f"📈 Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        return self.tests_passed == self.tests_run

def main():
    """Main test execution"""
    tester = AdditionalAPITester()
    
    try:
        success = tester.run_additional_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())