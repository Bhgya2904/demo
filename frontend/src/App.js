import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import axios from 'axios';
import './App.css';

// Import components
import { Button } from './components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Badge } from './components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Alert, AlertDescription } from './components/ui/alert';
import { Separator } from './components/ui/separator';

// Icons
import { Shield, Zap, Target, Globe, Lock, AlertTriangle, CheckCircle, XCircle, Activity, Users, BookOpen, Mail, Github, ExternalLink } from 'lucide-react';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-emerald-950">
        <Navigation />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/demo" element={<DemoPage />} />
          <Route path="/research" element={<ResearchPage />} />
          <Route path="/contact" element={<ContactPage />} />
        </Routes>
        <Footer />
      </div>
    </Router>
  );
}

function Navigation() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 bg-slate-950/95 backdrop-blur-xl border-b border-emerald-500/20">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30 group-hover:bg-emerald-500/20 transition-colors">
              <Shield className="w-6 h-6 text-emerald-400" />
            </div>
            <span className="text-xl font-bold text-white">Cyber Breach Forecaster</span>
          </Link>
          
          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <NavLink to="/">Home</NavLink>
            <NavLink to="/about">About</NavLink>
            <NavLink to="/demo">Demo</NavLink>
            <NavLink to="/research">Research</NavLink>
            <NavLink to="/contact">Contact</NavLink>
          </div>

          {/* Mobile menu button */}
          <button 
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden p-2 rounded-lg text-emerald-400 hover:bg-emerald-500/10"
          >
            <div className="w-6 h-6 flex flex-col justify-center space-y-1">
              <div className={`h-0.5 bg-current transition-transform ${isOpen ? 'rotate-45 translate-y-1.5' : ''}`}></div>
              <div className={`h-0.5 bg-current transition-opacity ${isOpen ? 'opacity-0' : ''}`}></div>
              <div className={`h-0.5 bg-current transition-transform ${isOpen ? '-rotate-45 -translate-y-1.5' : ''}`}></div>
            </div>
          </button>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <div className="md:hidden pb-4 border-t border-emerald-500/20 mt-2">
            <div className="flex flex-col space-y-2">
              <MobileNavLink to="/" onClick={() => setIsOpen(false)}>Home</MobileNavLink>
              <MobileNavLink to="/about" onClick={() => setIsOpen(false)}>About</MobileNavLink>
              <MobileNavLink to="/demo" onClick={() => setIsOpen(false)}>Demo</MobileNavLink>
              <MobileNavLink to="/research" onClick={() => setIsOpen(false)}>Research</MobileNavLink>
              <MobileNavLink to="/contact" onClick={() => setIsOpen(false)}>Contact</MobileNavLink>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}

function NavLink({ to, children }) {
  return (
    <Link 
      to={to} 
      className="text-slate-300 hover:text-emerald-400 transition-colors font-medium"
    >
      {children}
    </Link>
  );
}

function MobileNavLink({ to, children, onClick }) {
  return (
    <Link 
      to={to} 
      onClick={onClick}
      className="block py-2 px-4 text-slate-300 hover:text-emerald-400 hover:bg-emerald-500/10 rounded-lg transition-colors"
    >
      {children}
    </Link>
  );
}

function LandingPage() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative py-24 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-sm font-medium mb-8">
            <Zap className="w-4 h-4 mr-2" />
            Final Year Project 2024-25
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
            Cyber Breach
            <span className="bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent">
              {" "}Forecaster
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl text-slate-300 mb-12 max-w-4xl mx-auto leading-relaxed">
            Predicting Cyber Attacks Using Machine Learning
          </p>
          
          <p className="text-lg text-slate-400 mb-12 max-w-3xl mx-auto">
            An advanced data-driven system that leverages Random Forest and XGBoost ensemble models 
            to predict hacking incidents in real-time, protecting organizations from cyber threats.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button 
              asChild 
              size="lg"
              className="bg-emerald-600 hover:bg-emerald-700 text-white px-8 py-3 text-lg font-semibold"
            >
              <Link to="/demo">Try Demo</Link>
            </Button>
            <Button 
              asChild 
              variant="outline"
              size="lg" 
              className="border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/10 px-8 py-3 text-lg"
            >
              <Link to="/research">View Research</Link>
            </Button>
            <Button 
              asChild 
              variant="ghost"
              size="lg"
              className="text-slate-300 hover:text-emerald-400 px-8 py-3 text-lg"
            >
              <Link to="/about">Learn More</Link>
            </Button>
          </div>
        </div>

        {/* Floating Animation Elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
          <div className="absolute top-3/4 right-1/3 w-1 h-1 bg-cyan-400 rounded-full animate-ping"></div>
          <div className="absolute bottom-1/4 left-1/2 w-3 h-3 bg-blue-500 rounded-full animate-bounce"></div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-white text-center mb-16">
            Advanced Threat Detection Capabilities
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            <FeatureCard 
              icon={<Target className="w-8 h-8" />}
              title="Real-Time Detection"
              description="Instant analysis of network traffic patterns using ensemble machine learning models"
            />
            <FeatureCard 
              icon={<Activity className="w-8 h-8" />}
              title="High Accuracy"
              description="Achieves 95%+ accuracy with Random Forest + XGBoost ensemble on UNSW-NB15 dataset"
            />
            <FeatureCard 
              icon={<Globe className="w-8 h-8" />}
              title="Multiple Attack Types"
              description="Detects DDoS, Port Scanning, Brute Force, SQL Injection, and Malware communication"
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <Card className="bg-slate-900/50 border-emerald-500/30 p-8">
            <h3 className="text-3xl font-bold text-white mb-4">
              Ready to Test Our Cyber Breach Detection?
            </h3>
            <p className="text-slate-300 mb-8 text-lg">
              Experience our machine learning model in action with real-time predictions
            </p>
            <Button 
              asChild
              size="lg"
              className="bg-emerald-600 hover:bg-emerald-700 text-white px-8 py-3"
            >
              <Link to="/demo">Launch Demo</Link>
            </Button>
          </Card>
        </div>
      </section>
    </div>
  );
}

function FeatureCard({ icon, title, description }) {
  return (
    <Card className="bg-slate-900/30 border-slate-700/50 hover:border-emerald-500/50 transition-all duration-300 group">
      <CardContent className="p-6">
        <div className="text-emerald-400 mb-4 group-hover:scale-110 transition-transform">
          {icon}
        </div>
        <h3 className="text-xl font-semibold text-white mb-3">{title}</h3>
        <p className="text-slate-400 leading-relaxed">{description}</p>
      </CardContent>
    </Card>
  );
}

function AboutPage() {
  return (
    <div className="min-h-screen py-20 px-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-white text-center mb-16">About This Project</h1>
        
        <Card className="bg-slate-900/50 border-emerald-500/30 mb-12">
          <CardHeader>
            <CardTitle className="text-2xl text-emerald-400">Project Abstract</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6 text-slate-300">
            <div>
              <h3 className="text-xl font-semibold text-white mb-3">Problem Statement</h3>
              <p>
                Cybersecurity threats are increasing exponentially, with organizations facing sophisticated 
                attacks daily. Traditional signature-based detection systems fail to identify new and 
                evolving attack patterns, leaving critical infrastructure vulnerable.
              </p>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold text-white mb-3">Motivation</h3>
              <p>
                The need for proactive, intelligent threat detection systems that can predict and 
                prevent cyber attacks before they cause damage. Machine learning offers the capability 
                to identify patterns in network behavior that humans might miss.
              </p>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold text-white mb-3">Solution</h3>
              <p>
                We developed an ensemble machine learning system using Random Forest and XGBoost 
                algorithms trained on the comprehensive UNSW-NB15 dataset. The system analyzes 
                network traffic features to predict potential security breaches in real-time.
              </p>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold text-white mb-3">Goal</h3>
              <p>
                To create an accurate, fast, and reliable cyber threat prediction system that can 
                be deployed in enterprise environments to enhance cybersecurity posture and reduce 
                the risk of successful cyber attacks.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Team Section */}
        <Card className="bg-slate-900/50 border-emerald-500/30">
          <CardHeader>
            <CardTitle className="text-2xl text-emerald-400">Project Team</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-white mb-4">Team Members</h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <Users className="w-5 h-5 text-emerald-400" />
                    <span className="text-slate-300">M. Bhagyasri</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <Users className="w-5 h-5 text-emerald-400" />
                    <span className="text-slate-300">K. Rakesh</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <Users className="w-5 h-5 text-emerald-400" />
                    <span className="text-slate-300">G. Ramesh</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-white mb-4">Faculty Guide</h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <BookOpen className="w-5 h-5 text-emerald-400" />
                    <div>
                      <p className="text-slate-300">Project Guide: Santharaju Sir</p>
                      <p className="text-slate-300">HOD: Dr. Tamilkodi</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function DemoPage() {
  const [formData, setFormData] = useState({
    source_ip: '192.168.1.100',
    dest_ip: '10.0.0.50',
    duration: '',
    protocol: '',
    service: '',
    state: '',
    source_packets: '',
    dest_packets: '',
    source_bytes: '',
    dest_bytes: '',
    packet_rate: '',
    connection_attempts: '',
    login_failures: ''
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [scenarios, setScenarios] = useState([]);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetchScenarios();
    fetchMetrics();
  }, []);

  const fetchScenarios = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/attack-scenarios`);
      setScenarios(response.data);
    } catch (error) {
      console.error('Error fetching scenarios:', error);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/metrics`);
      setMetrics(response.data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const loadScenario = async (scenarioKey) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/attack-scenarios/${scenarioKey}`);
      const scenarioData = response.data.features;
      
      setFormData({
        source_ip: scenarioData.source_ip,
        dest_ip: scenarioData.dest_ip,
        duration: scenarioData.duration.toString(),
        protocol: scenarioData.protocol,
        service: scenarioData.service,
        state: scenarioData.state,
        source_packets: scenarioData.source_packets.toString(),
        dest_packets: scenarioData.dest_packets.toString(),
        source_bytes: scenarioData.source_bytes.toString(),
        dest_bytes: scenarioData.dest_bytes.toString(),
        packet_rate: scenarioData.packet_rate.toString(),
        connection_attempts: scenarioData.connection_attempts.toString(),
        login_failures: scenarioData.login_failures.toString()
      });
    } catch (error) {
      console.error('Error loading scenario:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const payload = {
        source_ip: formData.source_ip,
        dest_ip: formData.dest_ip,
        duration: parseFloat(formData.duration),
        protocol: formData.protocol,
        service: formData.service,
        state: formData.state,
        source_packets: parseInt(formData.source_packets),
        dest_packets: parseInt(formData.dest_packets),
        source_bytes: parseInt(formData.source_bytes),
        dest_bytes: parseInt(formData.dest_bytes),
        packet_rate: parseFloat(formData.packet_rate),
        connection_attempts: parseInt(formData.connection_attempts),
        login_failures: parseInt(formData.login_failures)
      };

      const response = await axios.post(`${API_BASE_URL}/api/predict`, payload);
      setPrediction(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
      setPrediction({
        prediction: 'Error',
        probability: 0,
        risk_level: 'Unknown',
        confidence: 0,
        detected_patterns: ['API Error - Please check backend connection'],
        timestamp: new Date().toISOString()
      });
    }
    
    setLoading(false);
  };

  return (
    <div className="min-h-screen py-20 px-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-white text-center mb-16">
          Breach Detection Demo
        </h1>
        
        <div className="grid lg:grid-cols-2 gap-12">
          {/* Input Form */}
          <div>
            <Card className="bg-slate-900/50 border-emerald-500/30 mb-8">
              <CardHeader>
                <CardTitle className="text-2xl text-emerald-400 flex items-center">
                  <Target className="w-6 h-6 mr-2" />
                  Input Network Features
                </CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="source_ip" className="text-slate-300">Source IP</Label>
                      <Input
                        id="source_ip"
                        type="text"
                        value={formData.source_ip}
                        onChange={(e) => handleInputChange('source_ip', e.target.value)}
                        className="bg-slate-800 border-slate-600 text-white"
                        required
                      />
                    </div>
                    <div>
                      <Label htmlFor="dest_ip" className="text-slate-300">Destination IP</Label>
                      <Input
                        id="dest_ip"
                        type="text"
                        value={formData.dest_ip}
                        onChange={(e) => handleInputChange('dest_ip', e.target.value)}
                        className="bg-slate-800 border-slate-600 text-white"
                        required
                      />
                    </div>
                  </div>

                  <div className="grid md:grid-cols-3 gap-4">
                    <div>
                      <Label htmlFor="duration" className="text-slate-300">Duration (seconds)</Label>
                      <Input
                        id="duration"
                        type="number"
                        step="0.001"
                        value={formData.duration}
                        onChange={(e) => handleInputChange('duration', e.target.value)}
                        className="bg-slate-800 border-slate-600 text-white"
                        required
                      />
                    </div>
                    <div>
                      <Label htmlFor="protocol" className="text-slate-300">Protocol</Label>
                      <Select value={formData.protocol} onValueChange={(value) => handleInputChange('protocol', value)}>
                        <SelectTrigger className="bg-slate-800 border-slate-600 text-white">
                          <SelectValue placeholder="Select protocol" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="tcp">TCP</SelectItem>
                          <SelectItem value="udp">UDP</SelectItem>
                          <SelectItem value="icmp">ICMP</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="service" className="text-slate-300">Service</Label>
                      <Select value={formData.service} onValueChange={(value) => handleInputChange('service', value)}>
                        <SelectTrigger className="bg-slate-800 border-slate-600 text-white">
                          <SelectValue placeholder="Select service" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="http">HTTP</SelectItem>
                          <SelectItem value="https">HTTPS</SelectItem>
                          <SelectItem value="ssh">SSH</SelectItem>
                          <SelectItem value="ftp">FTP</SelectItem>
                          <SelectItem value="smtp">SMTP</SelectItem>
                          <SelectItem value="-">Unknown</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="grid md:grid-cols-4 gap-4">
                    <div>
                      <Label htmlFor="source_packets" className="text-slate-300">Source Packets</Label>
                      <Input
                        id="source_packets"
                        type="number"
                        value={formData.source_packets}
                        onChange={(e) => handleInputChange('source_packets', e.target.value)}
                        className="bg-slate-800 border-slate-600 text-white"
                        required
                      />
                    </div>
                    <div>
                      <Label htmlFor="dest_packets" className="text-slate-300">Dest Packets</Label>
                      <Input
                        id="dest_packets"
                        type="number"
                        value={formData.dest_packets}
                        onChange={(e) => handleInputChange('dest_packets', e.target.value)}
                        className="bg-slate-800 border-slate-600 text-white"
                        required
                      />
                    </div>
                    <div>
                      <Label htmlFor="source_bytes" className="text-slate-300">Source Bytes</Label>
                      <Input
                        id="source_bytes"
                        type="number"
                        value={formData.source_bytes}
                        onChange={(e) => handleInputChange('source_bytes', e.target.value)}
                        className="bg-slate-800 border-slate-600 text-white"
                        required
                      />
                    </div>
                    <div>
                      <Label htmlFor="dest_bytes" className="text-slate-300">Dest Bytes</Label>
                      <Input
                        id="dest_bytes"
                        type="number"
                        value={formData.dest_bytes}
                        onChange={(e) => handleInputChange('dest_bytes', e.target.value)}
                        className="bg-slate-800 border-slate-600 text-white"
                        required
                      />
                    </div>
                  </div>

                  <div className="grid md:grid-cols-3 gap-4">
                    <div>
                      <Label htmlFor="packet_rate" className="text-slate-300">Packet Rate</Label>
                      <Input
                        id="packet_rate"
                        type="number"
                        step="0.1"
                        value={formData.packet_rate}
                        onChange={(e) => handleInputChange('packet_rate', e.target.value)}
                        className="bg-slate-800 border-slate-600 text-white"
                        required
                      />
                    </div>
                    <div>
                      <Label htmlFor="connection_attempts" className="text-slate-300">Connection Attempts</Label>
                      <Input
                        id="connection_attempts"
                        type="number"
                        value={formData.connection_attempts}
                        onChange={(e) => handleInputChange('connection_attempts', e.target.value)}
                        className="bg-slate-800 border-slate-600 text-white"
                        required
                      />
                    </div>
                    <div>
                      <Label htmlFor="login_failures" className="text-slate-300">Login Failures</Label>
                      <Input
                        id="login_failures"
                        type="number"
                        value={formData.login_failures}
                        onChange={(e) => handleInputChange('login_failures', e.target.value)}
                        className="bg-slate-800 border-slate-600 text-white"
                        required
                      />
                    </div>
                  </div>

                  <Button 
                    type="submit" 
                    disabled={loading}
                    className="w-full bg-emerald-600 hover:bg-emerald-700 text-white py-3"
                  >
                    {loading ? 'Analyzing...' : 'Predict Breach'}
                  </Button>
                </form>
              </CardContent>
            </Card>

            {/* Attack Scenarios */}
            <Card className="bg-slate-900/50 border-emerald-500/30">
              <CardHeader>
                <CardTitle className="text-xl text-emerald-400">Quick Test Scenarios</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  <Button 
                    variant="outline" 
                    onClick={() => loadScenario('ddos')}
                    className="border-red-500/50 text-red-400 hover:bg-red-500/10 text-sm"
                  >
                    DDoS Attack
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => loadScenario('port_scan')}
                    className="border-orange-500/50 text-orange-400 hover:bg-orange-500/10 text-sm"
                  >
                    Port Scan
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => loadScenario('brute_force')}
                    className="border-yellow-500/50 text-yellow-400 hover:bg-yellow-500/10 text-sm"
                  >
                    Brute Force
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => loadScenario('sql_injection')}
                    className="border-purple-500/50 text-purple-400 hover:bg-purple-500/10 text-sm"
                  >
                    SQL Injection
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => loadScenario('malware')}
                    className="border-pink-500/50 text-pink-400 hover:bg-pink-500/10 text-sm"
                  >
                    Malware
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => loadScenario('normal')}
                    className="border-green-500/50 text-green-400 hover:bg-green-500/10 text-sm"
                  >
                    Normal Traffic
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Results */}
          <div className="space-y-8">
            {/* Prediction Result */}
            {prediction && (
              <Card className="bg-slate-900/50 border-emerald-500/30">
                <CardHeader>
                  <CardTitle className="text-2xl text-emerald-400 flex items-center">
                    <Activity className="w-6 h-6 mr-2" />
                    Prediction Result
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Traffic Light Indicator */}
                  <div className="flex items-center justify-center">
                    <div className="flex flex-col items-center space-y-4">
                      <div className="relative">
                        <div className="w-24 h-32 bg-slate-800 rounded-xl border-2 border-slate-600 flex flex-col items-center justify-around py-3">
                          <div className={`w-6 h-6 rounded-full ${prediction.risk_level === 'Low' ? 'bg-green-500 shadow-green-500/50 shadow-lg' : 'bg-gray-700'}`}></div>
                          <div className={`w-6 h-6 rounded-full ${prediction.risk_level === 'Medium' ? 'bg-yellow-500 shadow-yellow-500/50 shadow-lg' : 'bg-gray-700'}`}></div>
                          <div className={`w-6 h-6 rounded-full ${prediction.risk_level === 'High' ? 'bg-red-500 shadow-red-500/50 shadow-lg' : 'bg-gray-700'}`}></div>
                        </div>
                      </div>
                      <Badge 
                        className={`text-sm px-3 py-1 ${
                          prediction.prediction === 'Safe' 
                            ? 'bg-green-500/20 text-green-400 border-green-500/50' 
                            : 'bg-red-500/20 text-red-400 border-red-500/50'
                        }`}
                      >
                        {prediction.prediction}
                      </Badge>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-slate-300">Threat Probability:</span>
                      <span className="text-white font-mono">{(prediction.probability * 100).toFixed(2)}%</span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-slate-300">Risk Level:</span>
                      <Badge className={`${
                        prediction.risk_level === 'Low' 
                          ? 'bg-green-500/20 text-green-400' 
                          : prediction.risk_level === 'Medium'
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {prediction.risk_level}
                      </Badge>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-slate-300">Confidence:</span>
                      <span className="text-white font-mono">{(prediction.confidence * 100).toFixed(2)}%</span>
                    </div>

                    {prediction.detected_patterns.length > 0 && (
                      <div>
                        <span className="text-slate-300 block mb-2">Detected Patterns:</span>
                        <div className="space-y-2">
                          {prediction.detected_patterns.map((pattern, index) => (
                            <Badge key={index} variant="outline" className="text-xs mr-2 mb-2 border-emerald-500/50 text-emerald-400">
                              {pattern}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Model Performance Metrics */}
            {metrics && (
              <Card className="bg-slate-900/50 border-emerald-500/30">
                <CardHeader>
                  <CardTitle className="text-xl text-emerald-400">Model Performance</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-white">{(metrics.accuracy * 100).toFixed(1)}%</div>
                      <div className="text-sm text-slate-400">Accuracy</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-white">{(metrics.precision * 100).toFixed(1)}%</div>
                      <div className="text-sm text-slate-400">Precision</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-white">{(metrics.recall * 100).toFixed(1)}%</div>
                      <div className="text-sm text-slate-400">Recall</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-white">{(metrics.f1_score * 100).toFixed(1)}%</div>
                      <div className="text-sm text-slate-400">F1-Score</div>
                    </div>
                  </div>
                  
                  {metrics.confusion_matrix && (
                    <div className="mt-6">
                      <h4 className="text-sm font-medium text-slate-300 mb-3">Confusion Matrix</h4>
                      <div className="grid grid-cols-2 gap-2 max-w-48 mx-auto">
                        <div className="bg-slate-800 p-3 rounded text-center">
                          <div className="text-lg font-bold text-green-400">{metrics.confusion_matrix[0][0]}</div>
                          <div className="text-xs text-slate-400">True Neg</div>
                        </div>
                        <div className="bg-slate-800 p-3 rounded text-center">
                          <div className="text-lg font-bold text-red-400">{metrics.confusion_matrix[0][1]}</div>
                          <div className="text-xs text-slate-400">False Pos</div>
                        </div>
                        <div className="bg-slate-800 p-3 rounded text-center">
                          <div className="text-lg font-bold text-red-400">{metrics.confusion_matrix[1][0]}</div>
                          <div className="text-xs text-slate-400">False Neg</div>
                        </div>
                        <div className="bg-slate-800 p-3 rounded text-center">
                          <div className="text-lg font-bold text-green-400">{metrics.confusion_matrix[1][1]}</div>
                          <div className="text-xs text-slate-400">True Pos</div>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function ResearchPage() {
  return (
    <div className="min-h-screen py-20 px-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-white text-center mb-16">Research & Methodology</h1>
        
        <div className="space-y-8">
          <Card className="bg-slate-900/50 border-emerald-500/30">
            <CardHeader>
              <CardTitle className="text-2xl text-emerald-400">Dataset: UNSW-NB15</CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p>
                The UNSW-NB15 dataset is a comprehensive network intrusion detection dataset created by 
                the Australian Centre for Cyber Security (ACCS). It contains realistic network traffic 
                with both normal activities and synthetic attack behaviors.
              </p>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-white mb-2">Dataset Features:</h4>
                  <ul className="space-y-1 text-sm">
                    <li>• 175,000+ network flow records</li>
                    <li>• 45 network traffic features</li>
                    <li>• 9 different attack categories</li>
                    <li>• Balanced normal vs malicious traffic</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold text-white mb-2">Attack Categories:</h4>
                  <ul className="space-y-1 text-sm">
                    <li>• Analysis, Backdoor, DoS</li>
                    <li>• Exploits, Fuzzers, Generic</li>
                    <li>• Reconnaissance, Shellcode, Worms</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-emerald-500/30">
            <CardHeader>
              <CardTitle className="text-2xl text-emerald-400">Preprocessing Pipeline</CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300">
              <Tabs defaultValue="data-cleaning" className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="data-cleaning">Data Cleaning</TabsTrigger>
                  <TabsTrigger value="feature-engineering">Feature Engineering</TabsTrigger>
                  <TabsTrigger value="scaling">Scaling & Encoding</TabsTrigger>
                </TabsList>
                <TabsContent value="data-cleaning" className="space-y-4">
                  <h4 className="font-semibold text-white">Data Cleaning Steps:</h4>
                  <ul className="space-y-2">
                    <li>• Remove duplicate records and handle missing values</li>
                    <li>• Filter out incomplete network flows</li>
                    <li>• Validate IP addresses and port numbers</li>
                    <li>• Remove outliers using IQR method</li>
                  </ul>
                </TabsContent>
                <TabsContent value="feature-engineering" className="space-y-4">
                  <h4 className="font-semibold text-white">Feature Engineering:</h4>
                  <ul className="space-y-2">
                    <li>• Calculate packet rates and data transfer speeds</li>
                    <li>• Extract timing patterns and connection states</li>
                    <li>• Create derived features from raw network metrics</li>
                    <li>• Aggregate connection-level statistics</li>
                  </ul>
                </TabsContent>
                <TabsContent value="scaling" className="space-y-4">
                  <h4 className="font-semibold text-white">Scaling & Encoding:</h4>
                  <ul className="space-y-2">
                    <li>• StandardScaler for numerical features</li>
                    <li>• LabelEncoder for categorical variables</li>
                    <li>• Handle unknown categories in test data</li>
                    <li>• Feature normalization for model compatibility</li>
                  </ul>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-emerald-500/30">
            <CardHeader>
              <CardTitle className="text-2xl text-emerald-400">Machine Learning Models</CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-6">
              <div>
                <h4 className="font-semibold text-white mb-3">Ensemble Approach: Random Forest + XGBoost</h4>
                <p className="mb-4">
                  We implemented an ensemble learning approach combining the strengths of both Random Forest 
                  and XGBoost algorithms to achieve superior prediction accuracy.
                </p>
                
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="bg-slate-800/50 p-4 rounded-lg">
                    <h5 className="font-semibold text-emerald-400 mb-2">Random Forest (60% weight)</h5>
                    <ul className="text-sm space-y-1">
                      <li>• Robust to overfitting</li>
                      <li>• Handles mixed data types well</li>
                      <li>• Provides feature importance</li>
                      <li>• Excellent for baseline performance</li>
                    </ul>
                  </div>
                  
                  <div className="bg-slate-800/50 p-4 rounded-lg">
                    <h5 className="font-semibold text-emerald-400 mb-2">XGBoost (40% weight)</h5>
                    <ul className="text-sm space-y-1">
                      <li>• Gradient boosting optimization</li>
                      <li>• Handles imbalanced datasets</li>
                      <li>• Superior pattern recognition</li>
                      <li>• Fine-tuned hyperparameters</li>
                    </ul>
                  </div>
                </div>
              </div>

              <Separator className="bg-slate-700" />

              <div>
                <h4 className="font-semibold text-white mb-3">Model Architecture Flow</h4>
                <div className="bg-slate-800/30 p-4 rounded-lg">
                  <div className="flex items-center justify-between text-sm">
                    <div className="text-center">
                      <div className="w-16 h-16 bg-emerald-500/20 border border-emerald-500/50 rounded-full flex items-center justify-center mb-2">
                        <span className="text-emerald-400 font-semibold">Input</span>
                      </div>
                      <p>Network Features</p>
                    </div>
                    <div className="text-emerald-400">→</div>
                    <div className="text-center">
                      <div className="w-16 h-16 bg-blue-500/20 border border-blue-500/50 rounded-full flex items-center justify-center mb-2">
                        <span className="text-blue-400 font-semibold">Pre</span>
                      </div>
                      <p>Preprocessing</p>
                    </div>
                    <div className="text-emerald-400">→</div>
                    <div className="text-center">
                      <div className="w-16 h-16 bg-purple-500/20 border border-purple-500/50 rounded-full flex items-center justify-center mb-2">
                        <span className="text-purple-400 font-semibold">ML</span>
                      </div>
                      <p>Ensemble Model</p>
                    </div>
                    <div className="text-emerald-400">→</div>
                    <div className="text-center">
                      <div className="w-16 h-16 bg-red-500/20 border border-red-500/50 rounded-full flex items-center justify-center mb-2">
                        <span className="text-red-400 font-semibold">Out</span>
                      </div>
                      <p>Prediction</p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function ContactPage() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle form submission here
    console.log('Contact form submitted:', formData);
    // Reset form
    setFormData({ name: '', email: '', message: '' });
    alert('Thank you for your message! We will get back to you soon.');
  };

  return (
    <div className="min-h-screen py-20 px-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-white text-center mb-16">Contact Us</h1>
        
        <div className="grid md:grid-cols-2 gap-12">
          {/* Contact Form */}
          <Card className="bg-slate-900/50 border-emerald-500/30">
            <CardHeader>
              <CardTitle className="text-2xl text-emerald-400 flex items-center">
                <Mail className="w-6 h-6 mr-2" />
                Send us a Message
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <Label htmlFor="name" className="text-slate-300">Name</Label>
                  <Input
                    id="name"
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({...formData, name: e.target.value})}
                    className="bg-slate-800 border-slate-600 text-white"
                    required
                  />
                </div>
                
                <div>
                  <Label htmlFor="email" className="text-slate-300">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({...formData, email: e.target.value})}
                    className="bg-slate-800 border-slate-600 text-white"
                    required
                  />
                </div>
                
                <div>
                  <Label htmlFor="message" className="text-slate-300">Message</Label>
                  <textarea
                    id="message"
                    rows={5}
                    value={formData.message}
                    onChange={(e) => setFormData({...formData, message: e.target.value})}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-md text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                    placeholder="Your message..."
                    required
                  />
                </div>
                
                <Button 
                  type="submit"
                  className="w-full bg-emerald-600 hover:bg-emerald-700 text-white"
                >
                  Send Message
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Team Information */}
          <div className="space-y-6">
            <Card className="bg-slate-900/50 border-emerald-500/30">
              <CardHeader>
                <CardTitle className="text-xl text-emerald-400">Project Team</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-slate-300">
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <Users className="w-5 h-5 text-emerald-400" />
                    <div>
                      <p className="font-semibold text-white">M. Bhagyasri</p>
                      <p className="text-sm">Team Member</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <Users className="w-5 h-5 text-emerald-400" />
                    <div>
                      <p className="font-semibold text-white">K. Rakesh</p>
                      <p className="text-sm">Team Member</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <Users className="w-5 h-5 text-emerald-400" />
                    <div>
                      <p className="font-semibold text-white">G. Ramesh</p>
                      <p className="text-sm">Team Member</p>
                    </div>
                  </div>
                </div>
                
                <Separator className="bg-slate-700" />
                
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <BookOpen className="w-5 h-5 text-emerald-400" />
                    <div>
                      <p className="font-semibold text-white">Santharaju Sir</p>
                      <p className="text-sm">Project Guide</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <BookOpen className="w-5 h-5 text-emerald-400" />
                    <div>
                      <p className="font-semibold text-white">Dr. Tamilkodi</p>
                      <p className="text-sm">Head of Department</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-emerald-500/30">
              <CardHeader>
                <CardTitle className="text-xl text-emerald-400">Project Information</CardTitle>
              </CardHeader>
              <CardContent className="text-slate-300 space-y-3">
                <p><strong className="text-white">Project Title:</strong> Cyber Breach Forecaster</p>
                <p><strong className="text-white">Subtitle:</strong> A Data-Driven System to Predict Hacking Incidents with Machine Learning</p>
                <p><strong className="text-white">Academic Year:</strong> 2024-25</p>
                <p><strong className="text-white">Technology Stack:</strong> Python, FastAPI, React, Machine Learning</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

function Footer() {
  return (
    <footer className="bg-slate-950 border-t border-emerald-500/20 py-8 px-6">
      <div className="max-w-6xl mx-auto text-center text-slate-400">
        <p className="mb-4">
          © 2024-25 Cyber Breach Forecaster. Final Year Project by M.Bhagyasri, K.Rakesh, G.Ramesh
        </p>
        <p className="text-sm">
          Guided by Santharaju Sir | HOD: Dr.Tamilkodi
        </p>
      </div>
    </footer>
  );
}

export default App;