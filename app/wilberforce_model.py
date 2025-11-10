import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class WilberforcePredictor:
    """LSTM-based predictive maintenance system for vehicle fault detection"""
    
    def __init__(self, model_path='../model/wilberforce_lstm_model.h5', scaler_path='../model/scaler.pkl'):
        try:
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.sequence_length = 20
            self.recent_data = []
            self.model_loaded = True
        except Exception as e:
            print(f"⚠️ Warning: Could not load model files. Error: {e}")
            print("🔧 Running in simulation mode with dummy predictions")
            self.model_loaded = False
            self.recent_data = []
            self.sequence_length = 20
        
    def preprocess_sensor_data(self, sensor_readings):
        """Preprocess new sensor data for prediction"""
        # Expected features: [temp, vibration, pressure, rpm, fuel_flow, temp*vibration, pressure/temp, sqrt(vibration*rpm)]
        temp, vibration, pressure, rpm, fuel_flow = sensor_readings
        
        # Calculate derived features
        temp_vibration = temp * vibration
        pressure_temp_ratio = pressure / max(temp, 1)
        vibration_rpm_metric = np.sqrt(vibration * rpm)
        
        features = [
            temp, vibration, pressure, rpm, fuel_flow,
            temp_vibration, pressure_temp_ratio, vibration_rpm_metric
        ]
        
        return np.array(features).reshape(1, -1)
    
    def add_sensor_data(self, sensor_readings):
        """Add new sensor data to the recent data buffer"""
        if not self.model_loaded:
            # If model not loaded, just store raw data for simulation
            self.recent_data.append(sensor_readings)
            if len(self.recent_data) > self.sequence_length:
                self.recent_data = self.recent_data[-self.sequence_length:]
            return
            
        processed_data = self.preprocess_sensor_data(sensor_readings)
        scaled_data = self.scaler.transform(processed_data)
        
        self.recent_data.append(scaled_data[0])
        
        # Keep only the most recent sequence_length data points
        if len(self.recent_data) > self.sequence_length:
            self.recent_data = self.recent_data[-self.sequence_length:]
    
    def predict_fault_probability(self):
        """Predict fault probability using the most recent sequence - FIXED TO RETURN 3 VALUES"""
        if not self.model_loaded:
            # Simulation mode - generate realistic probabilities based on sensor data
            if len(self.recent_data) < 5:
                return 0.1, "Initializing - collecting data", "none"
            
            # Simple heuristic based on sensor values
            latest_data = self.recent_data[-1]
            temp, vibration, pressure, rpm, fuel_flow = latest_data
            
            # Basic fault detection logic
            fault_score = 0.0
            if temp > 100: fault_score += 0.3
            if vibration > 4.0: fault_score += 0.3
            if pressure < 20: fault_score += 0.3
            if rpm > 3500: fault_score += 0.1
            
            probability = min(fault_score, 0.95)
            
        else:
            # Real model prediction
            if len(self.recent_data) < self.sequence_length:
                return 0.0, "Insufficient data", "none"
            
            sequence = np.array(self.recent_data[-self.sequence_length:])
            sequence = sequence.reshape(1, self.sequence_length, -1)
            
            probability = self.model.predict(sequence, verbose=0)[0][0]
        
        # Determine status based on probability - FIXED TO ALWAYS RETURN 3 VALUES
        if probability > 0.7:
            status = "CRITICAL - Immediate maintenance required"
            alert_level = "high"
        elif probability > 0.5:
            status = "WARNING - Maintenance recommended"
            alert_level = "medium"
        elif probability > 0.3:
            status = "MONITOR - Watch for changes"
            alert_level = "low"
        else:
            status = "NORMAL - No issues detected"
            alert_level = "none"
        
        return probability, status, alert_level  # ALWAYS RETURN 3 VALUES
    
    def generate_sensor_reading(self, base_values=None):
        """Generate simulated sensor reading for demonstration"""
        if base_values is None:
            base_values = {
                'temp': 85, 'vibration': 2.0, 'pressure': 35, 
                'rpm': 2500, 'fuel_flow': 8
            }
        
        # Add some random variation
        temp = max(60, base_values['temp'] + np.random.normal(0, 5))
        vibration = max(0.1, base_values['vibration'] + np.random.normal(0, 0.3))
        pressure = max(10, base_values['pressure'] + np.random.normal(0, 2))
        rpm = max(500, base_values['rpm'] + np.random.normal(0, 200))
        fuel_flow = max(2, base_values['fuel_flow'] + np.random.normal(0, 1))
        
        return [temp, vibration, pressure, rpm, fuel_flow]
    
    def get_system_status(self):
        """Get overall system status"""
        if len(self.recent_data) < 5:
            return "Initializing", "Collecting sensor data..."
        
        prob, status, alert_level = self.predict_fault_probability()
        
        data_points = len(self.recent_data)
        mode = "SIMULATION" if not self.model_loaded else "LSTM MODEL"
        return status, f"Analyzed {data_points} data points. Fault probability: {prob:.2%} ({mode})"

# Utility functions for the Streamlit app
def generate_vehicle_data(num_vehicles=5):
    """Generate sample vehicle data for dashboard"""
    vehicles = []
    for i in range(num_vehicles):
        vehicles.append({
            'vehicle_id': f'VH{1000 + i}',
            'model': np.random.choice(['Toyota Hilux', 'Ford Ranger', 'Nissan Navara', 'Isuzu D-Max']),
            'status': np.random.choice(['Normal', 'Maintenance Due', 'Critical']),
            'last_service': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 180)),
            'fault_probability': np.random.uniform(0, 0.8)
        })
    return pd.DataFrame(vehicles)

def generate_maintenance_history(vehicle_id, num_entries=3):
    """Generate sample maintenance history"""
    history = []
    for i in range(num_entries):
        history.append({
            'date': pd.Timestamp.now() - pd.Timedelta(days=(i+1)*30),
            'service_type': np.random.choice(['Oil Change', 'Brake Service', 'Engine Check', 'General Maintenance']),
            'cost': np.random.randint(50, 300),
            'technician': np.random.choice(['John M.', 'Sarah K.', 'Mike T.']),
            'status': 'Completed'
        })
    return pd.DataFrame(history)