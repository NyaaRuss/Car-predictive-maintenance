import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings
warnings.filterwarnings('ignore')

class VehicleDataGenerator:
    """Generate simulated vehicle sensor data for predictive maintenance"""
    
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        self.scaler = StandardScaler()
        
    def generate_sensor_data(self):
        """Generate realistic vehicle sensor data"""
        np.random.seed(42)
        
        # Time series data - 100 timesteps per vehicle
        time_steps = 100
        n_vehicles = self.n_samples // time_steps
        
        data = []
        labels = []
        
        for vehicle_id in range(n_vehicles):
            # Base parameters for normal operation
            base_temp = np.random.normal(85, 5)  # Engine temperature
            base_vibration = np.random.normal(2, 0.5)  # Vibration levels
            base_pressure = np.random.normal(35, 3)  # Oil pressure
            
            for timestep in range(time_steps):
                # Normal sensor readings with some noise
                if timestep < 80:  # Normal operation for first 80 timesteps
                    temp = base_temp + np.random.normal(0, 2)
                    vibration = base_vibration + np.random.normal(0, 0.2)
                    pressure = base_pressure + np.random.normal(0, 1)
                    rpm = np.random.normal(2500, 300)
                    fuel_flow = np.random.normal(8, 1)
                    fault_prob = 0.1  # Low fault probability
                    
                else:  # Potential fault developing in last 20 timesteps
                    # Simulate developing faults
                    fault_type = np.random.choice(['overheating', 'vibration', 'pressure_drop'])
                    
                    if fault_type == 'overheating':
                        temp = base_temp + np.random.normal(15, 3) + (timestep-80)*0.5
                        vibration = base_vibration + np.random.normal(0.5, 0.3)
                        pressure = base_pressure - np.random.normal(2, 1)
                    elif fault_type == 'vibration':
                        temp = base_temp + np.random.normal(5, 2)
                        vibration = base_vibration + np.random.normal(2, 0.5) + (timestep-80)*0.1
                        pressure = base_pressure + np.random.normal(0, 1)
                    else:  # pressure_drop
                        temp = base_temp + np.random.normal(8, 2)
                        vibration = base_vibration + np.random.normal(1, 0.3)
                        pressure = base_pressure - np.random.normal(5, 2) - (timestep-80)*0.3
                    
                    rpm = np.random.normal(2800, 400)
                    fuel_flow = np.random.normal(10, 2)
                    fault_prob = 0.8  # High fault probability
                
                # Create feature vector
                features = [
                    temp, vibration, pressure, rpm, fuel_flow,
                    temp * vibration,  # Interaction term
                    pressure / max(temp, 1),  # Pressure to temp ratio
                    np.sqrt(vibration * rpm)  # Combined vibration-RPM metric
                ]
                
                data.append(features)
                labels.append(1 if fault_prob > 0.5 else 0)
        
        return np.array(data), np.array(labels)
    
    def create_sequences(self, data, labels, sequence_length=20):
        """Convert data into sequences for LSTM"""
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(labels[i + sequence_length - 1])
        
        return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Build and compile LSTM model"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']  # Simplified metrics to avoid the error
    )
    
    return model

def calculate_metrics(model, X_test, y_test):
    """Calculate additional metrics manually"""
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    return accuracy, precision, recall

def main():
    print("🚗 Generating simulated vehicle sensor data...")
    
    # Generate data
    generator = VehicleDataGenerator(n_samples=20000)
    data, labels = generator.generate_sensor_data()
    
    print(f"📊 Generated {len(data)} samples")
    print(f"🎯 Fault distribution: {np.sum(labels)} faults out of {len(labels)} samples")
    
    # Scale features
    scaled_data = generator.scaler.fit_transform(data)
    
    # Create sequences
    sequence_length = 20
    X, y = generator.create_sequences(scaled_data, labels, sequence_length)
    
    print(f"🔄 Created sequences: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"📈 Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    
    print("🧠 Model architecture:")
    model.summary()
    
    # Train model
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    print("🏋️ Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    print("📊 Evaluating model...")
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Calculate additional metrics
    train_accuracy_manual, train_precision, train_recall = calculate_metrics(model, X_train, y_train)
    test_accuracy_manual, test_precision, test_recall = calculate_metrics(model, X_test, y_test)
    
    print(f"✅ Training - Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
    print(f"✅ Test - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    
    # Save model and artifacts
    model.save('wilberforce_lstm_model.h5')
    joblib.dump(generator.scaler, 'scaler.pkl')
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('training_history.csv', index=False)
    
    # Save model performance
    performance = {
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall
    }
    performance_df = pd.DataFrame([performance])
    performance_df.to_csv('model_performance.csv', index=False)
    
    print("💾 Model saved as 'wilberforce_lstm_model.h5'")
    print("💾 Scaler saved as 'scaler.pkl'")
    print("💾 Training history saved as 'training_history.csv'")
    print("🎉 Model training completed successfully!")

if __name__ == "__main__":
    main()