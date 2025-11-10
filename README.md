# 🚗 Wilberforce - Predictive Maintenance System

## 📖 Overview

Wilberforce is an **LSTM-based predictive maintenance system** designed for early detection of automotive faults using vehicle sensor data. The system employs deep learning to analyze real-time sensor readings and predict potential mechanical failures before they occur.

## 🎯 Project Objectives

1. **Detect vehicle faults** based on historical and simulated vehicle sensor data
2. **Develop an LSTM model** capable of predicting mechanical faults before failure
3. **Evaluate model performance** using accuracy, precision, and recall metrics

## 🔧 System Features

### Real-time Monitoring
- Continuous tracking of 5 critical vehicle sensors:
  - Engine Temperature (°C)
  - Vibration Levels (g)
  - Oil Pressure (psi)
  - Engine RPM
  - Fuel Flow Rate (L/h)

### AI-Powered Prediction
- **LSTM Neural Network** for time-series pattern recognition
- **Real-time fault probability** calculation
- **Early warning system** for proactive maintenance

### Alert System
- **Three-tier alert levels**:
  - 🟢 NORMAL (0-30% fault probability)
  - 🟠 WARNING (30-70% fault probability)
  - 🔴 CRITICAL (70-100% fault probability)

### User Interface
- **Interactive Streamlit dashboard**
- **Manual sensor input** for custom testing
- **Quick preset scenarios** for demonstration
- **Real-time visualization** with Plotly charts

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- TensorFlow 2.13+
- Streamlit 1.28+

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd wilberforce


LSTM-based automotive fault detection using historical vehicle sensor data or manual input.

## Installation
```bash
pip install -r requirements.txt

##python model/model_training.py

##streamlit run app/streamlit_app.py