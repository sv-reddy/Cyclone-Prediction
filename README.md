# 🌀 Cyclone Prediction and Early Warning System

A comprehensive machine learning system for predicting cyclones using weather data and satellite imagery. This project combines traditional weather parameters with deep learning-based satellite image analysis to provide accurate cyclone forecasting and early warning capabilities.

## 🎯 Project Overview

The Cyclone Prediction and Early Warning System is designed to:
- Predict cyclone formation using meteorological data
- Classify satellite images for cyclone detection
- Provide early warning alerts through SMS notifications
- Support disaster preparedness and response efforts

## 🏗️ Project Structure

```
Cyclone-Prediction-and-Early-warning/
├── 📊 cyclone_databased_on_weather.csv    # Weather dataset with cyclone occurrences
├── 📓 Cyclone_Prediction.ipynb            # Main Jupyter notebook with ML pipeline
├── 📋 requirements.txt                     # Python dependencies
├── 🖼️ Cyclone_Dataset/                    # Satellite image datasets
│   ├── cyclone/                           # Cyclone satellite images
│   └── noncyclone/                        # Non-cyclone images
├── 📝 logs/                               # Training and prediction logs
│   └── cyclone_prediction_20250706_173553.json
└── 🤖 models/                             # Trained ML models
    ├── weather_model.keras                     # Weather-based prediction model
    └── satellite_model_best.h5           # CNN model for satellite images
```

## 🚀 Features

### 1. Weather-Based Prediction
- **Multi-parameter Analysis**: Wind speed, atmospheric pressure, ocean temperature
- **Machine Learning Models**: Random Forest, Neural Networks
- **Real-time Processing**: Live weather data integration
- **Accuracy Metrics**: Precision, Recall, F1-score tracking

### 2. Satellite Image Classification
- **Deep Learning CNN**: TensorFlow/Keras-based image classification
- **Image Preprocessing**: Automated image augmentation and normalization
- **Real-time Analysis**: Process satellite feeds for cyclone detection
- **Visual Recognition**: Identify cyclone eye, spiral patterns, and cloud formations

### 3. Early Warning System
- **SMS Alerts**: Twilio integration for emergency notifications
- **API Integration**: Weather service APIs for real-time data
- **Geolocation**: Location-based warnings and alerts
- **Multi-channel Notifications**: SMS, email, and web-based alerts

### 4. Comprehensive Monitoring
- **Performance Tracking**: Model accuracy and prediction confidence
- **Logging System**: Detailed operation logs and audit trails
- **Visualization**: Interactive charts and maps for data analysis
- **Historical Analysis**: Trend analysis and seasonal patterns

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **TensorFlow/Keras**: Deep learning framework
- **Scikit-learn**: Traditional machine learning algorithms
- **OpenCV**: Computer vision and image processing

### Data Science Libraries
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive visualizations

### External Services
- **Twilio**: SMS notification service
- **Weather APIs**: Real-time meteorological data
- **Geopy**: Geolocation services
- **Folium**: Interactive mapping

## 📊 Dataset Information

### Weather Dataset (`cyclone_databased_on_weather.csv`)
- **Size**: 200+ records
- **Features**: 
  - Cyclone Name
  - Date and Location
  - Wind Speed (km/h)
  - Atmospheric Pressure (hPa)
  - Ocean Temperature (°C)
  - Cyclone Occurrence (Binary: 0/1)

### Satellite Images Dataset
- **Cyclone Images**: 150+ satellite images showing cyclone formations
- **Non-cyclone Images**: Control dataset for classification training
- **Image Format**: JPG format, processed and augmented
- **Source**: OCM GAC satellite data (2023-2024)

## 🚦 Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Git (for version control)
- 4GB+ RAM recommended
- GPU support optional (for faster deep learning training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sv-reddy/Cyclone-Prediction.git
   cd Cyclone-Prediction-and-Early-warning
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file with your API keys
   TWILIO_ACCOUNT_SID=your_twilio_sid
   TWILIO_AUTH_TOKEN=your_twilio_token
   WEATHER_API_KEY=your_weather_api_key
   ```

5. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook Cyclone_Prediction.ipynb
   ```

### Quick Start

1. Open `Cyclone_Prediction.ipynb` in Jupyter
2. Run cells sequentially to:
   - Load and preprocess data
   - Train weather-based prediction models
   - Build CNN for satellite image classification
   - Set up real-time prediction pipeline
   - Configure alert system

## 📈 Model Performance

### Weather-Based Prediction Model
- **Algorithm**: Random Forest + Neural Network ensemble
- **Accuracy**: ~85-90% on test data
- **Features Used**: Wind speed, pressure, ocean temperature, location
- **Prediction Time**: <1 second per sample

### Satellite Image Classification
- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224 RGB images
- **Accuracy**: ~92% cyclone detection rate
- **Processing Time**: ~0.5 seconds per image

## 🔧 Usage Examples

### Weather Data Prediction
```python
# Load trained model
model = joblib.load('models/my_model.keras')

# Predict cyclone probability
weather_data = {
    'wind_speed': 120,
    'pressure': 950,
    'ocean_temperature': 28.5
}
prediction = model.predict([list(weather_data.values())])
```

### Satellite Image Analysis
```python
# Load CNN model
cnn_model = tf.keras.models.load_model('models/satellite_model_best.h5')

# Classify satellite image
image = cv2.imread('satellite_image.jpg')
prediction = cnn_model.predict(preprocess_image(image))
```

### Send Alert
```python
# Initialize Twilio client
client = Client(account_sid, auth_token)

# Send cyclone warning
message = client.messages.create(
    body="⚠️ CYCLONE WARNING: High probability cyclone formation detected in your area.",
    from_='+1234567890',
    to='+0987654321'
)
```


## 📝 Output Format 

Cyclone prediction results are typically output in JSON format as shown below:

```json
{
   "timestamp": "2025-07-06T17:35:53.332631",
   "location": "Visakhapatnam",
   "coordinates": {
      "lat": 17.69,
      "lon": 83.2093
   },
   "weather_prediction": 0.52,
   "satellite_prediction": null,
   "combined_prediction": 0.52,
   "risk_level": "MODERATE",
   "confidence": "MEDIUM",
   "method": "Weather Only"
}
```

**Fields:**
- `timestamp`: Date and time of prediction
- `location`: Location name
- `coordinates`: Latitude and longitude
- `weather_prediction`: Probability from weather data
- `satellite_prediction`: Probability from satellite image (if available)
- `combined_prediction`: Final combined probability
- `risk_level`: Risk assessment (e.g., LOW, MODERATE, HIGH)
- `confidence`: Model confidence level
- `method`: Prediction method used

## 🌟 Key Features in Detail

### Data Preprocessing
- Automated data cleaning and normalization
- Missing value imputation using statistical methods
- Feature scaling for optimal model performance
- Data augmentation for image datasets

### Model Training
- Cross-validation for robust model evaluation
- Hyperparameter tuning using grid search
- Early stopping to prevent overfitting
- Model checkpointing for best performance

### Real-time Monitoring
- Live weather data integration
- Continuous model inference
- Alert threshold configuration
- Performance metrics tracking
