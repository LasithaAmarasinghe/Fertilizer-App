# 🌿 Plant Leaf Analyzer - AI-Powered Fertilizer Recommendation App

A web application that uses a TensorFlow Keras model to analyze plant leaf images and provide fertilizer recommendations based on detected diseases.

## Features

- **AI-Powered Detection**: Uses PlantVillage model to classify leaves with confidence scores
- **Disease Detection**: Identifies Early Blight, Late Blight, and Healthy leaves
- **Fertilizer Recommendations**: Provides detailed fertilizer suggestions including:
  - Primary treatment with NPK ratios
  - Secondary supplements
  - Application schedule
  - Care tips
- **Nutrient Analysis**: Visual representation of nutrient levels
- **Severity Indicator**: Shows the severity level of detected issues
- **Responsive Design**: Works on desktop and mobile devices

## Model Capabilities

The integrated `plant_village_model_v1.keras` model classifies leaves into:
- **Early Blight** - Fungal disease (Alternaria solani) with dark concentric spots
- **Late Blight** - Severe water mold disease (Phytophthora infestans)
- **Healthy** - Normal, healthy leaf tissue with maintenance recommendations

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Model File**
   Ensure `plant_village_model_v1.keras` is in the app directory

## Running the Application

### Method 1: Using the Batch Script (Windows)
Simply double-click `start_server.bat`

### Method 2: Manual Start
```bash
python app.py
```

The backend server will start on `http://localhost:5000`

### Using the Web Interface

1. Start the backend server (using one of the methods above)
2. Open `index.html` in your web browser
3. Upload a leaf image (drag & drop or click to browse)
4. Click "Analyze Leaf" button
5. View AI-powered diagnosis with confidence scores and fertilizer recommendations

## API Endpoints

### POST /predict
Accepts an image file and returns classification results

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
  "success": true,
  "prediction": "Early_Blight",
  "confidence": 0.95,
  "probabilities": {
    "Early_Blight": 0.95,
    "Late_Blight": 0.03,
    "Healthy": 0.02
  }
}
```

### GET /health
Health check endpoint to verify server status

## Technologies Used

- HTML5
- CSS3 (with gradients and animations)
- Vanilla JavaScript
- No external dependencies required

## File Structure

```
Fertilizer App/
├── index.html                      # Frontend interface
├── styles.css                      # Styling and animations
├── script.js                       # Frontend logic with API integration
├── app.py                          # Flask backend server
├── requirements.txt                # Python dependencies
├── plant_village_model_v1.keras   # TensorFlow model
├── start_server.bat               # Windows startup script
└── README.md                       # Documentation
```

## Technology Stack

**Frontend:**
- HTML5, CSS3, JavaScript
- Fetch API for backend communication

**Backend:**
- Flask (Python web framework)
- TensorFlow/Keras (ML model inference)
- Flask-CORS (Cross-origin support)
- Pillow (Image processing)

**Model:**
- TensorFlow Keras model trained on PlantVillage dataset
- Input: 256x256 RGB images
- Output: 3 classes (Early_Blight, Late_Blight, Healthy)

## Troubleshooting

### Server won't start
- Verify Python is installed: `python --version`
- Install dependencies: `pip install -r requirements.txt`
- Check if port 5000 is available

### Model prediction fails
- Ensure model file exists and is not corrupted
- Check image format (PNG, JPG supported)
- Verify image is a valid RGB image

### CORS errors
- Make sure Flask-CORS is installed
- Verify API_URL in script.js matches server address
- Ensure backend server is running

## Future Enhancements

- Support for more plant diseases and crops
- Mobile app version (React Native/Flutter)
- User accounts and analysis history tracking
- Batch image processing
- Multilingual support
- Integration with agricultural databases
- Real-time disease progression monitoring
- Weather-based recommendations

## License

Free to use for educational and personal projects.
