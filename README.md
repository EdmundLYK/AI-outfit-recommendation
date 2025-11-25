# AI Outfit Recommendation System

An intelligent fashion recommendation system that analyzes body measurements and skin tone from uploaded images to provide personalized outfit suggestions. The system combines computer vision, pose detection, n8n automation and AI-powered styling recommendations.

## 🌟 Features

- **Body Analysis**: Uses MediaPipe to detect key body landmarks and calculate measurements
- **Skin Tone Detection**: Automatically extracts and categorizes skin tone from facial features
- **Body Shape Classification**: Determines body type (Inverted Triangle, Pear, Rectangle/Balanced)
- **AI-Powered Recommendations**: Provides tailored outfit suggestions for different occasions
- **Web Interface**: Clean, modern UI for easy image upload and result visualization
- **Workflow Automation**: Integrated with n8n for seamless processing pipeline

## 🏗️ System Architecture

The system consists of multiple components working together:

1. **Flask API** (`app.py`) - Core pose detection and analysis service
2. **n8n Workflow** - Orchestrates the processing pipeline
3. **Coordinates Calculation** (`coordinates_calc.js`) - Processes body measurements in n8n
4. **AI Recommendation Engine** - Generates personalized outfit suggestions
5. **Web Interface** (`index.html`) - User-friendly frontend

### Workflow Overview

```
Image Upload → Pose Detection → Body Analysis → AI Recommendations → Response to User
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js (for n8n)
- Docker (optional)

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd FYP/AI-outfit-recommendation
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask API**

   ```bash
   python app.py
   ```

   The API will be available at `http://localhost:5000`

4. **Set up n8n workflow** (see [n8n Setup](#n8n-setup) section)

5. **Open the web interface**
   Open `index.html` in your browser or serve it using a local web server.

### Docker Setup

Build and run using Docker:

```bash
docker build -t ai-outfit-recommendation .
docker run -p 5000:5000 ai-outfit-recommendation
```

## 📋 API Endpoints

### POST `/analyze`

Analyzes an uploaded image and returns body measurements and skin tone data.

**Request:**

- Content-Type: `multipart/form-data`
- Body: Image file with key `file`

**Response:**

```json
{
  "keypoints": {
    "nose": { "x": 320, "y": 240, "z": -0.1, "visibility": 0.95 },
    "left_shoulder": { "x": 280, "y": 300, "z": 0.05, "visibility": 0.88 }
    // ... other landmarks
  },
  "skin_color": {
    "hex": "#D4A574",
    "rgb": [212, 165, 116],
    "tone_category": "Medium"
  },
  "body_width": {
    "shoulder_px": 120,
    "hip_px": 100,
    "shoulder_to_hip_ratio": 1.2,
    "body_shape": "Inverted Triangle"
  },
  "model_accuracy": {
    "overall_confidence": 0.85,
    "key_landmarks_detected": 5,
    "total_key_landmarks": 5
  }
}
```

## 🔧 n8n Setup

The system uses n8n to orchestrate the complete workflow from image processing to AI recommendations.

### Installation

1. **Install n8n**

   ```bash
   npm install -g n8n
   ```

2. **Start n8n**
   ```bash
   n8n start
   ```
   Access n8n at `http://localhost:5678`

### Workflow Configuration

The n8n workflow includes the following nodes:

1. **Webhook Trigger** - Receives image uploads
2. **HTTP Request** - Calls the Flask API for pose analysis
3. **Code Node** - Processes coordinates using `coordinates_calc.js`
4. **AI Model Node** - Generates outfit recommendations
5. **Response Node** - Returns formatted recommendations

### Webhook Setup

- **Method**: POST
- **Path**: `/webhook/upload-image`
- **Response Mode**: Respond to Webhook

The workflow processes the uploaded image through the entire pipeline and returns AI-generated outfit recommendations.

## 🎨 Recommendation System

The AI recommendation engine provides outfit suggestions based on:

### Input Parameters

- **Skin Tone**: Light, Medium, Medium-Dark, Dark
- **Body Shape**: Inverted Triangle, Pear, Rectangle/Balanced
- **Body Measurements**: Shoulder-to-waist ratio, torso length

### Output Categories

- **Casual**: Everyday comfortable styling
- **Business-Casual**: Professional yet relaxed looks
- **Streetwear/Trendy**: Modern, fashion-forward styles

### Sample Recommendation Format

```
### Casual
- **Top:** A relaxed-fit olive green henley with rolled sleeves
- **Accessory:** Simple leather bracelet and classic analog watch
- **Reasoning:** The olive green complements medium skin tone while the relaxed fit balances broad shoulders...
```

## 🖥️ Web Interface

The web interface (`index.html`) provides:

- **Drag & Drop Upload**: Easy image selection
- **Live Preview**: See uploaded image before analysis
- **Real-time Results**: Formatted AI recommendations
- **Responsive Design**: Works on desktop and mobile devices

### Usage

1. Open `index.html` in your browser
2. Upload or drag & drop an image
3. Click "Analyze Pose"
4. View personalized outfit recommendations

## 🔍 Technical Details

### Body Analysis Features

- **Pose Detection**: 33 body landmarks using MediaPipe
- **Excluded Landmarks**: Lower body points (ankles, knees, feet) for focused upper-body analysis
- **Visibility Threshold**: 0.5 minimum visibility for reliable measurements
- **Fallback Logic**: Multiple calculation methods for robust measurements

### Skin Tone Classification

- **Color Extraction**: Analyzes facial region (nose, eyes area)
- **RGB to Hex Conversion**: Provides both formats for styling
- **Tone Categories**: 4-tier classification system
- **Error Handling**: Graceful fallback when face detection fails

### Body Shape Logic

```javascript
if (shoulder_to_hip_ratio > 1.2) → "Inverted Triangle"
if (shoulder_to_hip_ratio < 0.8) → "Pear"
else → "Rectangle/Balanced"
```

## 📁 Project Structure

```
AI-outfit-recommendation/
├── app.py                    # Flask API server
├── coordinates_calc.js       # Body measurement calculations
├── index.html               # Web interface
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
├── recommendation_sys.txt   # AI prompt template
└── test_image.jpg          # Sample test image
```

## 🛠️ Development

### Adding New Features

1. **New Body Measurements**: Extend `coordinates_calc.js` with additional calculations
2. **Enhanced Recommendations**: Modify the prompt in `recommendation_sys.txt`
3. **UI Improvements**: Update `index.html` styling and functionality
4. **API Extensions**: Add new endpoints in `app.py`

### Testing

Test the API directly:

```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:5000/analyze
```

### Debugging

- Check Flask logs for API errors
- Use browser developer tools for frontend issues
- Monitor n8n execution logs for workflow problems
- Verify MediaPipe model accuracy in API responses

## 🚀 Deployment

### Production Considerations

1. **Security**: Add authentication and rate limiting
2. **Scalability**: Use load balancers and container orchestration
3. **Storage**: Implement proper image storage (S3, etc.)
4. **Monitoring**: Add logging and health checks
5. **HTTPS**: Use SSL certificates for secure communication

### Environment Variables

```bash
FLASK_ENV=production
PORT=5000
N8N_WEBHOOK_URL=http://localhost:5678/webhook/upload-image
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**API not responding:**

- Ensure Flask server is running on port 5000
- Check firewall settings

**n8n workflow errors:**

- Verify webhook URL in frontend
- Check n8n node configurations
- Ensure all required credentials are set

**Poor pose detection:**

- Use well-lit, front-facing images
- Ensure full upper body is visible
- Check image resolution and quality

**AI recommendations not loading:**

- Verify AI model node configuration in n8n
- Check API keys and model availability
- Monitor n8n execution logs

For issues and questions:

- Check the troubleshooting section
- Review n8n workflow logs
- Open an issue in the repository

---

Built with ❤️ using Flask, MediaPipe, n8n, and AI-powered fashion intelligence.
