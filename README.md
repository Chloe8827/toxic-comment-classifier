# Toxic Comment Classification System

A deep learning-based system that detects and classifies toxic content in text using XLM-RoBERTa with LORA (Low-Rank Adaptation) fine-tuning.

## Features

- Multi-label classification for different types of toxic content:
  - Toxic
  - Severely Toxic
  - Obscene
  - Threatening
  - Insulting
  - Identity Hate
- Real-time analysis with probability scores
- Visual feedback with color-coded results
- Responsive web interface
- GPU acceleration support

## System Requirements

### Backend
- Python 3.8+
- CUDA support (optional, for GPU acceleration)
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Django Rest Framework

### Frontend
- Node.js 16.0+
- React 18+
- Modern web browser

## Project Structure

```
toxic-classifier/
├── backend/
│   ├── toxic_classifier/
│   │   ├── models/            # Pre-trained model files
│   │   ├── views.py          # API endpoints
│   │   └── model_manager.py  # Model loading and inference
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   └── App.css          # Styles
│   └── package.json
└── training/
    └── train_lora_xlm.py    # Model training script
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Chloe8827/toxic-comment-classifier.git
cd toxic-classifier
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Download Models
Download the pre-trained models and place them in the `backend/toxic_classifier/models/` directory. The model structure should be:
```
models/
├── basic_lora_xlm_toxic/
├── basic_lora_xlm_severe_toxic/
├── basic_lora_xlm_obscene/
├── basic_lora_xlm_threat/
├── basic_lora_xlm_insult/
└── basic_lora_xlm_identity_hate/
```

## Usage

### Starting the Backend Server
```bash
cd backend
python manage.py runserver
```

### Starting the Frontend Development Server
```bash
cd frontend
npm start
```

The application will be available at `http://localhost:3000`

## Model Training

To train new models:

1. Prepare your training data in CSV format with columns: 'cleaned_text' and label columns
2. Update the configuration in `training/train_lora_xlm.py` if needed
3. Run the training script:
```bash
cd training
python train_lora_xlm.py
```

Training parameters can be adjusted in the script:
- Learning rate
- Batch size
- Number of epochs
- Early stopping patience
- LORA configuration (rank, alpha, dropout)

## API Endpoints

### Prediction Endpoint
- URL: `/api/predict/`
- Method: `POST`
- Request Body:
```json
{
    "text": "text to analyze"
}
```
- Response:
```json
{
    "scores": {
        "toxic": 0.123,
        "severe_toxic": 0.045,
        "obscene": 0.067,
        "threat": 0.012,
        "insult": 0.089,
        "identity_hate": 0.034
    }
}
```

## Error Handling

The system includes comprehensive error handling for:
- Invalid input
- Model loading failures
- API communication errors
- Resource unavailability

## Performance Considerations

- GPU acceleration is recommended for production deployment
- Model loading time depends on available system resources
- Batch processing is supported for multiple inputs

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Contact

For technical support or inquiries: chloe.c.chuqian@gmail.com