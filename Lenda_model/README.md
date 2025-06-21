# Brainwave Correspondence Analysis Model

A Django-based web service for analyzing correspondence between brainwave signals using PyTorch deep learning models with online learning capabilities.

## Features

- **Real-time Analysis**: Analyze correspondence between two brainwave signals
- **Online Learning**: Continuously train the model with new data streams
- **RESTful API**: Easy-to-use endpoints for analysis and training
- **Data Storage**: Persistent storage of brainwave data and analysis results
- **Model Monitoring**: Track training metrics and model performance
- **Sample Data Generation**: Built-in tools for testing and development

## Project Structure

```
model/
├── brainwave_analyzer/          # Main Django app
│   ├── models.py               # Database models
│   ├── views.py                # API views
│   ├── urls.py                 # URL routing
│   ├── admin.py                # Django admin
│   └── ml_model.py             # PyTorch model implementation
├── model/                      # Django project settings
│   ├── settings.py             # Project configuration
│   └── urls.py                 # Main URL routing
├── requirements.txt            # Python dependencies
└── manage.py                   # Django management script
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Lenda
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv model_env
   source model_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run database migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Create superuser (optional)**:
   ```bash
   python manage.py createsuperuser
   ```

6. **Run the development server**:
   ```bash
   python manage.py runserver
   ```

The server will be available at `http://localhost:8000`

## API Endpoints

### Analysis Endpoints

#### Analyze Brainwaves
- **URL**: `POST /api/analyze/`
- **Description**: Analyze correspondence between two brainwave signals
- **Request Body**:
  ```json
  {
    "wave1": [0.1, 0.2, 0.3, ...],
    "wave2": [0.15, 0.25, 0.35, ...],
    "sample_rate": 256
  }
  ```
- **Response**:
  ```json
  {
    "id": 1,
    "correspondence_score": 0.85,
    "corresponds": true,
    "confidence": 0.7,
    "features": {...},
    "timestamp": "2025-01-21T10:30:00Z"
  }
  ```

### Training Endpoints

#### Single Sample Training
- **URL**: `POST /api/train/`
- **Description**: Train the model with a single labeled sample
- **Request Body**:
  ```json
  {
    "wave1": [0.1, 0.2, 0.3, ...],
    "wave2": [0.15, 0.25, 0.35, ...],
    "label": true
  }
  ```

#### Batch Training
- **URL**: `POST /api/batch-train/`
- **Description**: Train the model with multiple samples
- **Request Body**:
  ```json
  {
    "training_data": [
      {
        "wave1": [...],
        "wave2": [...],
        "label": true
      },
      {
        "wave1": [...],
        "wave2": [...],
        "label": false
      }
    ]
  }
  ```

### Monitoring Endpoints

#### Model Statistics
- **URL**: `GET /api/stats/`
- **Description**: Get current model performance statistics

#### Analysis History
- **URL**: `GET /api/history/?page=1&page_size=20`
- **Description**: Get paginated analysis history

#### Health Check
- **URL**: `GET /api/health/`
- **Description**: Check if the service is running

#### Sample Data
- **URL**: `GET /api/sample-data/`
- **Description**: Generate sample brainwave data for testing

## Model Architecture

The brainwave correspondence model uses a deep neural network with the following components:

1. **Feature Extraction**: Dense layers to extract features from concatenated brainwave data
2. **LSTM Layer**: Captures temporal patterns in the brainwave signals
3. **Classification**: Dense layers with sigmoid activation for binary classification

### Model Parameters

- **Input Size**: 100 samples per brainwave (configurable)
- **Hidden Size**: 128 neurons
- **LSTM Layers**: 2 layers with dropout
- **Learning Rate**: 0.001 (Adam optimizer)

## Usage Examples

### Python Client Example

```python
import requests
import numpy as np

# Generate sample brainwave data
sample_rate = 256
duration = 1
t = np.linspace(0, duration, sample_rate)
wave1 = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
wave2 = np.sin(2 * np.pi * 10 * t + 0.1) + 0.1 * np.random.randn(len(t))

# Analyze correspondence
response = requests.post('http://localhost:8000/api/analyze/', json={
    'wave1': wave1.tolist(),
    'wave2': wave2.tolist(),
    'sample_rate': sample_rate
})

result = response.json()
print(f"Correspondence Score: {result['correspondence_score']:.3f}")
print(f"Corresponds: {result['corresponds']}")
```

### JavaScript Client Example

```javascript
// Generate sample data
const sampleRate = 256;
const duration = 1;
const t = Array.from({length: sampleRate}, (_, i) => i / sampleRate);
const wave1 = t.map(x => Math.sin(2 * Math.PI * 10 * x) + 0.1 * (Math.random() - 0.5));
const wave2 = t.map(x => Math.sin(2 * Math.PI * 10 * x + 0.1) + 0.1 * (Math.random() - 0.5));

// Analyze correspondence
fetch('http://localhost:8000/api/analyze/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        wave1: wave1,
        wave2: wave2,
        sample_rate: sampleRate
    })
})
.then(response => response.json())
.then(result => {
    console.log(`Correspondence Score: ${result.correspondence_score.toFixed(3)}`);
    console.log(`Corresponds: ${result.corresponds}`);
});
```

## Configuration

### Model Settings

Edit `model/settings.py` to configure model parameters:

```python
MODEL_SETTINGS = {
    'BRAINWAVE_SEQUENCE_LENGTH': 100,  # Number of samples to analyze
    'SAMPLE_RATE': 256,  # Hz
    'FREQUENCY_BANDS': {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }
}
```

## Development

### Running Tests

```bash
python manage.py test brainwave_analyzer
```

### Database Management

```bash
# Create migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Reset database
python manage.py flush
```

### Model Persistence

The model automatically saves its state and can be loaded on restart. Model files are stored in the project directory.

## Production Deployment

For production deployment, consider:

1. **Security**: Change `SECRET_KEY` and disable `DEBUG`
2. **Database**: Use PostgreSQL or MySQL instead of SQLite
3. **Static Files**: Configure static file serving
4. **WSGI**: Use Gunicorn or uWSGI
5. **Reverse Proxy**: Use Nginx for load balancing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
