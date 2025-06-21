import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict
import pickle
import os
from django.conf import settings

class BrainwaveCorrespondenceModel(nn.Module):
    """PyTorch model for analyzing brainwave correspondence"""
    
    def __init__(self, input_size: int = 100, hidden_size: int = 128, num_layers: int = 2):
        super(BrainwaveCorrespondenceModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),  # 2 waves concatenated
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, wave1: torch.Tensor, wave2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            wave1: First brainwave data (batch_size, sequence_length)
            wave2: Second brainwave data (batch_size, sequence_length)
        
        Returns:
            Correspondence probability (batch_size, 1)
        """
        # Concatenate the two waves
        combined = torch.cat([wave1, wave2], dim=1)
        
        # Extract features
        features = self.feature_extractor(combined)
        
        # Reshape for LSTM (batch_size, 1, hidden_size//2)
        features = features.unsqueeze(1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(lstm_out)
        
        return output
    
    def extract_features(self, wave1: torch.Tensor, wave2: torch.Tensor) -> Dict[str, float]:
        """Extract features from brainwave data for analysis"""
        with torch.no_grad():
            # Concatenate waves
            combined = torch.cat([wave1, wave2], dim=1)
            
            # Get features from feature extractor
            features = self.feature_extractor(combined)
            
            # Calculate statistical features
            wave1_mean = wave1.mean().item()
            wave1_std = wave1.std().item()
            wave2_mean = wave2.mean().item()
            wave2_std = wave2.std().item()
            
            # Correlation between waves
            correlation = torch.corrcoef(torch.stack([wave1.flatten(), wave2.flatten()]))[0, 1].item()
            
            # Frequency domain features (simplified)
            fft1 = torch.fft.fft(wave1)
            fft2 = torch.fft.fft(wave2)
            power1 = torch.abs(fft1).mean().item()
            power2 = torch.abs(fft2).mean().item()
            
            return {
                'wave1_mean': wave1_mean,
                'wave1_std': wave1_std,
                'wave2_mean': wave2_mean,
                'wave2_std': wave2_std,
                'correlation': correlation,
                'power1': power1,
                'power2': power2,
                'feature_vector': features.mean(dim=0).tolist()
            }

class BrainwaveAnalyzer:
    """Main class for brainwave analysis with online learning"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BrainwaveCorrespondenceModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
        # Training history
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'samples_seen': 0
        }
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_data(self, wave1: List[float], wave2: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess brainwave data for model input"""
        # Ensure both waves have the same length
        min_length = min(len(wave1), len(wave2))
        wave1 = wave1[:min_length]
        wave2 = wave2[:min_length]
        
        # Pad or truncate to required length
        sequence_length = settings.MODEL_SETTINGS['BRAINWAVE_SEQUENCE_LENGTH']
        
        if len(wave1) < sequence_length:
            # Pad with zeros
            wave1.extend([0.0] * (sequence_length - len(wave1)))
            wave2.extend([0.0] * (sequence_length - len(wave2)))
        else:
            # Truncate
            wave1 = wave1[:sequence_length]
            wave2 = wave2[:sequence_length]
        
        # Convert to tensors
        wave1_tensor = torch.tensor(wave1, dtype=torch.float32).unsqueeze(0).to(self.device)
        wave2_tensor = torch.tensor(wave2, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return wave1_tensor, wave2_tensor
    
    def analyze_correspondence(self, wave1: List[float], wave2: List[float]) -> Dict:
        """Analyze correspondence between two brainwaves"""
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess data
            wave1_tensor, wave2_tensor = self.preprocess_data(wave1, wave2)
            
            # Get prediction
            prediction = self.model(wave1_tensor, wave2_tensor)
            correspondence_score = prediction.item()
            
            # Extract features
            features = self.model.extract_features(wave1_tensor, wave2_tensor)
            
            # Determine if waves correspond (threshold at 0.5)
            corresponds = correspondence_score > 0.5
            
            # Calculate confidence (distance from decision boundary)
            confidence = abs(correspondence_score - 0.5) * 2
            
            return {
                'correspondence_score': correspondence_score,
                'corresponds': corresponds,
                'confidence': confidence,
                'features': features
            }
    
    def online_train(self, wave1: List[float], wave2: List[float], 
                    label: bool, learning_rate: float = 0.001) -> Dict:
        """Online training with a single sample"""
        self.model.train()
        
        # Preprocess data
        wave1_tensor, wave2_tensor = self.preprocess_data(wave1, wave2)
        label_tensor = torch.tensor([[float(label)]], dtype=torch.float32).to(self.device)
        
        # Forward pass
        prediction = self.model(wave1_tensor, wave2_tensor)
        
        # Calculate loss
        loss = self.criterion(prediction, label_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update training history
        self.training_history['losses'].append(loss.item())
        self.training_history['samples_seen'] += 1
        
        # Calculate accuracy
        predicted_label = prediction.item() > 0.5
        accuracy = 1.0 if predicted_label == label else 0.0
        self.training_history['accuracies'].append(accuracy)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'prediction': prediction.item(),
            'samples_seen': self.training_history['samples_seen']
        }
    
    def batch_train(self, data_batch: List[Tuple[List[float], List[float], bool]]) -> Dict:
        """Batch training with multiple samples"""
        self.model.train()
        
        total_loss = 0
        correct_predictions = 0
        
        for wave1, wave2, label in data_batch:
            # Preprocess data
            wave1_tensor, wave2_tensor = self.preprocess_data(wave1, wave2)
            label_tensor = torch.tensor([[float(label)]], dtype=torch.float32).to(self.device)
            
            # Forward pass
            prediction = self.model(wave1_tensor, wave2_tensor)
            
            # Calculate loss
            loss = self.criterion(prediction, label_tensor)
            total_loss += loss.item()
            
            # Count correct predictions
            predicted_label = prediction.item() > 0.5
            if predicted_label == label:
                correct_predictions += 1
        
        # Average loss
        avg_loss = total_loss / len(data_batch)
        accuracy = correct_predictions / len(data_batch)
        
        # Update training history
        self.training_history['losses'].append(avg_loss)
        self.training_history['accuracies'].append(accuracy)
        self.training_history['samples_seen'] += len(data_batch)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples_seen': self.training_history['samples_seen']
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
    
    def get_training_stats(self) -> Dict:
        """Get current training statistics"""
        if not self.training_history['losses']:
            return {
                'avg_loss': 0.0,
                'avg_accuracy': 0.0,
                'samples_seen': 0
            }
        
        return {
            'avg_loss': np.mean(self.training_history['losses'][-100:]),  # Last 100 samples
            'avg_accuracy': np.mean(self.training_history['accuracies'][-100:]),
            'samples_seen': self.training_history['samples_seen']
        }

# Global model instance
analyzer = BrainwaveAnalyzer() 
