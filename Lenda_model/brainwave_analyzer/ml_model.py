import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict
import pickle
import os
from django.conf import settings

class BrainwaveCorrespondenceModel(nn.Module):
    """PyTorch model for analyzing brainwave correspondence with multi-channel data"""
    
    def __init__(self, num_pods: int = 8, hidden_size: int = 128, num_layers: int = 2):
        super(BrainwaveCorrespondenceModel, self).__init__()
        
        # Get sequence length from Django settings with fallback
        try:
            self.sequence_length = settings.MODEL_SETTINGS['BRAINWAVE_SEQUENCE_LENGTH']
        except (AttributeError, KeyError):
            # Fallback to default value if Django settings are not available
            self.sequence_length = 100
        
        self.num_pods = num_pods
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Pod-specific feature extractors (each pod represents a different brain area)
        self.pod_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.sequence_length, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, hidden_size // 4)
            ) for _ in range(num_pods)
        ])
        
        # LSTM for temporal patterns within each pod
        self.pod_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=1,  # Single time point
                hidden_size=hidden_size // 4,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2
            ) for _ in range(num_pods)
        ])
        
        # Spatial attention mechanism to learn relationships between brain areas
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_size // 4,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Cross-wave attention to compare corresponding pods between waves
        self.cross_wave_attention = nn.MultiheadAttention(
            embed_dim=hidden_size // 4,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature fusion layer to combine pod features
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # Pod-specific correspondence classifiers
        self.pod_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_pods)
        ])
        
        # Global correspondence classifier
        self.global_classifier = nn.Sequential(
            nn.Linear(hidden_size * num_pods, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
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
    
    def forward(self, wave1: torch.Tensor, wave2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            wave1: First brainwave data (batch_size, num_pods, sequence_length)
            wave2: Second brainwave data (batch_size, num_pods, sequence_length)
        
        Returns:
            Dictionary containing global and pod-specific correspondence scores
        """
        batch_size = wave1.size(0)
        
        # Extract features for each pod in wave1
        wave1_pod_features = []
        for i in range(self.num_pods):
            pod_data = wave1[:, i, :]  # (batch_size, sequence_length)
            
            # Process temporal patterns with LSTM
            pod_data_reshaped = pod_data.unsqueeze(-1)  # (batch_size, sequence_length, 1)
            lstm_out, _ = self.pod_lstms[i](pod_data_reshaped)  # (batch_size, sequence_length, hidden_size//4)
            lstm_features = lstm_out[:, -1, :]  # Take last LSTM output (batch_size, hidden_size//4)
            
            # Combine with static features
            static_features = self.pod_extractors[i](pod_data)  # (batch_size, hidden_size//4)
            combined_features = (lstm_features + static_features) / 2  # Average both features
            wave1_pod_features.append(combined_features)
        
        # Extract features for each pod in wave2
        wave2_pod_features = []
        for i in range(self.num_pods):
            pod_data = wave2[:, i, :]  # (batch_size, sequence_length)
            
            # Process temporal patterns with LSTM
            pod_data_reshaped = pod_data.unsqueeze(-1)  # (batch_size, sequence_length, 1)
            lstm_out, _ = self.pod_lstms[i](pod_data_reshaped)  # (batch_size, sequence_length, hidden_size//4)
            lstm_features = lstm_out[:, -1, :]  # Take last LSTM output (batch_size, hidden_size//4)
            
            # Combine with static features
            static_features = self.pod_extractors[i](pod_data)  # (batch_size, hidden_size//4)
            combined_features = (lstm_features + static_features) / 2  # Average both features
            wave2_pod_features.append(combined_features)
        
        # Stack pod features for spatial attention
        wave1_pod_stack = torch.stack(wave1_pod_features, dim=1)  # (batch_size, num_pods, hidden_size//4)
        wave2_pod_stack = torch.stack(wave2_pod_features, dim=1)  # (batch_size, num_pods, hidden_size//4)
        
        # Apply spatial attention within each wave
        wave1_attended, _ = self.spatial_attention(wave1_pod_stack, wave1_pod_stack, wave1_pod_stack)
        wave2_attended, _ = self.spatial_attention(wave2_pod_stack, wave2_pod_stack, wave2_pod_stack)
        
        # Cross-wave attention to compare corresponding pods
        cross_attended, _ = self.cross_wave_attention(wave1_attended, wave2_attended, wave2_attended)
        
        # Feature fusion for each pod
        fused_features = []
        for i in range(self.num_pods):
            pod_features = cross_attended[:, i, :]  # (batch_size, hidden_size//4)
            fused_pod = self.feature_fusion(pod_features)  # (batch_size, hidden_size)
            fused_features.append(fused_pod)
        
        fused_features = torch.stack(fused_features, dim=1)  # (batch_size, num_pods, hidden_size)
        
        # Pod-specific correspondence scores
        pod_scores = []
        for i in range(self.num_pods):
            pod_features = fused_features[:, i, :]  # (batch_size, hidden_size)
            pod_score = self.pod_classifiers[i](pod_features)  # (batch_size, 1)
            pod_scores.append(pod_score)
        
        # Global correspondence score
        global_features = fused_features.view(batch_size, -1)  # Flatten all pod features
        global_score = self.global_classifier(global_features)  # (batch_size, 1)
        
        return {
            'global_score': global_score,
            'pod_scores': torch.cat(pod_scores, dim=1),  # (batch_size, num_pods)
            'pod_features': fused_features  # (batch_size, num_pods, hidden_size)
        }
    
    def extract_features(self, wave1: torch.Tensor, wave2: torch.Tensor) -> Dict[str, float]:
        """Extract features from brainwave data for analysis"""
        with torch.no_grad():
            # Get model outputs
            outputs = self.forward(wave1, wave2)
            
            # Calculate statistical features for each pod
            pod_features = {}
            for i in range(self.num_pods):
                wave1_pod = wave1[:, i, :]
                wave2_pod = wave2[:, i, :]
                
                pod_features[f'pod_{i}_wave1_mean'] = wave1_pod.mean().item()
                pod_features[f'pod_{i}_wave1_std'] = wave1_pod.std().item()
                pod_features[f'pod_{i}_wave2_mean'] = wave2_pod.mean().item()
                pod_features[f'pod_{i}_wave2_std'] = wave2_pod.std().item()
                
                # Correlation between corresponding pods
                correlation = torch.corrcoef(torch.stack([wave1_pod.flatten(), wave2_pod.flatten()]))[0, 1].item()
                pod_features[f'pod_{i}_correlation'] = correlation
                
                # Frequency domain features
                fft1 = torch.fft.fft(wave1_pod)
                fft2 = torch.fft.fft(wave2_pod)
                pod_features[f'pod_{i}_power1'] = torch.abs(fft1).mean().item()
                pod_features[f'pod_{i}_power2'] = torch.abs(fft2).mean().item()
            
            # Global features
            global_correlation = torch.corrcoef(torch.stack([wave1.flatten(), wave2.flatten()]))[0, 1].item()
            
            return {
                **pod_features,
                'global_correlation': global_correlation,
                'global_score': outputs['global_score'].item(),
                'pod_scores': outputs['pod_scores'].squeeze().tolist(),
                'feature_vector': outputs['pod_features'].mean(dim=(0, 1)).tolist()
            }

class BrainwaveAnalyzer:
    """Main class for brainwave analysis with online learning for multi-channel data"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BrainwaveCorrespondenceModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
        # Training history
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'pod_accuracies': [[] for _ in range(8)],  # Track accuracy per pod
            'samples_seen': 0
        }
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def update_sequence_length(self, new_sequence_length: int):
        """Update the model's sequence length and reinitialize if necessary"""
        if self.model.sequence_length != new_sequence_length:
            # Create new model with updated sequence length
            self.model = BrainwaveCorrespondenceModel(
                num_pods=self.model.num_pods,
                hidden_size=self.model.hidden_size,
                num_layers=self.model.num_layers
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            print(f"Model reinitialized with sequence length: {self.model.sequence_length}")
    
    def preprocess_data(self, wave1: List[List[float]], wave2: List[List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess multi-channel brainwave data for model input"""
        # Ensure both waves have the same number of pods
        num_pods = min(len(wave1), len(wave2))
        wave1 = wave1[:num_pods]
        wave2 = wave2[:num_pods]
        
        # Get sequence length from the model
        sequence_length = self.model.sequence_length
        
        processed_wave1 = []
        processed_wave2 = []
        
        for pod_idx in range(num_pods):
            pod1_data = wave1[pod_idx]
            pod2_data = wave2[pod_idx]
            
            # Ensure both pods have the same length
            min_length = min(len(pod1_data), len(pod2_data))
            pod1_data = pod1_data[:min_length]
            pod2_data = pod2_data[:min_length]
            
            # Pad or truncate to required length
            if len(pod1_data) < sequence_length:
                # Pad with zeros
                pod1_data.extend([0.0] * (sequence_length - len(pod1_data)))
                pod2_data.extend([0.0] * (sequence_length - len(pod2_data)))
            else:
                # Truncate
                pod1_data = pod1_data[:sequence_length]
                pod2_data = pod2_data[:sequence_length]
            
            processed_wave1.append(pod1_data)
            processed_wave2.append(pod2_data)
        
        # Convert to tensors: (batch_size=1, num_pods, sequence_length)
        wave1_tensor = torch.tensor([processed_wave1], dtype=torch.float32).to(self.device)
        wave2_tensor = torch.tensor([processed_wave2], dtype=torch.float32).to(self.device)
        
        return wave1_tensor, wave2_tensor
    
    def analyze_correspondence(self, wave1: List[List[float]], wave2: List[List[float]]) -> Dict:
        """Analyze correspondence between two multi-channel brainwaves"""
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess data
            wave1_tensor, wave2_tensor = self.preprocess_data(wave1, wave2)
            
            # Get predictions
            outputs = self.model(wave1_tensor, wave2_tensor)
            global_score = outputs['global_score'].item()
            pod_scores = outputs['pod_scores'].squeeze().tolist()
            
            # Extract features
            features = self.model.extract_features(wave1_tensor, wave2_tensor)
            
            # Determine if waves correspond (threshold at 0.5)
            global_corresponds = global_score > 0.5
            pod_corresponds = [score > 0.5 for score in pod_scores]
            
            # Calculate confidence
            global_confidence = abs(global_score - 0.5) * 2
            pod_confidences = [abs(score - 0.5) * 2 for score in pod_scores]
            
            return {
                'global_correspondence_score': global_score,
                'global_corresponds': global_corresponds,
                'global_confidence': global_confidence,
                'pod_correspondence_scores': pod_scores,
                'pod_corresponds': pod_corresponds,
                'pod_confidences': pod_confidences,
                'features': features
            }
    
    def online_train(self, wave1: List[List[float]], wave2: List[List[float]], 
                    label: bool, learning_rate: float = 0.001) -> Dict:
        """Online training with a single multi-channel sample"""
        self.model.train()
        
        # Preprocess data
        wave1_tensor, wave2_tensor = self.preprocess_data(wave1, wave2)
        label_tensor = torch.tensor([[float(label)]], dtype=torch.float32).to(self.device)
        
        # Forward pass
        outputs = self.model(wave1_tensor, wave2_tensor)
        global_prediction = outputs['global_score']
        pod_predictions = outputs['pod_scores']
        
        # Calculate global loss
        global_loss = self.criterion(global_prediction, label_tensor)
        
        # Calculate pod-specific losses (optional: can be weighted differently)
        pod_losses = []
        for i in range(pod_predictions.size(1)):
            pod_loss = self.criterion(pod_predictions[:, i:i+1], label_tensor)
            pod_losses.append(pod_loss)
        
        # Combined loss (global + average pod losses)
        total_loss = global_loss + 0.5 * torch.mean(torch.stack(pod_losses))
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Update training history
        self.training_history['losses'].append(total_loss.item())
        self.training_history['samples_seen'] += 1
        
        # Calculate accuracies
        global_predicted = global_prediction.item() > 0.5
        global_accuracy = 1.0 if global_predicted == label else 0.0
        self.training_history['accuracies'].append(global_accuracy)
        
        # Calculate pod-specific accuracies
        pod_accuracies = []
        for i in range(pod_predictions.size(1)):
            pod_predicted = pod_predictions[0, i].item() > 0.5
            pod_accuracy = 1.0 if pod_predicted == label else 0.0
            self.training_history['pod_accuracies'][i].append(pod_accuracy)
            pod_accuracies.append(pod_accuracy)
        
        return {
            'loss': total_loss.item(),
            'global_accuracy': global_accuracy,
            'pod_accuracies': pod_accuracies,
            'global_prediction': global_prediction.item(),
            'pod_predictions': pod_predictions.squeeze().tolist(),
            'samples_seen': self.training_history['samples_seen']
        }
    
    def batch_train(self, data_batch: List[Tuple[List[List[float]], List[List[float]], bool]]) -> Dict:
        """Batch training with multiple multi-channel samples"""
        self.model.train()
        
        total_loss = 0
        correct_predictions = 0
        pod_correct_predictions = [0] * 8
        
        for wave1, wave2, label in data_batch:
            # Preprocess data
            wave1_tensor, wave2_tensor = self.preprocess_data(wave1, wave2)
            label_tensor = torch.tensor([[float(label)]], dtype=torch.float32).to(self.device)
            
            # Forward pass
            outputs = self.model(wave1_tensor, wave2_tensor)
            global_prediction = outputs['global_score']
            pod_predictions = outputs['pod_scores']
            
            # Calculate losses
            global_loss = self.criterion(global_prediction, label_tensor)
            pod_losses = []
            for i in range(pod_predictions.size(1)):
                pod_loss = self.criterion(pod_predictions[:, i:i+1], label_tensor)
                pod_losses.append(pod_loss)
            
            total_loss += global_loss + 0.5 * torch.mean(torch.stack(pod_losses))
            
            # Count correct predictions
            global_predicted = global_prediction.item() > 0.5
            if global_predicted == label:
                correct_predictions += 1
            
            # Count pod-specific correct predictions
            for i in range(pod_predictions.size(1)):
                pod_predicted = pod_predictions[0, i].item() > 0.5
                if pod_predicted == label:
                    pod_correct_predictions[i] += 1
        
        # Average loss and accuracy
        avg_loss = total_loss.item() / len(data_batch)
        global_accuracy = correct_predictions / len(data_batch)
        pod_accuracies = [correct / len(data_batch) for correct in pod_correct_predictions]
        
        # Update training history
        self.training_history['losses'].append(avg_loss)
        self.training_history['accuracies'].append(global_accuracy)
        for i, acc in enumerate(pod_accuracies):
            self.training_history['pod_accuracies'][i].append(acc)
        self.training_history['samples_seen'] += len(data_batch)
        
        return {
            'loss': avg_loss,
            'global_accuracy': global_accuracy,
            'pod_accuracies': pod_accuracies,
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
                'avg_global_accuracy': 0.0,
                'avg_pod_accuracies': [0.0] * 8,
                'samples_seen': 0
            }
        
        # Calculate averages for last 100 samples
        recent_losses = self.training_history['losses'][-100:]
        recent_accuracies = self.training_history['accuracies'][-100:]
        
        pod_accuracies = []
        for pod_acc_history in self.training_history['pod_accuracies']:
            if pod_acc_history:
                pod_accuracies.append(np.mean(pod_acc_history[-100:]))
            else:
                pod_accuracies.append(0.0)
        
        return {
            'avg_loss': np.mean(recent_losses),
            'avg_global_accuracy': np.mean(recent_accuracies),
            'avg_pod_accuracies': pod_accuracies,
            'samples_seen': self.training_history['samples_seen']
        }

# Global model instance
analyzer = BrainwaveAnalyzer() 
