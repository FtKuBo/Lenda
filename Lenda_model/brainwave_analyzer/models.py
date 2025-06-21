from django.db import models
from django.utils import timezone
import json

# Create your models here.

class BrainwaveData(models.Model):
    """Model to store brainwave data streams"""
    timestamp = models.DateTimeField(default=timezone.now)
    wave1_data = models.JSONField()  # Store as JSON array
    wave2_data = models.JSONField()  # Store as JSON array
    sample_rate = models.IntegerField(default=256)
    sequence_length = models.IntegerField(default=100)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Brainwave Data {self.id} - {self.timestamp}"

class AnalysisResult(models.Model):
    """Model to store analysis results"""
    brainwave_data = models.ForeignKey(BrainwaveData, on_delete=models.CASCADE, related_name='results')
    timestamp = models.DateTimeField(default=timezone.now)
    correspondence_score = models.FloatField()  # 0-1 score indicating correspondence
    confidence = models.FloatField()  # Model confidence in prediction
    prediction = models.BooleanField()  # True if waves correspond, False otherwise
    features = models.JSONField(default=dict)  # Extracted features
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Analysis {self.id} - Score: {self.correspondence_score:.3f}"

class ModelMetrics(models.Model):
    """Model to store training metrics and model performance"""
    timestamp = models.DateTimeField(default=timezone.now)
    model_version = models.CharField(max_length=50)
    accuracy = models.FloatField()
    loss = models.FloatField()
    training_samples = models.IntegerField()
    validation_samples = models.IntegerField()
    training_time = models.FloatField()  # in seconds
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Model {self.model_version} - Acc: {self.accuracy:.3f}"
