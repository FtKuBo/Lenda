from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
import json
import numpy as np
from .models import BrainwaveData, AnalysisResult, ModelMetrics
from .ml_model import analyzer
import threading
import time

@csrf_exempt
@require_http_methods(["POST"])
def analyze_brainwaves(request):
    """Analyze correspondence between two brainwaves"""
    try:
        data = json.loads(request.body)
        wave1 = data.get('wave1', [])
        wave2 = data.get('wave2', [])
        
        if not wave1 or not wave2:
            return JsonResponse({
                'error': 'Both wave1 and wave2 data are required'
            }, status=400)
        
        # Store the brainwave data
        brainwave_data = BrainwaveData.objects.create(
            wave1_data=wave1,
            wave2_data=wave2,
            sample_rate=data.get('sample_rate', 256),
            sequence_length=len(wave1)
        )
        
        # Analyze correspondence
        result = analyzer.analyze_correspondence(wave1, wave2)
        
        # Store the analysis result
        analysis_result = AnalysisResult.objects.create(
            brainwave_data=brainwave_data,
            correspondence_score=result['correspondence_score'],
            confidence=result['confidence'],
            prediction=result['corresponds'],
            features=result['features']
        )
        
        return JsonResponse({
            'id': analysis_result.id,
            'correspondence_score': result['correspondence_score'],
            'corresponds': result['corresponds'],
            'confidence': result['confidence'],
            'features': result['features'],
            'timestamp': analysis_result.timestamp.isoformat()
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def train_model(request):
    """Train the model with labeled data"""
    try:
        data = json.loads(request.body)
        wave1 = data.get('wave1', [])
        wave2 = data.get('wave2', [])
        label = data.get('label')  # True if waves correspond, False otherwise
        
        if not wave1 or not wave2 or label is None:
            return JsonResponse({
                'error': 'wave1, wave2, and label are required'
            }, status=400)
        
        # Online training
        training_result = analyzer.online_train(wave1, wave2, label)
        
        # Store training metrics
        ModelMetrics.objects.create(
            model_version='v1.0',
            accuracy=training_result['accuracy'],
            loss=training_result['loss'],
            training_samples=1,
            validation_samples=0,
            training_time=0.1  # Approximate time for single sample
        )
        
        return JsonResponse({
            'loss': training_result['loss'],
            'accuracy': training_result['accuracy'],
            'prediction': training_result['prediction'],
            'samples_seen': training_result['samples_seen']
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
def get_model_stats(request):
    """Get current model statistics"""
    try:
        stats = analyzer.get_training_stats()
        
        # Get recent analysis results
        recent_results = AnalysisResult.objects.all()[:10]
        recent_data = [{
            'id': result.id,
            'correspondence_score': result.correspondence_score,
            'confidence': result.confidence,
            'prediction': result.prediction,
            'timestamp': result.timestamp.isoformat()
        } for result in recent_results]
        
        return JsonResponse({
            'model_stats': stats,
            'recent_analyses': recent_data,
            'total_analyses': AnalysisResult.objects.count(),
            'total_training_samples': stats['samples_seen']
        })
        
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
def get_analysis_history(request):
    """Get analysis history with pagination"""
    try:
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 20))
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        results = AnalysisResult.objects.all()[start_idx:end_idx]
        
        data = [{
            'id': result.id,
            'correspondence_score': result.correspondence_score,
            'confidence': result.confidence,
            'prediction': result.prediction,
            'timestamp': result.timestamp.isoformat(),
            'wave1_length': len(result.brainwave_data.wave1_data),
            'wave2_length': len(result.brainwave_data.wave2_data)
        } for result in results]
        
        return JsonResponse({
            'results': data,
            'page': page,
            'page_size': page_size,
            'total_count': AnalysisResult.objects.count()
        })
        
    except ValueError:
        return JsonResponse({
            'error': 'Invalid page or page_size parameters'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def batch_train(request):
    """Batch training with multiple samples"""
    try:
        data = json.loads(request.body)
        training_data = data.get('training_data', [])
        
        if not training_data:
            return JsonResponse({
                'error': 'training_data is required'
            }, status=400)
        
        # Prepare batch data
        batch = []
        for item in training_data:
            wave1 = item.get('wave1', [])
            wave2 = item.get('wave2', [])
            label = item.get('label')
            
            if wave1 and wave2 and label is not None:
                batch.append((wave1, wave2, label))
        
        if not batch:
            return JsonResponse({
                'error': 'No valid training samples found'
            }, status=400)
        
        # Batch training
        start_time = time.time()
        training_result = analyzer.batch_train(batch)
        training_time = time.time() - start_time
        
        # Store training metrics
        ModelMetrics.objects.create(
            model_version='v1.0',
            accuracy=training_result['accuracy'],
            loss=training_result['loss'],
            training_samples=len(batch),
            validation_samples=0,
            training_time=training_time
        )
        
        return JsonResponse({
            'loss': training_result['loss'],
            'accuracy': training_result['accuracy'],
            'samples_trained': len(batch),
            'total_samples_seen': training_result['samples_seen'],
            'training_time': training_time
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint"""
    return JsonResponse({
        'status': 'healthy',
        'model_loaded': True,
        'timestamp': timezone.now().isoformat()
    })

@require_http_methods(["GET"])
def generate_sample_data(request):
    """Generate sample brainwave data for testing"""
    try:
        # Generate sample brainwave data
        sample_rate = 256
        duration = 1  # 1 second
        t = np.linspace(0, duration, sample_rate)
        
        # Generate two similar waves (should correspond)
        wave1 = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
        wave2 = np.sin(2 * np.pi * 10 * t + 0.1) + 0.1 * np.random.randn(len(t))
        
        # Generate two different waves (should not correspond)
        wave3 = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))
        wave4 = np.cos(2 * np.pi * 20 * t) + 0.1 * np.random.randn(len(t))
        
        return JsonResponse({
            'corresponding_waves': {
                'wave1': wave1.tolist(),
                'wave2': wave2.tolist(),
                'label': True
            },
            'non_corresponding_waves': {
                'wave1': wave3.tolist(),
                'wave2': wave4.tolist(),
                'label': False
            },
            'sample_rate': sample_rate,
            'duration': duration
        })
        
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)
