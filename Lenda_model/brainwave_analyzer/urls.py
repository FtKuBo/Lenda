from django.urls import path
from . import views

urlpatterns = [
    path('analyze/', views.analyze_brainwaves, name='analyze_brainwaves'),
    path('train/', views.train_model, name='train_model'),
    path('batch-train/', views.batch_train, name='batch_train'),
    path('stats/', views.get_model_stats, name='get_model_stats'),
    path('history/', views.get_analysis_history, name='get_analysis_history'),
    path('health/', views.health_check, name='health_check'),
    path('sample-data/', views.generate_sample_data, name='generate_sample_data'),
] 
