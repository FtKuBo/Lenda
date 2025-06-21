from django.contrib import admin
from .models import BrainwaveData, AnalysisResult, ModelMetrics

@admin.register(BrainwaveData)
class BrainwaveDataAdmin(admin.ModelAdmin):
    list_display = ('id', 'timestamp', 'sample_rate', 'sequence_length')
    list_filter = ('timestamp', 'sample_rate')
    search_fields = ('id',)
    readonly_fields = ('timestamp',)

@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'brainwave_data', 'correspondence_score', 'confidence', 'prediction', 'timestamp')
    list_filter = ('prediction', 'timestamp', 'confidence')
    search_fields = ('id', 'brainwave_data__id')
    readonly_fields = ('timestamp', 'features')

@admin.register(ModelMetrics)
class ModelMetricsAdmin(admin.ModelAdmin):
    list_display = ('id', 'model_version', 'accuracy', 'loss', 'training_samples', 'training_time', 'timestamp')
    list_filter = ('model_version', 'timestamp')
    search_fields = ('model_version',)
    readonly_fields = ('timestamp',)
