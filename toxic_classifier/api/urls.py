# api/urls.py

from django.urls import path
from .views import PredictToxicComment

urlpatterns = [
    path('predict/', PredictToxicComment.as_view(), name='predict-toxic-comment'),
]
