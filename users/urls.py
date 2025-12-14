from django.urls import path
from .views import home, profile, RegisterView, predict_cancer

urlpatterns = [
    path('', home, name='users-home'),
    path('register/', RegisterView.as_view(), name='users-register'),
    path('profile/', profile, name='users-profile'),
    path("predict/", predict_cancer, name="predict"),
]
