"""
URL configuration for Aerosync project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from Aerosync import views

urlpatterns = [
    path('admin/', admin.site.urls),

    # add these to configure our home page (default view) and result web page
    path('', views.home, name='home'),
    path('flight_fare_prediction/', views.flight_fare_prediction, name='flight_fare_prediction'),
    path('flight_delay_prediction/', views.flight_delay_prediction, name='flight_delay_prediction'),
    path('flight_arrival_delay_result/', views.flight_arrival_delay_result, name='flight_arrival_delay_result'),
    path('flight_arrival_delay/', views.flight_arrival_delay, name='flight_arrival_delay'),
    path('flight_departure_delay_result/', views.flight_departure_delay_result, name='flight_departure_delay_result'),
    path('flight_departure_delay/', views.flight_departure_delay, name='flight_departure_delay'),
    path('fareresult/', views.fare_result, name='fare_result'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('generate_response/', views.generate_response, name='generate_response'),
    # path('delayresult/', views.delay_result, name='delay_result'),
]
