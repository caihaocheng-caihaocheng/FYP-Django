from django.urls import path
from . import views

app_name = 'NRE'
urlpatterns = [
    path('', views.index, name='index'),
    path('NER/', views.NER_page, name='NER'),
    

]