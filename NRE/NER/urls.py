from django.urls import path
from . import views

app_name = 'NRE'
urlpatterns = [
    path('', views.index, name='index'),
    path('SRE/', views.NER_page, name='SRE'),
    path('DRE/', views.Document_page, name='DRE')
    

]