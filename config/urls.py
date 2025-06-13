# config/urls.py
from django.contrib import admin
from django.urls import path
from main import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('chat/', views.chat_api, name='chat_api'),
    
    # mypage URL은 이제 단 하나입니다.
    path('mypage/', views.mypage_view, name='mypage'),
]