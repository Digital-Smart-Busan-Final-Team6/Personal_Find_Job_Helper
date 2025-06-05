# config/urls.py
from django.contrib import admin
from django.urls import path
from main import views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),

    path('mypage/', views.mypage_view, name='mypage'), 
    path('logout/', views.logout_view, name='logout'), 
]
