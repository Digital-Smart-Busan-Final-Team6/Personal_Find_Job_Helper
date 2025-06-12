# config/urls.py
from django.contrib import admin
from django.urls import path
from main import views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'), 
    path('register/', views.register_view, name='register'),
    path('mypage/', views.mypage_view, name='mypage'), 
    path('chat/', views.chat_api, name='chat_api'),
    path('mypage/education/', views.mypage_education_view, name='mypage_education'),
    path('mypage/job/', views.mypage_job_view, name='mypage_job'),
    path('mypage/location/', views.mypage_location_view, name='mypage_location'),
    path('mypage/skills/', views.mypage_skills_view, name='mypage_skills'),
]
