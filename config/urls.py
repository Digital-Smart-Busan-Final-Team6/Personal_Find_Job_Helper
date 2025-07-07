# config/urls.py
from django.contrib import admin
from django.urls import path
from main import views

urlpatterns = [
    # --- 기존 URL 패턴들은 그대로 유지 ---
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('chat_api', views.chat_api, name='chat_api'),
    path('resumes/', views.resume_list_view, name='resume_list'),
    path('resumes/add/', views.resume_add_view, name='resume_add'),
    path('resumes/edit/<int:resume_id>/', views.resume_edit_view, name='resume_edit'),
    path('resumes/delete/<int:resume_id>/', views.resume_delete_view, name='resume_delete'),
    path('resumes/export/json/', views.export_resumes_to_json, name='export_resumes_json'),
    path('resumes/set-selection/', views.set_selected_resumes, name='set_selected_resumes'),
    path('export-resumes-to-json/', views.export_resumes_to_json, name='export_resumes_to_json'),
    path('reports/job-search/', views.job_search_report_page, name='job_search_report'),
    path('reports/recommending/', views.recommend_recommending_view, name='recommend_recommending'),
    path('reports/results/', views.recommend_result_view, name='recommend_result'),
    path('reports/generate-final-report/', views.generate_final_report_view, name='generate_final_report'),
]