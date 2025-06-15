# config/urls.py
from django.contrib import admin
from django.urls import path
from main import views

urlpatterns = [
    # 1. 관리자 페이지
    path('admin/', admin.site.urls),
    
    # 2. 챗봇이 있는 메인 페이지 ('/')
    path('', views.home, name='home'),
    
    # 3. 챗봇 API 엔드포인트
    path('chat_api', views.chat_api, name='chat_api'),

    # --- 이력서 관련 URL 패턴들 ---

    # 4. 이력서 목록을 보여주는 페이지 (예: http://127.0.0.1:8000/resumes/)
    #    Figma 디자인의 메인 화면입니다.
    path('resumes/', views.resume_list_view, name='resume_list'),
    
    # 5. 새 이력서를 추가하는 페이지 (예: http://127.0.0.1:8000/resumes/add/)
    #    '이력서 추가하기' 버튼을 누르면 이 주소로 이동합니다.
    path('resumes/add/', views.resume_add_view, name='resume_add'),
    
    # 6. 특정 이력서를 수정하는 페이지 (예: http://127.0.0.1:8000/resumes/edit/1/)
    #    <int:resume_id>는 숫자 형태의 이력서 ID를 받는다는 의미입니다.
    path('resumes/edit/<int:resume_id>/', views.resume_edit_view, name='resume_edit'),
    
    # 7. 특정 이력서를 삭제하는 처리 URL (예: http://127.0.0.1:8000/resumes/delete/1/)
    path('resumes/delete/<int:resume_id>/', views.resume_delete_view, name='resume_delete'),
    
    # 8. 모든 이력서를 JSON으로 내보내는 URL
    path('resumes/export/json/', views.export_resumes_to_json, name='export_resumes_json'),

    path('export-resumes-to-json/', views.export_resumes_to_json, name='export_resumes_to_json'),
]