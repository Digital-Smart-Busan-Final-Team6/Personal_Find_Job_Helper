from django.apps import AppConfig
import os


class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    def ready(self):
        """
        Django 앱이 준비되었을 때 호출되는 메소드.
        서버 시작 시 단 한 번 실행됩니다.
        """

        if os.environ.get('RUN_MAIN') == 'true':
            return

        print("Django 앱 준비 시작...")
        # Run_Pipeline에서 Agent 생성 함수를 가져옵니다.
        from Run_Pipeline.Agent_Manager import get_agent_chain
        
        print("Agent 초기화를 시작합니다...")
        # get_agent_chain() 함수를 호출하여 전역 변수에 Agent를 로드합니다.
        get_agent_chain(mode = "chat")
        print("✅ MainConfig.ready(): Agent가 성공적으로 준비되었습니다.")
