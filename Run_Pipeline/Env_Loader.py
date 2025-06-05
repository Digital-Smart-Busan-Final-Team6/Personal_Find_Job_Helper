# env_loader.py
import os
from dotenv import load_dotenv

class EnvLoader:
    """
    .env 파일 사용 여부에 따라 환경 변수를 로드합니다.
    """
    @staticmethod
    def load_local(dotenv_path: str = "../.env"):
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            # 예시: 필요한 키들을 환경 변수로 설정
            os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
            os.environ['LANGSMITH_ENDPOINT'] = os.getenv('LANGSMITH_ENDPOINT')
            os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')
            os.environ['LANGSMITH_TRACING'] = os.getenv('LANGSMITH_TRACING')
            os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')

    @staticmethod
    def load_colab():
        # Colab 환경에서 userdata.get() 사용 예시
        from google.colab import userdata
        os.environ['OPENAI_API_KEY']       = userdata.get('OPENAI_API_KEY')
        os.environ['LANGSMITH_ENDPOINT']   = userdata.get('LANGSMITH_ENDPOINT2')
        os.environ['LANGSMITH_PROJECT']    = userdata.get('LANGSMITH_PROJECT2')
        os.environ['LANGSMITH_TRACING']    = userdata.get('LANGSMITH_TRACING')
        os.environ['LANGSMITH_API_KEY']    = userdata.get('LANGSMITH_API_KEY2')
