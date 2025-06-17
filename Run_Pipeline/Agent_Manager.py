# Run_Pipeline/Agent_Manager.py

# main 함수를  가져오기기
from .Main_Pipeline import main as create_agent_from_pipeline

# Agent 체인을 저장할 변수
agent_chain = None

def get_agent_chain():
    """
    Agent 체인을 초기화하고 반환하는 함수.
    이미 초기화되었다면 저장된 체인을 즉시 반환 (싱글톤 패턴)
    """
    global agent_chain
    if agent_chain is None:
        print("Agent 체인을 최초로 초기화합니다. 잠시만 기다려주세요...")
        # main 함수를 return_chain_only=True 옵션으로 호출
        agent_chain = create_agent_from_pipeline(return_chain_only=True)
        print("✅ Agent 체인 초기화 완료!")
    return agent_chain

# 이 파일은 직접 실행되지 않습니다.