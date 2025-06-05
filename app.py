# app.py
from Run_Pipeline.Run_Model import main  # Run_Model.py 의 main(return_chain_only=True)를 불러온다

def get_chain():
    # return_chain_only=True 로 호출하면 chain 객체만 반환
    return main(return_chain_only=True)
