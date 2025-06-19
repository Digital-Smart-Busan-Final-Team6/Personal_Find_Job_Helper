from .Main_Pipeline import main as create_agent_from_pipeline

_agent_chain = {}

def get_agent_chain(mode: str = "chat"):
    if mode not in _agent_chain:
        _agent_chain[mode] = create_agent_from_pipeline(
            return_chain_only=True,
            mode=mode
        )
    return _agent_chain[mode]
