from hot_projects.agent.state import AgentState


def test_state_defaults():
    s = AgentState(db={"projects": {}})
    assert s.ranking_cache is not None
    assert s.active_repo is None
    assert s.pending_confirmation_signature is None
    assert isinstance(s.conversation, list)
