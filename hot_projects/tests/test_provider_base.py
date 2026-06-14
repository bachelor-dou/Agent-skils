import pytest

from hot_projects.providers.base import Repo, Provider


def test_repo_from_github_item():
    item = {
        "full_name": "a/b", "stargazers_count": 1500, "description": "x",
        "language": "Python", "topics": ["ai"], "created_at": "2026-01-01T00:00:00Z",
        "forks_count": 12,
    }
    r = Repo.from_github(item)
    assert r.full_name == "a/b"
    assert r.star == 1500
    assert r.language == "Python"
    assert r.topics == ["ai"]
    assert r.forks == 12
    assert r.created_at.startswith("2026-01-01")
    assert r.raw is item


def test_repo_from_internal_dict_uses_star_key():
    r = Repo.from_github({"full_name": "a/b", "star": 999})
    assert r.star == 999


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        Provider()
