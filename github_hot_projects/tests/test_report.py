"""
测试 report 模块
================
覆盖：报告生成、LLM 描述缓存、文件输出格式。
"""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest


class TestReport:
    def test_generate_report_comprehensive(self, tmp_path):
        """综合模式报告生成 + 文件输出。"""
        recent_created_at = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        older_created_at = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
        db = {
            "date": "2026-04-17",
            "projects": {
                "hot-org/hot-repo": {
                    "desc": "",
                    "short_desc": "A hot repository",
                    "language": "Python",
                    "topics": ["ai", "agent"],
                    "created_at": older_created_at,
                    "refreshed_at": "2026-04-17T03:25:00Z",
                    "readme_url": "https://github.com/hot-org/hot-repo/blob/HEAD/README.md",
                },
                "new-org/new-repo": {
                    "desc": "",
                    "short_desc": "A new repository",
                    "language": "TypeScript",
                    "topics": ["web", "ui"],
                    "created_at": recent_created_at,
                    "refreshed_at": "2026-04-17T03:30:00Z",
                    "readme_url": "https://github.com/new-org/new-repo/blob/HEAD/README.md",
                },
            },
        }
        top_projects = [
            ("hot-org/hot-repo", {"growth": 2000, "star": 15000}),
            ("new-org/new-repo", {"growth": 800, "star": 1200}),
        ]

        with patch("github_hot_projects.report.REPORT_DIR", str(tmp_path)):
            with patch("github_hot_projects.report.call_llm_describe", return_value="测试描述内容"):
                from github_hot_projects.report import step3_generate_report
                path = step3_generate_report(top_projects, db, mode="comprehensive")

        assert path != ""
        assert os.path.exists(path)
        content = open(path, "r", encoding="utf-8").read()
        assert "GitHub 热门项目" in content
        assert "## 1. hot-org/hot-repo" in content
        assert "## 2. new-org/new-repo" in content
        assert "链接: https://github.com/hot-org/hot-repo" in content
        assert "- 创建时间: " in content
        assert "- 总 Star: 15,000" in content
        assert "- 近7天增长: +2,000" in content
        assert "- 项目状态: NEW（45天内）" in content
        assert "- 主题标签: ai, agent" in content
        assert "项目定位与用途" in content
        assert "解决的问题" in content
        assert "使用场景" in content
        assert "repo-card" not in content
        assert "repo-copy-btn" not in content
        assert "最近刷新" not in content
        assert "## 1. hot-org/hot-repo（+2000，⭐15000）" not in content
        assert "测试描述内容" in content

    def test_generate_report_hot_new(self, tmp_path):
        """新项目榜报告需区分创建窗口和增长统计窗口。"""
        recent_created_at = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        db = {
            "date": "2026-04-17",
            "projects": {
                "new-org/new-repo": {
                    "desc": "",
                    "short_desc": "A new repository",
                    "language": "TypeScript",
                    "topics": ["web"],
                    "created_at": recent_created_at,
                    "refreshed_at": "2026-04-17T03:30:00Z",
                },
            },
        }
        top_projects = [
            ("new-org/new-repo", {"growth": 800, "star": 1200}),
        ]

        with patch("github_hot_projects.report.REPORT_DIR", str(tmp_path)):
            with patch("github_hot_projects.report.call_llm_describe", return_value="新项目描述"):
                from github_hot_projects.report import step3_generate_report
                path = step3_generate_report(
                    top_projects,
                    db,
                    mode="hot_new",
                    new_project_days=30,
                )

        assert "_hot_new_30d.md" in path
        content = open(path, "r", encoding="utf-8").read()
        assert "新项目热度榜" in content
        assert "新项目创建窗口: <= 30 天" in content
        assert "增长统计窗口: 7 天" in content
        assert "## 1. new-org/new-repo" in content
        assert "⭐1200" not in content
        assert "- 总 Star: 1,200" in content
        assert "- 近7天增长: +800" in content
        assert "- 项目状态: NEW（30天内）" in content

    def test_generate_report_comprehensive_custom_time_window(self, tmp_path):
        db = {
            "date": "2026-04-17",
            "projects": {
                "hot-org/hot-repo": {
                    "desc": "",
                    "short_desc": "A hot repository",
                    "language": "Python",
                    "topics": ["ai"],
                    "created_at": "2026-04-01T00:00:00Z",
                    "refreshed_at": "2026-04-17T03:25:00Z",
                    "readme_url": "https://github.com/hot-org/hot-repo/blob/HEAD/README.md",
                },
            },
        }
        top_projects = [("hot-org/hot-repo", {"growth": 2000, "star": 15000})]

        with patch("github_hot_projects.report.REPORT_DIR", str(tmp_path)):
            with patch("github_hot_projects.report.call_llm_describe", return_value="测试描述内容"):
                from github_hot_projects.report import step3_generate_report
                path = step3_generate_report(
                    top_projects,
                    db,
                    mode="comprehensive",
                    time_window_days=10,
                )

        assert path.endswith("_10d.md")
        content = open(path, "r", encoding="utf-8").read()
        assert "增长统计窗口: 10 天" in content
        assert "- 近10天增长: +2,000" in content

    def test_generate_report_uses_db_cache(self, tmp_path):
        """DB 中已有描述的项目不调用 LLM。"""
        db = {
            "projects": {
                "cached/repo": {
                    "desc": "项目定位与用途：这是一个缓存项目。\n解决的问题：它减少重复劳动。\n使用场景：适合测试缓存命中。",
                    "star": 5000,
                },
            }
        }
        top_projects = [("cached/repo", {"growth": 1000, "star": 5000})]

        with patch("github_hot_projects.report.REPORT_DIR", str(tmp_path)):
            with patch("github_hot_projects.report.call_llm_describe") as mock_llm:
                from github_hot_projects.report import step3_generate_report
                path = step3_generate_report(top_projects, db, mode="comprehensive")
                mock_llm.assert_not_called()

        content = open(path, "r", encoding="utf-8").read()
        assert "## 1. cached/repo" in content
        assert "这是一个缓存项目" in content
        assert "它减少重复劳动" in content
        assert "适合测试缓存命中" in content

    def test_generate_report_empty_projects(self, tmp_path):
        """空项目列表 → 仍生成文件（仅有标题）。"""
        with patch("github_hot_projects.report.REPORT_DIR", str(tmp_path)):
            from github_hot_projects.report import step3_generate_report
            path = step3_generate_report([], {"projects": {}}, mode="comprehensive")
        assert os.path.exists(path)
        content = open(path, "r", encoding="utf-8").read()
        assert "共 0 个项目" in content
