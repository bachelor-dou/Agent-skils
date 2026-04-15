"""
测试 report 模块
================
覆盖：报告生成、LLM 描述缓存、文件输出格式。
"""

import os
from unittest.mock import patch, MagicMock

import pytest


class TestReport:
    def test_generate_report_comprehensive(self, tmp_path, sample_db):
        """综合模式报告生成 + 文件输出。"""
        top_projects = [
            ("hot-org/hot-repo", {"growth": 2000, "star": 15000}),
            ("new-org/new-repo", {"growth": 800, "star": 1200}),
        ]

        with patch("github_hot_projects.report.REPORT_DIR", str(tmp_path)):
            with patch("github_hot_projects.report.call_llm_describe", return_value="测试描述内容"):
                from github_hot_projects.report import step3_generate_report
                path = step3_generate_report(top_projects, sample_db, mode="comprehensive")

        assert path != ""
        assert os.path.exists(path)
        content = open(path, "r", encoding="utf-8").read()
        assert "GitHub 热门项目" in content
        assert "hot-org/hot-repo" in content
        assert "+2000" in content
        assert "测试描述内容" in content

    def test_generate_report_hot_new(self, tmp_path, sample_db):
        """新项目榜报告文件名含 _hot_new 后缀。"""
        top_projects = [
            ("new-org/new-repo", {"growth": 800, "star": 1200}),
        ]

        with patch("github_hot_projects.report.REPORT_DIR", str(tmp_path)):
            with patch("github_hot_projects.report.call_llm_describe", return_value="新项目描述"):
                from github_hot_projects.report import step3_generate_report
                path = step3_generate_report(top_projects, sample_db, mode="hot_new")

        assert "_hot_new.md" in path
        content = open(path, "r", encoding="utf-8").read()
        assert "新项目热度榜" in content

    def test_generate_report_uses_db_cache(self, tmp_path):
        """DB 中已有描述的项目不调用 LLM。"""
        db = {
            "projects": {
                "cached/repo": {"desc": "已缓存的描述", "star": 5000},
            }
        }
        top_projects = [("cached/repo", {"growth": 1000, "star": 5000})]

        with patch("github_hot_projects.report.REPORT_DIR", str(tmp_path)):
            with patch("github_hot_projects.report.call_llm_describe") as mock_llm:
                from github_hot_projects.report import step3_generate_report
                path = step3_generate_report(top_projects, db, mode="comprehensive")
                mock_llm.assert_not_called()

        content = open(path, "r", encoding="utf-8").read()
        assert "已缓存的描述" in content

    def test_generate_report_empty_projects(self, tmp_path):
        """空项目列表 → 仍生成文件（仅有标题）。"""
        with patch("github_hot_projects.report.REPORT_DIR", str(tmp_path)):
            from github_hot_projects.report import step3_generate_report
            path = step3_generate_report([], {"projects": {}}, mode="comprehensive")
        assert os.path.exists(path)
        content = open(path, "r", encoding="utf-8").read()
        assert "共 0 个项目" in content
