import inspect
from pathlib import Path

import pytest

from kairorag.cloud import index as cloud_index_cli
from kairorag.cloud import query_cli
from kairorag.demos import run_batch_eval, run_single_query
from kairorag.indexing import build_index
from kairorag.scripts import build_index as script_build_index


def test_legacy_run_single_query_only_prints_deprecated_message(capsys):
    assert run_single_query.main(["--question", "旧入口还能用吗？"]) == 2

    output = capsys.readouterr().out
    assert "已废弃" in output
    assert "kairo query" in output


def test_legacy_build_index_only_prints_deprecated_message(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        build_index.build_indexes(tmp_path, tmp_path)

    assert exc_info.value.code == 2
    output = capsys.readouterr().out
    assert "已废弃" in output
    assert not (tmp_path / "keyword_index.pkl").exists()
    assert not (tmp_path / "vector_store.pkl").exists()


def test_legacy_build_index_script_paths_are_deprecated(capsys):
    assert build_index.main([]) == 2
    assert script_build_index.main([]) == 2
    assert run_batch_eval.main([]) == 2

    output = capsys.readouterr().out
    assert output.count("已废弃") == 3


def test_cloud_query_source_does_not_import_legacy_react_agent():
    source = inspect.getsource(query_cli) + inspect.getsource(cloud_index_cli)

    assert "kairorag.agent.react_agent" not in source
    assert "ReactAgent" not in source


def test_readme_marks_legacy_commands_as_deprecated_only():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "旧 offline demo 与旧 pickle index 不再是支持入口" in text
    assert "kairo query" in text
    assert "python -m kairorag.demos.run_single_query 仍是 legacy demo" not in text
