"""本地仓库卫生和发布前风险检查。"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SECRET_ASSIGNMENT_RE = re.compile(
    r"(OPENAI_API_KEY|COHERE_API_KEY|TAVILY_API_KEY|SERPAPI_API_KEY|BING_API_KEY|VOYAGE_API_KEY|JINA_API_KEY)"
    r"\s*=\s*([^\s#]+)"
)
SECRET_VALUE_RE = re.compile(r"sk-[A-Za-z0-9_\-]{12,}")
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".example", ".cfg", ".ini"}
GENERATED_PATTERNS = [
    "data/*.jsonl",
    "data/*trace*",
    "data/*audit*",
    "data/eval_outputs/*",
    "eval_outputs/*",
    "results/eval_dashboard/*",
    "htmlcov/*",
    ".coverage",
    "*.pkl",
    "keyword_index.pkl",
    "vector_store.pkl",
]
REQUIRED_GITIGNORE_RULES = [
    ".env",
    ".env.*",
    "!.env.example",
    "data/*.jsonl",
    "data/*trace*",
    "data/*audit*",
    "data/eval_outputs/",
    "eval_outputs/",
    "htmlcov/",
    ".coverage",
    "*.pkl",
    "keyword_index.pkl",
    "vector_store.pkl",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".pyright/",
    "__pycache__/",
]


@dataclass(frozen=True)
class RepoHygieneIssue:
    code: str
    severity: str
    message: str
    path: str | None = None
    suggestion: str | None = None


@dataclass(frozen=True)
class RepoHygieneReport:
    ok: bool
    issues: list[RepoHygieneIssue]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RepoHygieneChecker:
    """检查本地仓库是否适合进入发布流程。"""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or Path.cwd()).resolve()

    def run(self) -> RepoHygieneReport:
        issues: list[RepoHygieneIssue] = []
        issues.extend(self._check_env_files())
        issues.extend(self._check_secret_literals())
        issues.extend(self._check_gitignore_rules())
        issues.extend(self._check_generated_files())
        issues.extend(self._check_cache_dirs())
        issues.extend(self._check_git_status())
        issues.extend(self._check_readme_legacy_text())
        issues.extend(self._check_cloud_legacy_imports())
        issues.extend(self._check_pyproject_entrypoint())
        issues.extend(self._check_required_artifacts())
        ok = not any(issue.severity == "error" for issue in issues)
        return RepoHygieneReport(ok=ok, issues=issues)

    def _check_env_files(self) -> list[RepoHygieneIssue]:
        env_path = self.root / ".env"
        if not env_path.exists():
            return []
        return [
            RepoHygieneIssue(
                code="env_file_present",
                severity="error",
                message="仓库根目录存在 .env，发布前不能提交或打包本地密钥文件。",
                path=".env",
                suggestion="删除 .env 或仅保留 .env.example。",
            )
        ]

    def _check_secret_literals(self) -> list[RepoHygieneIssue]:
        issues: list[RepoHygieneIssue] = []
        for path in self._iter_text_files():
            relative = _rel(path, self.root)
            if relative == ".env.example":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for match in SECRET_ASSIGNMENT_RE.finditer(text):
                if not _is_placeholder_secret(match.group(2)):
                    issues.append(
                        RepoHygieneIssue(
                            "possible_secret_assignment",
                            "error",
                            f"发现疑似真实 {match.group(1)} 赋值。",
                            relative,
                            "将值清空或替换为占位符。",
                        )
                    )
                    break
            else:
                if any(not _is_placeholder_secret(match.group(0)) for match in SECRET_VALUE_RE.finditer(text)):
                    issues.append(
                        RepoHygieneIssue(
                            "possible_secret_literal",
                            "error",
                            "发现疑似真实 API key 字面量。",
                            relative,
                            "移除真实 key，改用环境变量或 GitHub Secrets。",
                        )
                    )
        return issues

    def _check_gitignore_rules(self) -> list[RepoHygieneIssue]:
        gitignore = self.root / ".gitignore"
        if not gitignore.exists():
            return [
                RepoHygieneIssue(
                    "gitignore_missing",
                    "error",
                    ".gitignore 不存在，无法保护本地输出和密钥文件。",
                    ".gitignore",
                    "新增 .gitignore 并覆盖 env、trace、audit、eval、pkl 和缓存目录。",
                )
            ]
        rules = _normalize_rules(gitignore.read_text(encoding="utf-8"))
        return [
            RepoHygieneIssue(
                "gitignore_rule_missing",
                "error",
                f".gitignore 缺少必要规则：{rule}",
                ".gitignore",
                "补齐发布卫生要求中的忽略规则。",
            )
            for rule in REQUIRED_GITIGNORE_RULES
            if rule not in rules
        ]

    def _check_generated_files(self) -> list[RepoHygieneIssue]:
        tracked = set(self._git_lines(["ls-files"]))
        issues: list[RepoHygieneIssue] = []
        for path in self._iter_existing_generated_files():
            relative = _rel(path, self.root)
            severity = "warning"
            message = "发现本地生成文件，发布前请确认不会误提交。"
            if relative in tracked:
                message = "生成文件当前已被 Git 跟踪，发布前建议迁移到可复现构建产物。"
            issues.append(
                RepoHygieneIssue(
                    "generated_file_present",
                    severity,
                    message,
                    relative,
                    "确认 .gitignore 覆盖该文件，并在后续版本中逐步移除已跟踪生成产物。",
                )
            )
        return issues

    def _check_cache_dirs(self) -> list[RepoHygieneIssue]:
        issues: list[RepoHygieneIssue] = []
        for name in ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".pyright"):
            for path in self.root.rglob(name):
                if _is_ignored_path(path, self.root):
                    continue
                issues.append(
                    RepoHygieneIssue(
                        "cache_dir_present",
                        "warning",
                        "发现本地缓存目录，发布前可清理。",
                        _rel(path, self.root),
                        "删除缓存目录，确保发布包只包含可复现内容。",
                    )
                )
        return issues

    def _check_git_status(self) -> list[RepoHygieneIssue]:
        lines = self._git_lines(["status", "--porcelain"])
        if not lines:
            return []
        issues: list[RepoHygieneIssue] = []
        for line in lines:
            status = line[:2]
            path = line[3:].strip()
            severity = "warning"
            code = "worktree_dirty"
            message = "工作区存在未提交变更。"
            if status == "??":
                code = "untracked_file"
                message = "工作区存在未跟踪文件。"
                if Path(path).suffix in {".py", ".md", ".toml", ".yml", ".yaml"}:
                    message = "工作区存在未跟踪的重要源码或配置文件。"
            issues.append(RepoHygieneIssue(code, severity, message, path, "发布前请审阅、提交或忽略该文件。"))
        return issues

    def _check_readme_legacy_text(self) -> list[RepoHygieneIssue]:
        readme = self.root / "README.md"
        if not readme.exists():
            return []
        text = readme.read_text(encoding="utf-8")
        legacy_mentions = ["kairorag.demos.run_single_query", "kairorag.indexing.build_index", "vector_store.pkl"]
        if any(item in text for item in legacy_mentions) and not any(word in text for word in ("废弃", "不再推荐", "Legacy")):
            return [
                RepoHygieneIssue(
                    "readme_legacy_recommendation",
                    "warning",
                    "README 中可能仍在推荐 legacy CLI 或旧 pickle 索引路径。",
                    "README.md",
                    "确认 README 只推荐 kairo 统一 CLI 和 cloud-native 主路径。",
                )
            ]
        return []

    def _check_cloud_legacy_imports(self) -> list[RepoHygieneIssue]:
        patterns = [
            "from kairorag.agent.react_agent",
            "import kairorag.agent.react_agent",
            "from kairorag.agent.planner",
            "from kairorag.indexing.vector_store",
            "from kairorag.indexing.keyword_index",
            "from kairorag.indexing.embedding",
        ]
        hits: list[RepoHygieneIssue] = []
        for path in [self.root / "kairorag" / "cloud", self.root / "kairorag" / "cli.py"]:
            for file_path in _iter_files(path):
                if file_path.suffix != ".py":
                    continue
                try:
                    text = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for line in text.splitlines():
                    stripped = line.strip()
                    if not (stripped.startswith("from ") or stripped.startswith("import ")):
                        continue
                    for pattern in patterns:
                        if pattern in stripped:
                            hits.append(
                                RepoHygieneIssue(
                                    "cloud_legacy_import",
                                    "error",
                                    "cloud 主路径重新引用了 legacy 模块。",
                                    _rel(file_path, self.root),
                                    f"移除 import：{pattern}",
                                )
                            )
        return hits

    def _check_pyproject_entrypoint(self) -> list[RepoHygieneIssue]:
        pyproject = self.root / "pyproject.toml"
        if not pyproject.exists():
            return [
                RepoHygieneIssue("pyproject_missing", "error", "缺少 pyproject.toml。", "pyproject.toml", None)
            ]
        text = pyproject.read_text(encoding="utf-8")
        if 'kairo = "kairorag.cli:main"' in text:
            return []
        return [
            RepoHygieneIssue(
                "kairo_entrypoint_missing",
                "error",
                "pyproject.toml 缺少 kairo entry point。",
                "pyproject.toml",
                '在 [project.scripts] 中配置 kairo = "kairorag.cli:main"。',
            )
        ]

    def _check_required_artifacts(self) -> list[RepoHygieneIssue]:
        required = [
            ".github/workflows/ci.yml",
            "Dockerfile",
            "docker-compose.yml",
            ".env.example",
            "Makefile",
        ]
        return [
            RepoHygieneIssue(
                "required_artifact_missing",
                "error",
                "发布所需文件不存在。",
                path,
                "补齐 CI、Docker、Compose、环境示例或 Makefile。",
            )
            for path in required
            if not (self.root / path).exists()
        ]

    def _iter_text_files(self) -> Iterable[Path]:
        skip_dirs = {".git", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".pyright", ".venv", "venv"}
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in skip_dirs for part in path.relative_to(self.root).parts):
                continue
            if path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", "Makefile"} or path.name.startswith(".env"):
                yield path

    def _iter_existing_generated_files(self) -> Iterable[Path]:
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            relative = _rel(path, self.root)
            if any(fnmatch.fnmatch(relative, pattern) for pattern in GENERATED_PATTERNS):
                yield path

    def _git_lines(self, args: list[str]) -> list[str]:
        command = ["git", "-c", f"safe.directory={self.root.as_posix()}", *args]
        try:
            completed = subprocess.run(command, cwd=self.root, text=True, capture_output=True, check=False)
        except OSError:
            return []
        if completed.returncode != 0:
            return []
        return [line for line in completed.stdout.splitlines() if line.strip()]


def _normalize_rules(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")}


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from (item for item in path.rglob("*") if item.is_file())


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _is_ignored_path(path: Path, root: Path) -> bool:
    parts = set(path.resolve().relative_to(root).parts)
    return ".git" in parts


def _is_placeholder_secret(value: str) -> bool:
    clean = value.strip().strip('"').strip("'").lower()
    if clean in {"", "...", "placeholder", "secret", "your-key", "changeme", "<your-key>"}:
        return True
    return any(token in clean for token in ("placeholder", "secret", "redacted", "example", "fake"))
