"""Cloud-native 生产化配置检查。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from kairorag.config import KairoCloudSettings, PROJECT_ROOT
from kairorag.config.settings import CloudRuntimeConfigurationError, validate_cloud_runtime
from kairorag.providers import build_vector_store_provider


SECRET_ASSIGNMENT_RE = re.compile(
    r"(?P<key>OPENAI_API_KEY|COHERE_API_KEY|TAVILY_API_KEY|SERPAPI_API_KEY|BING_API_KEY|VOYAGE_API_KEY|JINA_API_KEY)\s*=\s*(?P<value>[^\s#]+)"
)
SECRET_VALUE_RE = re.compile(r"sk-[A-Za-z0-9_\-]{12,}")
SECRET_PLACEHOLDERS = {"", "...", "placeholder", "secret", "your-key", "changeme", "<your-key>"}
SUPPORTED_WEB_SEARCH_PROVIDERS = {"tavily", "serpapi", "bing"}
SUPPORTED_RERANKERS = {"base_score", "cohere", "jina", "voyage", "openai_listwise", "cross_encoder"}
LEGACY_IMPORT_PATTERNS = {
    "from kairorag.agent.react_agent",
    "import kairorag.agent.react_agent",
    "from kairorag.agent.planner",
    "from kairorag.maintenance.job_verifier",
    "from kairorag.maintenance.web_search",
    "from kairorag.agent.answer_generator",
    "from kairorag.indexing.vector_store",
    "from kairorag.indexing.keyword_index",
    "from kairorag.indexing.embedding",
    "from kairorag.indexing.build_index",
}


@dataclass(frozen=True)
class DoctorCheckResult:
    """单个 doctor 检查项的结果。"""

    name: str
    ok: bool
    severity: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DoctorReport:
    """doctor 总报告。"""

    ok: bool
    checks: list[DoctorCheckResult]


class CloudDoctor:
    """执行 cloud-native runtime 的本地配置与轻量健康检查。"""

    def __init__(self, settings: KairoCloudSettings) -> None:
        self.settings = settings

    def run(self, *, live: bool = False, checks: list[str] | None = None) -> DoctorReport:
        """运行指定检查项；默认不调用外部付费服务。"""

        selected = _normalize_checks(checks)
        results: list[DoctorCheckResult] = []
        for name in selected:
            if name == "config":
                results.append(self._check_config())
            elif name == "paths":
                results.append(self._check_paths())
            elif name == "qdrant":
                results.append(self._check_qdrant(live=live))
            elif name == "openai":
                results.append(self._check_openai(live=live))
            elif name == "websearch":
                results.append(self._check_websearch(live=live))
            elif name == "reranker":
                results.append(self._check_reranker(live=live))
            elif name == "legacy":
                results.append(self._check_legacy_imports())
        return DoctorReport(ok=all(item.ok for item in results), checks=results)

    def _check_config(self) -> DoctorCheckResult:
        problems: list[str] = []
        try:
            validate_cloud_runtime(self.settings)
        except CloudRuntimeConfigurationError as exc:
            problems.append(str(exc))
        secret_hits = _scan_repo_for_secret_literals(PROJECT_ROOT)
        if secret_hits:
            problems.append("发现疑似硬编码 secret。")
        return DoctorCheckResult(
            name="config",
            ok=not problems,
            severity="error" if problems else "info",
            message="配置检查通过。" if not problems else "配置检查失败：" + "；".join(problems),
            detail={
                "providers": _redact(
                    {
                        "llm_provider": self.settings.llm_provider,
                        "embedding_provider": self.settings.embedding_provider,
                        "vector_store_provider": self.settings.vector_store_provider,
                        "web_search_provider": self.settings.web_search_provider,
                        "reranker_provider": self.settings.reranker_provider,
                    }
                ),
                "secret_hits": secret_hits,
            },
        )

    def _check_paths(self) -> DoctorCheckResult:
        manifest = Path(self.settings.cloud_index_manifest_path)
        bm25 = Path(self.settings.cloud_bm25_index_path)
        audit_dir_ok = _ensure_writable_parent(Path(self.settings.freshness_audit_log_path))
        trace_dir_ok = _ensure_writable_parent(Path(self.settings.trace_log_path))
        missing = []
        if not manifest.exists():
            missing.append(f"manifest 不存在：{manifest}")
        if not bm25.exists():
            missing.append(f"BM25 index 不存在：{bm25}")
        if not audit_dir_ok:
            missing.append(f"audit log 目录不可写：{Path(self.settings.freshness_audit_log_path).parent}")
        if not trace_dir_ok:
            missing.append(f"trace log 目录不可写：{Path(self.settings.trace_log_path).parent}")
        return DoctorCheckResult(
            name="paths",
            ok=not missing,
            severity="warning" if missing else "info",
            message="路径检查通过。" if not missing else "路径检查发现问题：" + "；".join(missing),
            detail={
                "manifest_path": str(manifest),
                "manifest_exists": manifest.exists(),
                "bm25_path": str(bm25),
                "bm25_exists": bm25.exists(),
                "audit_dir_writable": audit_dir_ok,
                "trace_dir_writable": trace_dir_ok,
            },
        )

    def _check_qdrant(self, *, live: bool) -> DoctorCheckResult:
        configured = bool(self.settings.qdrant_url and self.settings.qdrant_collection)
        details = {
            "qdrant_url": _mask_url(self.settings.qdrant_url),
            "qdrant_collection": self.settings.qdrant_collection,
            "live": live,
        }
        if not configured:
            return DoctorCheckResult(
                name="qdrant",
                ok=False,
                severity="error",
                message="Qdrant 配置缺失：需要 QDRANT_URL 和 QDRANT_COLLECTION。",
                detail=details,
            )
        if not live:
            return DoctorCheckResult(
                name="qdrant",
                ok=True,
                severity="info",
                message="Qdrant 离线配置检查通过；未执行 live healthcheck。",
                detail=details,
            )
        try:
            vector_store = build_vector_store_provider(self.settings)
            ok = bool(vector_store.healthcheck())
        except Exception as exc:
            details["error"] = str(exc)
            ok = False
        return DoctorCheckResult(
            name="qdrant",
            ok=ok,
            severity="error" if not ok else "info",
            message="Qdrant live healthcheck 通过。" if ok else "Qdrant live healthcheck 失败。",
            detail=_redact(details),
        )

    def _check_openai(self, *, live: bool) -> DoctorCheckResult:
        has_key = _has_secret(self.settings.openai_api_key)
        has_models = bool(self.settings.openai_chat_model and self.settings.openai_embedding_model)
        ok = has_key and has_models
        message = "OpenAI 配置检查通过。" if ok else "OpenAI 配置缺失：需要 OPENAI_API_KEY 和模型名称。"
        detail = {
            "has_openai_api_key": has_key,
            "openai_chat_model": self.settings.openai_chat_model,
            "openai_embedding_model": self.settings.openai_embedding_model,
            "live": live,
            "live_note": "为避免付费调用，doctor 不会发起 chat/completions 请求。",
        }
        return DoctorCheckResult("openai", ok, "error" if not ok else "info", message, detail)

    def _check_websearch(self, *, live: bool) -> DoctorCheckResult:
        provider = self.settings.web_search_provider
        has_provider = provider in SUPPORTED_WEB_SEARCH_PROVIDERS
        has_key = _websearch_has_key(self.settings)
        ok = has_provider and has_key
        message = "Web Search 配置检查通过。" if ok else "Web Search 配置缺失或 provider 不受支持。"
        detail = {
            "provider": provider,
            "has_key": has_key,
            "live": live,
            "live_note": "默认不发起外部搜索请求；需要真实探测时请单独运行 live 测试。",
        }
        return DoctorCheckResult("websearch", ok, "warning" if not ok else "info", message, detail)

    def _check_reranker(self, *, live: bool) -> DoctorCheckResult:
        provider = self.settings.reranker_provider
        has_provider = provider in SUPPORTED_RERANKERS
        has_key = _reranker_has_key(self.settings)
        ok = has_provider and has_key
        message = "Reranker 配置检查通过。" if ok else "Reranker 配置缺失或 provider 不受支持。"
        detail = {
            "provider": provider,
            "has_required_key": has_key,
            "live": live,
            "live_note": "默认不调用外部 rerank 服务。",
        }
        return DoctorCheckResult("reranker", ok, "warning" if not ok else "info", message, detail)

    def _check_legacy_imports(self) -> DoctorCheckResult:
        hits = _scan_cloud_for_legacy_imports(PROJECT_ROOT)
        return DoctorCheckResult(
            name="legacy",
            ok=not hits,
            severity="error" if hits else "info",
            message="cloud 主路径未发现 legacy import。" if not hits else "cloud 主路径仍存在 legacy import。",
            detail={"hits": hits},
        )


def report_to_dict(report: DoctorReport) -> dict[str, Any]:
    """把 doctor 报告转换成可 JSON 序列化的 dict。"""

    return _redact(asdict(report))


def report_to_json(report: DoctorReport) -> str:
    """输出脱敏 JSON 报告。"""

    return json.dumps(report_to_dict(report), ensure_ascii=False, indent=2)


def format_doctor_report(report: DoctorReport) -> str:
    """输出中文可读报告。"""

    lines = ["KairoRAG Doctor 报告", f"整体状态：{'通过' if report.ok else '失败'}"]
    for item in report.checks:
        status = "通过" if item.ok else "失败"
        lines.append(f"- [{status}] {item.name}（{item.severity}）：{item.message}")
    return "\n".join(lines)


def settings_summary(settings: KairoCloudSettings, *, show_paths: bool = False) -> dict[str, Any]:
    """生成脱敏配置摘要。"""

    payload: dict[str, Any] = {
        "kairo_env": settings.kairo_env,
        "llm_provider": settings.llm_provider,
        "openai_chat_model": settings.openai_chat_model,
        "embedding_provider": settings.embedding_provider,
        "openai_embedding_model": settings.openai_embedding_model,
        "vector_store_provider": settings.vector_store_provider,
        "qdrant_url": _mask_url(settings.qdrant_url),
        "qdrant_collection": settings.qdrant_collection,
        "keyword_search_provider": settings.keyword_search_provider,
        "web_search_provider": settings.web_search_provider,
        "reranker_provider": settings.reranker_provider,
        "rerank_top_k": settings.rerank_top_k,
        "qdrant_metadata_write_enabled": settings.qdrant_metadata_write_enabled,
        "qdrant_metadata_write_dry_run": settings.qdrant_metadata_write_dry_run,
        "has_openai_api_key": _has_secret(settings.openai_api_key),
        "has_qdrant_api_key": _has_secret(settings.qdrant_api_key),
        "has_tavily_api_key": _has_secret(settings.tavily_api_key),
        "has_cohere_api_key": _has_secret(settings.cohere_api_key),
        "has_jina_api_key": _has_secret(settings.jina_api_key),
        "has_voyage_api_key": _has_secret(settings.voyage_api_key),
    }
    if show_paths:
        payload.update(
            {
                "cloud_index_manifest_path": settings.cloud_index_manifest_path,
                "cloud_bm25_index_path": settings.cloud_bm25_index_path,
                "freshness_audit_log_path": settings.freshness_audit_log_path,
                "trace_log_path": settings.trace_log_path,
            }
        )
    return _redact(payload)


def _normalize_checks(checks: list[str] | None) -> list[str]:
    all_checks = ["config", "paths", "qdrant", "openai", "websearch", "reranker", "legacy"]
    if not checks or "all" in checks:
        return all_checks
    aliases = {"web_search": "websearch"}
    normalized = [aliases.get(item, item) for item in checks]
    return [item for item in all_checks if item in normalized]


def _has_secret(value: SecretStr | None) -> bool:
    return bool(value and value.get_secret_value().strip())


def _websearch_has_key(settings: KairoCloudSettings) -> bool:
    if settings.web_search_provider == "tavily":
        return _has_secret(settings.tavily_api_key)
    if settings.web_search_provider == "serpapi":
        return _has_secret(settings.serpapi_api_key)
    if settings.web_search_provider == "bing":
        return _has_secret(settings.bing_api_key)
    return False


def _reranker_has_key(settings: KairoCloudSettings) -> bool:
    if settings.reranker_provider == "base_score":
        return True
    if settings.reranker_provider == "cohere":
        return _has_secret(settings.cohere_api_key)
    if settings.reranker_provider == "jina":
        return _has_secret(settings.jina_api_key)
    if settings.reranker_provider == "voyage":
        return _has_secret(settings.voyage_api_key)
    if settings.reranker_provider == "openai_listwise":
        return _has_secret(settings.openai_api_key)
    if settings.reranker_provider == "cross_encoder":
        return bool(settings.cross_encoder_model)
    return False


def _ensure_writable_parent(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / ".kairo_doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _scan_repo_for_secret_literals(root: Path) -> list[str]:
    hits: list[str] = []
    search_roots = [root / "kairorag", root / "tests", root / "README.md", root / "pyproject.toml"]
    for path in _iter_text_files(search_roots):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(not _is_placeholder_secret(match.group(0)) for match in SECRET_VALUE_RE.finditer(text)):
            hits.append(str(path.relative_to(root)))
            continue
        for match in SECRET_ASSIGNMENT_RE.finditer(text):
            value = match.group("value").strip().strip('"').strip("'")
            if not _is_placeholder_secret(value):
                hits.append(str(path.relative_to(root)))
                break
    return sorted(set(hits))


def _scan_cloud_for_legacy_imports(root: Path) -> list[str]:
    paths = [root / "kairorag" / "cloud"]
    cli_path = root / "kairorag" / "cli.py"
    if cli_path.exists():
        paths.append(cli_path)
    hits: list[str] = []
    for path in _iter_text_files(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("from ") or stripped.startswith("import ")):
                continue
            for pattern in LEGACY_IMPORT_PATTERNS:
                if pattern in stripped:
                    hits.append(f"{path.relative_to(root)}: {pattern}")
    return sorted(hits)


def _iter_text_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix in {".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".example"}
            )
    return files


def _redact(value: Any) -> Any:
    if is_dataclass(value):
        return _redact(asdict(value))
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text.startswith("has_"):
                redacted[key] = _redact(item)
            elif "api_key" in key_text or "secret" in key_text:
                redacted[key] = "[已隐藏]" if item else item
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item) for item in value)
    if isinstance(value, SecretStr):
        return "[已隐藏]" if value.get_secret_value().strip() else ""
    if isinstance(value, str):
        return SECRET_VALUE_RE.sub("[已隐藏]", value)
    return value


def _is_placeholder_secret(value: str) -> bool:
    clean = value.strip().strip('"').strip("'").lower()
    if clean in SECRET_PLACEHOLDERS or clean.startswith("<"):
        return True
    return "placeholder" in clean or "secret" in clean or "redacted" in clean or "example" in clean


def _mask_url(url: str | None) -> str | None:
    if not url:
        return url
    if "@" not in url:
        return url
    scheme, rest = url.split("://", 1) if "://" in url else ("", url)
    suffix = rest.split("@", 1)[1]
    return f"{scheme + '://' if scheme else ''}[已隐藏]@{suffix}"
