"""Deny-by-default adapter for approved local derived research tables.

The adapter never discovers files. Every input must be identified by an exact
path, hash, and schema in an explicit manifest. The default repository template
contains an empty allowlist.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


class DerivedDataAccessError(RuntimeError):
    """Raised when an input is not explicitly approved or fails validation."""


FORBIDDEN_COLUMN_PATTERNS = (
    re.compile(r"(^|_)(full_)?name($|_)", re.I),
    re.compile(r"phone|mobile|email|contact", re.I),
    re.compile(r"resume|self_description|free_text", re.I),
    re.compile(r"company|url|job_description|job_text", re.I),
    re.compile(r"(^|_)(prompt|raw_prompt|request_prompt|raw_response|request_body)($|_)", re.I),
    re.compile(r"^(original_job_id|stage2_job_id)$", re.I),
)

ALLOWED_TOP_LEVEL_KEYS = {
    "format_version",
    "real_data_access_enabled",
    "approved_inputs",
    "notes",
}

ALLOWED_ENTRY_KEYS = {
    "input_id",
    "path",
    "sha256",
    "file_size",
    "format",
    "schema",
    "provenance_asset_id",
    "approval_id",
}

ALLOWED_SCHEMA_KEYS = {
    "columns",
    "dtypes",
    "unique_key",
    "required_nonmissing",
    "categorical_domains",
    "pairing",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _has_symlink_component(path: Path) -> bool:
    """Return True if the exact path or any existing parent is a symlink."""

    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        if current.exists() and current.is_symlink():
            return True
    return False


def _safe_exact_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if ".." in candidate.parts:
        raise DerivedDataAccessError("Path traversal is prohibited")
    candidate = Path(os.path.abspath(candidate))
    if _has_symlink_component(candidate):
        raise DerivedDataAccessError("Symlinked inputs are prohibited")
    if not candidate.is_file():
        raise DerivedDataAccessError("Approved input is not a regular file")
    return candidate


def _validate_schema_definition(schema: dict[str, Any]) -> None:
    unexpected = set(schema) - ALLOWED_SCHEMA_KEYS
    if unexpected:
        raise DerivedDataAccessError(f"Unexpected schema settings: {sorted(unexpected)}")
    columns = schema.get("columns")
    if not isinstance(columns, list) or not columns or len(columns) != len(set(columns)):
        raise DerivedDataAccessError("Schema columns must be a nonempty unique list")
    for column in columns:
        if any(pattern.search(str(column)) for pattern in FORBIDDEN_COLUMN_PATTERNS):
            raise DerivedDataAccessError(f"Restricted column is not allowed: {column}")
    dtypes = schema.get("dtypes", {})
    if set(dtypes) != set(columns):
        raise DerivedDataAccessError("Every approved column must have one declared dtype")
    if set(dtypes.values()) - {"string", "float", "integer", "boolean"}:
        raise DerivedDataAccessError("Unsupported dtype in derived-data schema")
    for field in ("unique_key", "required_nonmissing"):
        if set(schema.get(field, [])) - set(columns):
            raise DerivedDataAccessError(f"Unknown {field} column")
    if set(schema.get("categorical_domains", {})) - set(columns):
        raise DerivedDataAccessError("Categorical domain references an unknown column")
    pairing = schema.get("pairing")
    if pairing is not None:
        expected = {"group_by", "member_column", "expected_members"}
        if set(pairing) != expected:
            raise DerivedDataAccessError("Pairing definition must use the fixed three fields")
        if set(pairing["group_by"]) - set(columns) or pairing["member_column"] not in columns:
            raise DerivedDataAccessError("Pairing definition references an unknown column")


class DerivedDataAdapter:
    """Read exact, hash-pinned, schema-approved local CSV inputs."""

    def __init__(self, manifest_path: Path, *, repository_root: Path):
        self.manifest_path = _safe_exact_path(str(manifest_path))
        self.repository_root = Path(repository_root).resolve(strict=True)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        unexpected = set(payload) - ALLOWED_TOP_LEVEL_KEYS
        if unexpected:
            raise DerivedDataAccessError(f"Unexpected manifest fields: {sorted(unexpected)}")
        if payload.get("format_version") != 1:
            raise DerivedDataAccessError("Unsupported derived-input manifest version")
        if not isinstance(payload.get("approved_inputs"), list):
            raise DerivedDataAccessError("approved_inputs must be a list")
        self.enabled = payload.get("real_data_access_enabled") is True
        self.entries: dict[str, dict[str, Any]] = {}
        for entry in payload["approved_inputs"]:
            if set(entry) - ALLOWED_ENTRY_KEYS:
                raise DerivedDataAccessError("Unexpected approved-input manifest field")
            input_id = str(entry.get("input_id", "")).strip()
            if not input_id or input_id in self.entries:
                raise DerivedDataAccessError("Input IDs must be nonempty and unique")
            if entry.get("format") != "csv":
                raise DerivedDataAccessError("Only explicitly schema-checked CSV is supported")
            if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
                raise DerivedDataAccessError("Approved input requires a SHA-256")
            _validate_schema_definition(entry.get("schema", {}))
            self.entries[input_id] = entry

    def approved_input_ids(self) -> tuple[str, ...]:
        return tuple(self.entries)

    def load_csv(self, input_id: str) -> pd.DataFrame:
        """Load an approved CSV read-only, then validate hash, schema, and pairing."""

        if not self.enabled:
            raise DerivedDataAccessError("Real-data access is disabled by the manifest")
        if input_id not in self.entries:
            raise DerivedDataAccessError("Input path is not on the exact allowlist")
        entry = self.entries[input_id]
        path = _safe_exact_path(str(entry["path"]))
        before_hash = sha256_file(path)
        if before_hash != entry["sha256"]:
            raise DerivedDataAccessError("Approved input SHA-256 mismatch")
        if entry.get("file_size") is not None and path.stat().st_size != int(entry["file_size"]):
            raise DerivedDataAccessError("Approved input size mismatch")
        string_columns = {
            column: "string"
            for column, declared in entry["schema"]["dtypes"].items()
            if declared == "string"
        }
        with path.open("rb") as handle:
            frame = pd.read_csv(handle, dtype=string_columns)
        self._validate_frame(frame, entry["schema"])
        if sha256_file(path) != before_hash:
            raise DerivedDataAccessError("Source file changed during read-only loading")
        return frame

    @staticmethod
    def _validate_frame(frame: pd.DataFrame, schema: dict[str, Any]) -> None:
        expected_columns = list(schema["columns"])
        if list(frame.columns) != expected_columns:
            raise DerivedDataAccessError("Input columns or order differ from the approved schema")
        for column, dtype in schema["dtypes"].items():
            try:
                if dtype == "string":
                    frame[column] = frame[column].astype("string")
                elif dtype == "float":
                    frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
                elif dtype == "integer":
                    frame[column] = pd.to_numeric(frame[column], errors="raise").astype("Int64")
                elif dtype == "boolean":
                    values = frame[column].dropna()
                    if not values.isin([True, False, 0, 1]).all():
                        raise ValueError("invalid boolean")
                    frame[column] = frame[column].astype("boolean")
            except (TypeError, ValueError) as error:
                raise DerivedDataAccessError(f"Invalid dtype for {column}: {dtype}") from error
        nonmissing = schema.get("required_nonmissing", [])
        if nonmissing and frame[nonmissing].isna().any().any():
            raise DerivedDataAccessError("Required key or identifier is missing")
        key = schema.get("unique_key", [])
        if key and frame.duplicated(key).any():
            raise DerivedDataAccessError("Duplicate approved-input key")
        for column, permitted in schema.get("categorical_domains", {}).items():
            observed = set(frame[column].dropna().astype(str))
            if observed - set(map(str, permitted)):
                raise DerivedDataAccessError(f"Unexpected categorical value in {column}")
        pairing = schema.get("pairing")
        if pairing:
            expected_members = set(map(str, pairing["expected_members"]))
            grouped = frame.groupby(pairing["group_by"], dropna=False, sort=False)
            for _, group in grouped:
                observed = group[pairing["member_column"]].astype(str)
                if set(observed) != expected_members or observed.duplicated().any():
                    raise DerivedDataAccessError("Source-category pairing is incomplete or duplicated")


def validate_replay_output_root(
    output_root: Path,
    *,
    repository_root: Path,
    cloud_roots: tuple[Path, ...] = (),
) -> Path:
    """Verify that future replay output is isolated from repository/cloud roots."""

    raw = Path(output_root).expanduser()
    if ".." in raw.parts or _has_symlink_component(raw):
        raise DerivedDataAccessError("Replay output may not use traversal or symlinks")
    resolved = Path(os.path.abspath(raw))
    repository = repository_root.resolve(strict=True)
    if resolved == repository or repository in resolved.parents:
        raise DerivedDataAccessError("Replay output may not be inside the repository")
    for cloud_root in cloud_roots:
        cloud = Path(os.path.abspath(cloud_root.expanduser()))
        if resolved == cloud or cloud in resolved.parents:
            raise DerivedDataAccessError("Replay output may not use a cloud-synchronized location")
    return resolved
