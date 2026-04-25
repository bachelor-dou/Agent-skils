"""Pure helper functions used by route parsing in agent."""

import json


def extract_json_object(content: str) -> dict | None:
    """Extract the first JSON object from a text payload."""
    text = (content or "").strip()
    if not text:
        return None
    for candidate in (text, text[text.find("{"):] if "{" in text else ""):
        if not candidate:
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def looks_like_structured_confirmation_text(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False

    lowered = stripped.lower()
    structured_keys = (
        '"turn_kind"',
        '"intent_family"',
        '"intent_label_zh"',
        '"target_repo"',
        '"specified_params"',
        '"ambiguous_fields"',
        '"confirmation_text_zh"',
    )
    if any(key in lowered for key in structured_keys):
        return True

    return stripped.startswith("```") or (stripped.startswith("{") and stripped.endswith("}"))


def sanitize_confirmation_fallback(content: str, default_message: str) -> str:
    """Return safe fallback text when route output is empty/structured."""
    stripped = (content or "").strip()
    if not stripped:
        return default_message
    if looks_like_structured_confirmation_text(stripped):
        return default_message
    return stripped


def normalize_intent_family(
    raw_intent: object,
    *,
    intent_aliases: dict[str, str],
    intent_labels: dict[str, str],
) -> str:
    """Normalize intent name with aliases and whitelist enforcement."""
    if not isinstance(raw_intent, str) or not raw_intent.strip():
        return "unknown"
    normalized = raw_intent.strip().lower()
    normalized = intent_aliases.get(normalized, normalized)
    return normalized if normalized in intent_labels else "unknown"


def normalize_tool_names(raw_tools: object, *, allowed_tool_names: set[str]) -> list[str]:
    """Filter invalid tool names and preserve unique order."""
    if not isinstance(raw_tools, list):
        return []
    cleaned: list[str] = []
    for item in raw_tools:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name or name not in allowed_tool_names:
            continue
        cleaned.append(name)

    deduped = []
    seen = set()
    for name in cleaned:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def ordered_tool_names(tool_names: set[str], *, all_tool_names: list[str]) -> list[str]:
    """Return tool names in canonical order for stable prompting/logging."""
    return [name for name in all_tool_names if name in tool_names]


def normalize_turn_kind(raw_turn_kind: object, *, turn_kinds: set[str]) -> str:
    """Normalize turn kind and clamp to allowed enum values."""
    if not isinstance(raw_turn_kind, str) or not raw_turn_kind.strip():
        return "unknown"
    normalized = raw_turn_kind.strip().lower()
    return normalized if normalized in turn_kinds else "unknown"


def normalize_specified_params(
    raw_params: object,
    *,
    allowed_param_names: set[str],
) -> tuple[dict[str, object], list[str], list[str]]:
    """Normalize route extracted params into canonical keys only.

    Returns:
      - normalized params (canonical keys)
      - dropped keys that cannot be mapped deterministically
      - normalization notes for debug logging
    """
    if not isinstance(raw_params, dict):
        return {}, [], []

    canonical_by_lower = {name.lower(): name for name in allowed_param_names}
    canonical_by_compact = {
        lowered.replace("_", ""): canonical
        for lowered, canonical in canonical_by_lower.items()
    }

    normalized: dict[str, object] = {}
    dropped: list[str] = []
    notes: list[str] = []

    for raw_key, value in raw_params.items():
        if not isinstance(raw_key, str):
            dropped.append("<non_string_key>")
            notes.append("drop:non_string_key")
            continue

        key = raw_key.strip()
        if not key:
            continue

        canonical: str | None = None
        reason: str | None = None

        if key in allowed_param_names:
            canonical = key
        else:
            normalized_key = key.lower().replace("-", "_").replace(" ", "_")
            if normalized_key in canonical_by_lower:
                canonical = canonical_by_lower[normalized_key]
                reason = "case"
            else:
                compact = normalized_key.replace("_", "")
                if compact in canonical_by_compact:
                    canonical = canonical_by_compact[compact]
                    reason = "shape"

        if canonical is None:
            dropped.append(key)
            notes.append(f"drop:{key}")
            continue

        if canonical in normalized and normalized[canonical] != value:
            dropped.append(key)
            notes.append(f"conflict:{key}->{canonical}")
            continue

        normalized[canonical] = value
        if canonical != key:
            notes.append(f"map:{key}->{canonical}({reason or 'normalized'})")

    return normalized, dropped, notes
