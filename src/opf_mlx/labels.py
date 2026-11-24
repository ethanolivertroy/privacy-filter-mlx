"""The label space: which model output index means which boundary of which category.

The released checkpoint stores no label names. Like the reference, this module
infers them from `num_labels` (33 is the "v2" taxonomy) and also accepts the
custom names a fine-tuned checkpoint may carry in its config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

BACKGROUND = "O"
BOUNDARIES: tuple[str, ...] = ("B", "I", "E", "S")

SPAN_NAMES_BY_VERSION: dict[str, tuple[str, ...]] = {
    "v2": (
        BACKGROUND,
        "account_number",
        "private_address",
        "private_date",
        "private_email",
        "private_person",
        "private_phone",
        "private_url",
        "secret",
    ),
    "v4": (
        BACKGROUND,
        "private_person",
        "other_person",
        "personal_url",
        "other_url",
        "personal_location",
        "other_location",
        "personal_email",
        "other_email",
        "personal_phone",
        "other_phone",
        "personal_date",
        "other_date",
        "personal_id",
        "secret",
    ),
    "v7": (
        BACKGROUND,
        "personal_name",
        "personal_handle",
        "other_person",
        "personal_email",
        "other_email",
        "personal_phone",
        "other_phone",
        "personal_location",
        "other_location",
        "personal_url",
        "other_url",
        "personal_org",
        "personal_gov_id",
        "personal_fin_id",
        "personal_health_id",
        "personal_device_id",
        "personal_vehicle_id",
        "personal_property_id",
        "personal_edu_id",
        "personal_emp_id",
        "personal_membership_id",
        "personal_registry_id",
        "personal_date",
        "secret",
        "secret_url",
    ),
}
DEFAULT_VERSION = "v2"


class LabelError(ValueError):
    """The checkpoint config describes an inconsistent or unknown label space."""


@dataclass(frozen=True)
class LabelSpace:
    """Token classes (what the model outputs) and span classes (what users see)."""

    class_names: tuple[str, ...]
    span_names: tuple[str, ...]
    span_of: tuple[int, ...]
    tag_of: tuple[str | None, ...]
    background: int
    category_version: str


def expand_with_boundaries(span_names: Sequence[str]) -> tuple[str, ...]:
    """`O` followed by `B-`, `I-`, `E-`, `S-` of each category, in order."""
    expanded = [BACKGROUND]
    for name in span_names:
        if name != BACKGROUND:
            expanded.extend(f"{boundary}-{name}" for boundary in BOUNDARIES)
    return tuple(expanded)


CLASS_NAMES_BY_VERSION = {version: expand_with_boundaries(names) for version, names in SPAN_NAMES_BY_VERSION.items()}
_VERSION_BY_NUM_LABELS = {len(names): version for version, names in CLASS_NAMES_BY_VERSION.items()}


def build_label_space(class_names: Sequence[str], category_version: str) -> LabelSpace:
    """Derive the lookup tables from an ordered list of token class names."""
    span_names = [BACKGROUND]
    span_index = {BACKGROUND: 0}
    span_of: list[int] = []
    tag_of: list[str | None] = []
    background: int | None = None
    seen: dict[str, set[str]] = {}

    for index, name in enumerate(class_names):
        if name == BACKGROUND:
            background = index
            span_of.append(0)
            tag_of.append(None)
            continue
        boundary, _, category = name.partition("-")
        if boundary not in BOUNDARIES or not category:
            raise LabelError(f"class name {name!r} must use '<B|I|E|S>-<label>' format")
        if category not in span_index:
            span_index[category] = len(span_names)
            span_names.append(category)
        span_of.append(span_index[category])
        tag_of.append(boundary)
        seen.setdefault(category, set()).add(boundary)

    if background is None:
        raise LabelError(f"class names must include background label {BACKGROUND!r}")
    for category, boundaries in seen.items():
        missing = sorted(set(BOUNDARIES) - boundaries)
        if missing:
            raise LabelError(f"class names are missing boundaries [{', '.join(missing)}] for label {category!r}")

    return LabelSpace(
        class_names=tuple(class_names),
        span_names=tuple(span_names),
        span_of=tuple(span_of),
        tag_of=tuple(tag_of),
        background=background,
        category_version=category_version,
    )


def resolve_label_space(config: Mapping[str, object]) -> LabelSpace:
    """Resolve the label space a checkpoint config describes."""
    custom = _resolve_custom(config)
    if custom is not None:
        return custom

    configured: str | None = None
    if config.get("category_version") is not None:
        configured = str(config["category_version"]).strip().lower()
        if configured not in SPAN_NAMES_BY_VERSION:
            known = ", ".join(sorted(SPAN_NAMES_BY_VERSION))
            raise LabelError(f"unsupported category_version={config['category_version']!r}; known values: {known}")

    inferred: str | None = None
    num_labels = _num_labels(config)
    if num_labels is not None:
        inferred = _VERSION_BY_NUM_LABELS.get(num_labels)
        if inferred is None:
            known = ", ".join(f"{version}:{count}" for count, version in sorted(_VERSION_BY_NUM_LABELS.items()))
            raise LabelError(f"num_labels={num_labels} does not match a known label space ({known})")

    if configured is not None and inferred is not None and configured != inferred:
        raise LabelError(
            f"inconsistent label-space hints: category_version={configured!r}, "
            f"num_labels={num_labels} (implies {inferred!r})"
        )

    version = configured or inferred or DEFAULT_VERSION
    return build_label_space(CLASS_NAMES_BY_VERSION[version], version)


def _resolve_custom(config: Mapping[str, object]) -> LabelSpace | None:
    raw_span = config.get("span_class_names")
    raw_class = config.get("ner_class_names")
    if raw_span is None and raw_class is None:
        return None

    span_names: tuple[str, ...] = ()
    if raw_span is not None:
        parsed = _strings(raw_span, "span_class_names")
        if BACKGROUND not in parsed:
            raise LabelError(f"span_class_names must include background label {BACKGROUND!r}")
        span_names = (BACKGROUND, *(name for name in parsed if name != BACKGROUND))

    if raw_class is None:
        class_names = expand_with_boundaries(span_names)
    else:
        class_names = tuple(_strings(raw_class, "ner_class_names"))
        if span_names and class_names != expand_with_boundaries(span_names):
            raise LabelError("custom ner_class_names do not match span_class_names expansion")

    if config.get("category_version") is None:
        version = "custom"
    else:
        version = str(config["category_version"]).strip().lower()
        if not version:
            raise LabelError("category_version must be non-empty when provided")

    space = build_label_space(class_names, version)

    if version in SPAN_NAMES_BY_VERSION and (
        space.span_names != SPAN_NAMES_BY_VERSION[version] or space.class_names != CLASS_NAMES_BY_VERSION[version]
    ):
        raise LabelError(f"custom label names conflict with built-in category_version={version!r}")

    num_labels = _num_labels(config)
    if num_labels is not None and num_labels != len(class_names):
        raise LabelError(f"num_labels={num_labels} does not match ner_class_names length={len(class_names)}")
    return space


def _strings(value: object, field: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LabelError(f"{field} must be a sequence of strings")
    names: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise LabelError(f"{field}[{index}] must be a non-empty string (got {item!r})")
        names.append(item.strip())
    if not names:
        raise LabelError(f"{field} must not be empty")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise LabelError(f"{field} has duplicate values: {', '.join(duplicates)}")
    return names


def _num_labels(config: Mapping[str, object]) -> int | None:
    value = config.get("num_labels")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabelError(f"num_labels must be an integer (got {value!r})")
    return value
