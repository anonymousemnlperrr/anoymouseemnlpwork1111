from __future__ import annotations

UNKNOWN_MATERIAL_ID = -1

MATERIAL_CLASSES = ("cotton", "sand", "soybeans")
MATERIAL_TO_ID = {name: idx for idx, name in enumerate(MATERIAL_CLASSES)}
ID_TO_MATERIAL = {idx: name for name, idx in MATERIAL_TO_ID.items()}


def normalize_material_name(material_name: str | None) -> str | None:
    if material_name is None:
        return None

    name_lower = str(material_name).strip().lower()
    if not name_lower:
        return None
    if "cotton" in name_lower:
        return "cotton"
    if "sand" in name_lower:
        return "sand"
    if "soybean" in name_lower or "soybeans" in name_lower or name_lower == "bean":
        return "soybeans"
    return None


def is_material_probe_task(task_name: str) -> bool:
    return get_material_name_from_task_name(task_name) is not None


def get_material_name_from_task_name(task_name: str) -> str | None:
    name_lower = str(task_name).strip().lower()
    if "cotton" in name_lower:
        return "cotton"
    if "sand" in name_lower:
        return "sand"
    if "soybean" in name_lower or "soybeans" in name_lower or "bean" in name_lower:
        return "soybeans"
    return None


def get_material_label(material_name: str | None) -> int:
    normalized = normalize_material_name(material_name)
    if normalized is None:
        return UNKNOWN_MATERIAL_ID
    return MATERIAL_TO_ID.get(normalized, UNKNOWN_MATERIAL_ID)


def get_material_label_from_task_name(task_name: str) -> int:
    return get_material_label(get_material_name_from_task_name(task_name))


def decode_material_label(label: int) -> str:
    return ID_TO_MATERIAL.get(int(label), "unknown")