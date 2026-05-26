"""
VTLA/dataset/instructions.py

三级指令模板 + 物理属性标签

训练时随机采样, 评估时指定 level
无需逐帧标注

任务语义:
  - inboxpicking:   柔软布料, grip stability
  - grasp-blueberry: 脆弱蓝莓, 避免压碎
  - grasp-egg:      极脆弱鸡蛋, 避免破裂
  - grasp-sponge:   海绵, hard/soft 指抓取力度而非物体硬度
    - grasp-Sponge-hard = 用力抓取
    - grasp-Sponge-soft-* = 轻柔抓取
"""

import random


INSTRUCTIONS = {
    "inboxpicking": {
        "L0": [
            "Pick up the object from the box",
            "Grab an item from the bin",
            "Retrieve something from the container",
            "Take out the object",
            "Lift the item from inside the box",
        ],
        "L1": [
            "Pick up the soft cloth from the box",
            "Grab the flexible fabric from the bin",
            "Retrieve the soft textile material",
            "Take out the lightweight cloth",
            "Lift the foldable fabric from the container",
        ],
        "L2": [
            "Carefully grab the flexible cloth without dropping it",
            "Gently pick up the soft fabric and hold it securely",
            "Retrieve the lightweight cloth without letting it slip",
            "Firmly grasp the foldable textile from the box",
            "Pick up the soft cloth while maintaining a stable grip",
        ],
    },
    "grasp-blueberry": {
        "L0": [
            "Pick up the object",
            "Grab the item from the plate",
            "Take the small object",
            "Lift the item carefully",
            "Retrieve the object from the surface",
        ],
        "L1": [
            "Pick up the fragile blueberry",
            "Grab the delicate small berry",
            "Take the soft, crushable fruit",
            "Lift the tender blueberry from the plate",
            "Retrieve the easily damaged berry",
        ],
        "L2": [
            "Gently grasp the delicate blueberry without crushing it",
            "Very carefully pick up the fragile berry with minimal force",
            "Softly lift the tender blueberry without squeezing",
            "Handle the crushable berry with extreme care",
            "Pick up the delicate fruit using the lightest possible grip",
        ],
    },
    "grasp-egg": {
        "L0": [
            "Pick up the object",
            "Grab the item from the table",
            "Lift the object",
            "Take the item",
            "Retrieve the object from the surface",
        ],
        "L1": [
            "Pick up the fragile egg",
            "Lift the delicate egg carefully",
            "Take the breakable egg from the table",
            "Grab the easily cracked egg gently",
            "Retrieve the thin-shelled egg with care",
        ],
        "L2": [
            "Gently lift the fragile egg using enough force to support it without squeezing",
            "Pick up the delicate egg with steady light support and no crushing pressure",
            "Carefully grasp the breakable egg with controlled gentle force",
            "Lift the thin-shelled egg securely while keeping the grip soft and stable",
            "Support the fragile egg with the minimal necessary force to raise it safely",
        ],
    },
    "grasp-sponge": {
        "L0": [
            "Pick up the sponge",
            "Grab the sponge from the table",
            "Lift the sponge",
            "Take the sponge",
            "Retrieve the sponge from the surface",
        ],
        "L1": [
            "Pick up the compressible sponge",
            "Grab the soft, squeezable sponge",
            "Take the deformable sponge",
            "Lift the elastic sponge",
            "Retrieve the flexible sponge",
        ],
        "L2": [
            "Gently pick up the sponge without squeezing it",
            "Softly grasp the sponge with minimal compression",
            "Carefully lift the sponge using light force",
            "Pick up the sponge without deforming it",
            "Lift the sponge gently to preserve its shape",
        ],
    },
    "grabbing-cotton": {
        "L0": [
            "Pick up the bag",
            "Grab the object from the table",
            "Lift the bag",
            "Take the bag from the surface",
            "Retrieve the object",
        ],
        "L1": [
            "Pick up the soft bag",
            "Grab the compressible opaque bag",
            "Lift the light, deformable bag",
            "Take the soft, lightweight bag",
            "Retrieve the flexible bag",
        ],
        "L2": [
            "Gently lift the soft bag with stable support",
            "Pick up the compressible bag using minimal pressure while keeping it secure",
            "Carefully grasp the light, deformable bag without over-squeezing it",
            "Lift the soft bag gently while maintaining a stable hold",
            "Support the compressible bag with light pressure and smooth motion",
        ],
    },
    "grabbing-sand": {
        "L0": [
            "Pick up the bag",
            "Grab the object from the table",
            "Lift the bag",
            "Take the bag from the surface",
            "Retrieve the object",
        ],
        "L1": [
            "Pick up the dense bag",
            "Grab the heavy opaque bag",
            "Lift the compact bag",
            "Take the heavy, fine-grained bag",
            "Retrieve the dense bag",
        ],
        "L2": [
            "Firmly lift the heavy bag and keep it level",
            "Pick up the dense bag with steady support and no sudden jerk",
            "Raise the heavy, compact bag smoothly while maintaining a stable grip",
            "Lift the fine-grained bag with controlled support and smooth motion",
            "Support the dense bag securely while keeping the motion stable",
        ],
    },
    "grabbing-soybeans": {
        "L0": [
            "Pick up the bag",
            "Grab the object from the table",
            "Lift the bag",
            "Take the bag from the surface",
            "Retrieve the object",
        ],
        "L1": [
            "Pick up the grainy bag",
            "Grab the bumpy opaque bag",
            "Lift the coarse-filled bag",
            "Take the grainy, coarse bag",
            "Retrieve the bumpy bag",
        ],
        "L2": [
            "Lift the grainy bag with steady support while avoiding slips",
            "Pick up the bumpy bag with controlled pressure and a stable hold",
            "Raise the coarse-filled bag smoothly without shaking it",
            "Support the grainy bag with stable pressure and smooth motion",
            "Lift the coarse-grained bag carefully while keeping the grip steady",
        ],
    },
}


MATERIAL_PROBE_CLASSIFICATION_PROMPTS = [
    "Identify the material inside the opaque opaque bag by touch.",
    "Classify the material inside the opaque bag from the tactile signal.",
    "Determine whether the opaque bag contains cotton, sand, or soybeans.",
]


def get_task_key(task_name: str) -> str:
    """从完整任务名模糊匹配到模板 key

    e.g. "inboxpicking-03" → "inboxpicking"
         "grasp-blueberry-01" → "grasp-blueberry"
         "grasp-egg" → "grasp-egg"
         "grasp-egg-02" → "grasp-egg"
         "grasp-Sponge-hard" → "grasp-sponge"
         "grasp-Sponge-soft-01" → "grasp-sponge"
    """
    name_lower = task_name.lower()
    # 精确匹配优先
    for key in INSTRUCTIONS:
        if key.lower() in name_lower:
            return key
    # sponge 文件夹名为 grasp-Sponge-* 需要特殊匹配
    if "sponge" in name_lower:
        return "grasp-sponge"
    if "cotton" in name_lower:
        return "grabbing-cotton"
    if "sand" in name_lower:
        return "grabbing-sand"
    if "soybean" in name_lower or "soybeans" in name_lower or "bean" in name_lower:
        return "grabbing-soybeans"
    return "inboxpicking"  # fallback


def sample_instruction(task_name: str, level: str | None = None) -> str:
    """
    采样一条指令

    task_name: 任务名 (e.g., "inboxpicking-03", "grasp-blueberry-01")
    level:     None=随机, "L0"/"L1"/"L2"=指定
    """
    key = get_task_key(task_name)
    templates = INSTRUCTIONS[key]
    if level is None:
        level = random.choice(["L0", "L1", "L2"])
    return random.choice(templates[level])


def sample_material_probe_classification_prompt() -> str:
    return random.choice(MATERIAL_PROBE_CLASSIFICATION_PROMPTS)


# ============================================================================
# Minimal-pair instructions for controlled Instruction Gradient evaluation
# 每对只替换 physical-property adjective (PPA), 其余语法完全一致
# ============================================================================

MINIMAL_PAIRS = {
    "grasp-blueberry": [
        {
            "source": "Gently grasp the fragile berry without crushing it",
            "swap":   "Gently grasp the sturdy berry without crushing it",
            "changed_ppa": ("fragile", "sturdy"),
        },
        {
            "source": "Very carefully pick up the delicate berry with minimal force",
            "swap":   "Very carefully pick up the robust berry with minimal force",
            "changed_ppa": ("delicate", "robust"),
        },
        {
            "source": "Softly lift the tender blueberry without squeezing",
            "swap":   "Softly lift the tough blueberry without squeezing",
            "changed_ppa": ("tender", "tough"),
        },
    ],
    "inboxpicking": [
        {
            "source": "Carefully grab the flexible cloth without dropping it",
            "swap":   "Carefully grab the rigid cloth without dropping it",
            "changed_ppa": ("flexible", "rigid"),
        },
        {
            "source": "Gently pick up the soft fabric and hold it securely",
            "swap":   "Gently pick up the stiff fabric and hold it securely",
            "changed_ppa": ("soft", "stiff"),
        },
        {
            "source": "Retrieve the lightweight cloth without letting it slip",
            "swap":   "Retrieve the dense cloth without letting it slip",
            "changed_ppa": ("lightweight", "dense"),
        },
    ],
    "grasp-egg": [
        {
            "source": "Gently support the fragile egg with steady light force to lift it safely",
            "swap":   "Gently support the sturdy egg with steady light force to lift it safely",
            "changed_ppa": ("fragile", "sturdy"),
        },
        {
            "source": "Carefully grasp the brittle egg with controlled gentle support",
            "swap":   "Carefully grasp the durable egg with controlled gentle support",
            "changed_ppa": ("brittle", "durable"),
        },
    ],
    "grasp-sponge": [
        {
            "source": "Gently pick up the sponge without squeezing it",
            "swap":   "Forcefully pick up the sponge without squeezing it",
            "changed_ppa": ("Gently", "Forcefully"),
        },
        {
            "source": "Softly grasp the sponge with minimal compression",
            "swap":   "Firmly grasp the sponge with minimal compression",
            "changed_ppa": ("Softly", "Firmly"),
        },
    ],
    "grabbing-cotton": [
        {
            "source": "Gently lift the soft bag with stable support",
            "swap":   "Gently lift the dense bag with stable support",
            "changed_ppa": ("soft", "dense"),
        },
        {
            "source": "Pick up the compressible bag using minimal pressure while keeping it secure",
            "swap":   "Pick up the firm bag using minimal pressure while keeping it secure",
            "changed_ppa": ("compressible", "firm"),
        },
        {
            "source": "Carefully grasp the light, deformable bag without over-squeezing it",
            "swap":   "Carefully grasp the heavy, rigid bag without over-squeezing it",
            "changed_ppa": ("light, deformable", "heavy, rigid"),
        },
    ],
    "grabbing-sand": [
        {
            "source": "Firmly lift the heavy bag and keep it level",
            "swap":   "Firmly lift the light bag and keep it level",
            "changed_ppa": ("heavy", "light"),
        },
        {
            "source": "Pick up the dense bag with steady support and no sudden jerk",
            "swap":   "Pick up the soft bag with steady support and no sudden jerk",
            "changed_ppa": ("dense", "soft"),
        },
        {
            "source": "Lift the fine-grained bag with controlled support and smooth motion",
            "swap":   "Lift the coarse-grained bag with controlled support and smooth motion",
            "changed_ppa": ("fine-grained", "coarse-grained"),
        },
    ],
    "grabbing-soybeans": [
        {
            "source": "Lift the grainy bag with steady support while avoiding slips",
            "swap":   "Lift the smooth bag with steady support while avoiding slips",
            "changed_ppa": ("grainy", "smooth"),
        },
        {
            "source": "Pick up the bumpy bag with controlled pressure and a stable hold",
            "swap":   "Pick up the flat bag with controlled pressure and a stable hold",
            "changed_ppa": ("bumpy", "flat"),
        },
        {
            "source": "Raise the coarse-filled bag smoothly without shaking it",
            "swap":   "Raise the fine-filled bag smoothly without shaking it",
            "changed_ppa": ("coarse-filled", "fine-filled"),
        },
    ],
}
