"""
维修性设计核查专用实体本体定义
Maintainability Design Verification Entity Ontology

根据客户需求定义，针对以下核查领域：
1. 简化维修 - LRU、维修流程、维修资源
2. 维修可达性 - 口盖信息、维修通道
3. 标准化与互换性 - 标准件
4. 维修安全 - 危险注意事项
"""

MAINTAINABILITY_ENTITY_ONTOLOGY = {
    # ==================== 简化维修 ====================
    "LRU": {
        "description": "可更换单元 (Line Replaceable Unit)",
        "chinese": "可更换单元",
        "examples": ["circuit card", "module", "assembly", "replaceable unit", "LRU", "SRU", "WRA"],
        "attributes": ["name", "part_number", "location", "replacement_time", "skill_level"],
        "patterns": [
            r'\b(LRU|SRU|WRA|replaceable unit|replaceable module|replaceable assembly)\b',
            r'\b(circuit card|card assembly|module assembly)\b',
            r'\b(can be replaced|is replaceable|replace the)\b',
        ]
    },
    
    "MaintenanceProcedure": {
        "description": "维修流程/操作步骤",
        "chinese": "维修流程",
        "examples": ["removal procedure", "installation steps", "calibration process", "alignment procedure"],
        "attributes": ["name", "steps", "duration", "frequency", "prerequisites"],
        "patterns": [
            r'\b(procedure|process|steps|instructions)\b',
            r'\b(remove|install|replace|calibrate|align|adjust|inspect|test|verify)\b',
            r'\b(maintenance|repair|service|overhaul)\b',
        ]
    },
    
    "MaintenanceResource": {
        "description": "维修资源（人员、场地、维修级别、工具）",
        "chinese": "维修资源",
        "examples": ["technician", "depot", "organizational level", "test equipment", "special tool"],
        "attributes": ["resource_type", "name", "specification", "quantity", "availability"],
        "sub_types": {
            "Personnel": {
                "patterns": [r'\b(technician|operator|mechanic|engineer|personnel|MOS)\b'],
                "examples": ["electronics technician", "maintenance personnel"]
            },
            "Facility": {
                "patterns": [r'\b(depot|shop|facility|bench|station|area)\b'],
                "examples": ["maintenance shop", "test station", "calibration facility"]
            },
            "MaintenanceLevel": {
                "patterns": [r'\b(organizational|intermediate|depot|O-level|I-level|D-level)\b'],
                "examples": ["organizational maintenance", "depot-level repair"]
            },
            "Tool": {
                "patterns": [r'\b(tool|equipment|tester|meter|gauge|wrench|driver|kit)\b'],
                "examples": ["multimeter", "torque wrench", "test set", "special tool"]
            }
        }
    },
    
    # ==================== 维修可达性 ====================
    "AccessPanel": {
        "description": "口盖/检修口信息",
        "chinese": "口盖信息",
        "examples": ["access panel", "access door", "cover", "hatch", "inspection port"],
        "attributes": ["name", "location", "size", "fastener_type", "opening_direction"],
        "patterns": [
            r'\b(access panel|access door|access cover|access port)\b',
            r'\b(inspection (panel|port|door|cover|hatch))\b',
            r'\b(removable (panel|cover|plate))\b',
            r'\b(hinged (panel|door|cover))\b',
        ]
    },
    
    "MaintenanceAccess": {
        "description": "维修通道/可达性信息",
        "chinese": "维修通道",
        "examples": ["maintenance access", "service access", "reach distance", "clearance"],
        "attributes": ["access_point", "clearance", "reach_requirement", "body_position"],
        "patterns": [
            r'\b(access|accessibility|clearance|reach)\b',
            r'\b(work space|working space|maintenance area)\b',
            r'\b(can be accessed|accessible from|provides access)\b',
        ]
    },
    
    # ==================== 标准化与互换性 ====================
    "StandardPart": {
        "description": "标准件/通用件",
        "chinese": "标准件",
        "examples": ["AN bolt", "MS screw", "NAS fastener", "standard hardware"],
        "attributes": ["standard_number", "name", "specification", "interchangeable_with"],
        "patterns": [
            r'\b(AN|MS|NAS|MIL-STD|SAE|ISO|ASTM)[\s-]?\d+\b',
            r'\b(standard (part|hardware|fastener|bolt|screw|nut|washer))\b',
            r'\b(commercial off-the-shelf|COTS)\b',
        ]
    },
    
    # ==================== 维修安全 ====================
    "SafetyWarning": {
        "description": "危险注意事项/安全警告",
        "chinese": "安全警告",
        "examples": ["WARNING", "CAUTION", "NOTE", "hazard", "danger"],
        "attributes": ["warning_type", "hazard_description", "precaution", "consequence"],
        "patterns": [
            r'\b(WARNING|CAUTION|DANGER|NOTE)\b',
            r'\b(hazard|hazardous|dangerous|risk)\b',
            r'\b(electric shock|high voltage|burn|injury|toxic|explosive)\b',
            r'\b(safety precaution|protective equipment|PPE)\b',
        ]
    },
}

# 实体类型列表
ENTITY_TYPES = list(MAINTAINABILITY_ENTITY_ONTOLOGY.keys())

# 关系类型定义
RELATION_TYPES = [
    "requires_tool",           # LRU/Procedure → Tool
    "requires_personnel",      # Procedure → Personnel
    "performed_at_level",      # Procedure → MaintenanceLevel
    "accessed_via",            # LRU → AccessPanel
    "uses_standard_part",      # LRU → StandardPart
    "has_warning",             # Procedure → SafetyWarning
    "located_in",              # LRU → MaintenanceAccess
    "part_of",                 # LRU → LRU (hierarchy)
    "followed_by",             # Procedure → Procedure (sequence)
    "defined_in",              # Entity → Source (document, page)
]

# 中英文映射
ENTITY_TYPE_CHINESE = {
    "LRU": "可更换单元",
    "MaintenanceProcedure": "维修流程",
    "MaintenanceResource": "维修资源",
    "AccessPanel": "口盖信息",
    "MaintenanceAccess": "维修通道",
    "StandardPart": "标准件",
    "SafetyWarning": "安全警告",
}

def get_all_patterns():
    """获取所有实体类型的正则模式"""
    all_patterns = {}
    for etype, config in MAINTAINABILITY_ENTITY_ONTOLOGY.items():
        patterns = config.get('patterns', [])
        if 'sub_types' in config:
            for sub_type, sub_config in config['sub_types'].items():
                patterns.extend(sub_config.get('patterns', []))
        all_patterns[etype] = patterns
    return all_patterns
