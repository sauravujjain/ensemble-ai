"""
Ensemble Rules Module for Fashion Outfit Building

Defines:
- Garment type taxonomy
- Compatibility rules between garment types
- Occasion-based outfit templates
- Style coherence rules

Author: Fashion Ensemble Builder
"""

from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


# =============================================================================
# GARMENT TAXONOMY
# =============================================================================

class GarmentCategory(Enum):
    """High-level garment categories"""
    TOP = "top"
    BOTTOM = "bottom"
    FULL_BODY = "full_body"       # Dresses, jumpsuits, sarees
    OUTERWEAR = "outerwear"
    FOOTWEAR = "footwear"
    ACCESSORY = "accessory"


class StyleFamily(Enum):
    """Style families that should generally stay together"""
    CASUAL = "casual"
    FORMAL = "formal"
    ETHNIC = "ethnic"
    STREETWEAR = "streetwear"
    ATHLEISURE = "athleisure"
    BOHEMIAN = "bohemian"
    MINIMALIST = "minimalist"
    PREPPY = "preppy"


class Occasion(Enum):
    """Occasion types for outfit matching"""
    DAILY = "daily"
    WORK = "work"
    FORMAL_EVENT = "formal_event"
    PARTY = "party"
    WEDDING = "wedding"
    FESTIVE = "festive"
    BEACH = "beach"
    WORKOUT = "workout"
    DATE = "date"
    BRUNCH = "brunch"


# =============================================================================
# GARMENT TYPE DEFINITIONS
# =============================================================================

@dataclass
class GarmentType:
    """Definition of a garment type"""
    name: str
    category: GarmentCategory
    style_families: Set[StyleFamily]
    occasions: Set[Occasion]
    formality_score: int  # 1 (very casual) to 5 (very formal)
    keywords: List[str]   # For text matching


# =============================================================================
# TOPS
# =============================================================================

TOPS = {
    # Casual Tops
    'tshirt': GarmentType(
        name='tshirt',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.CASUAL, StyleFamily.STREETWEAR, StyleFamily.ATHLEISURE},
        occasions={Occasion.DAILY, Occasion.BEACH, Occasion.BRUNCH},
        formality_score=1,
        keywords=['t-shirt', 'tee', 'graphic tee', 'plain tee']
    ),
    'tank_top': GarmentType(
        name='tank_top',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.CASUAL, StyleFamily.ATHLEISURE},
        occasions={Occasion.DAILY, Occasion.BEACH, Occasion.WORKOUT},
        formality_score=1,
        keywords=['tank', 'sleeveless', 'vest top', 'camisole']
    ),
    'polo': GarmentType(
        name='polo',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.CASUAL, StyleFamily.PREPPY},
        occasions={Occasion.DAILY, Occasion.WORK, Occasion.BRUNCH},
        formality_score=2,
        keywords=['polo shirt', 'golf shirt', 'collar tee']
    ),
    'henley': GarmentType(
        name='henley',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.CASUAL},
        occasions={Occasion.DAILY, Occasion.DATE},
        formality_score=2,
        keywords=['henley', 'button placket']
    ),
    'hoodie': GarmentType(
        name='hoodie',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.CASUAL, StyleFamily.STREETWEAR, StyleFamily.ATHLEISURE},
        occasions={Occasion.DAILY},
        formality_score=1,
        keywords=['hoodie', 'hooded sweatshirt', 'pullover hoodie']
    ),
    'sweatshirt': GarmentType(
        name='sweatshirt',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.CASUAL, StyleFamily.STREETWEAR},
        occasions={Occasion.DAILY},
        formality_score=1,
        keywords=['sweatshirt', 'crewneck', 'pullover']
    ),
    'crop_top': GarmentType(
        name='crop_top',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.CASUAL, StyleFamily.STREETWEAR, StyleFamily.BOHEMIAN},
        occasions={Occasion.DAILY, Occasion.PARTY, Occasion.BEACH},
        formality_score=1,
        keywords=['crop top', 'cropped', 'midriff']
    ),
    
    # Formal/Smart Tops
    'shirt': GarmentType(
        name='shirt',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.FORMAL, StyleFamily.CASUAL, StyleFamily.PREPPY},
        occasions={Occasion.WORK, Occasion.FORMAL_EVENT, Occasion.DATE, Occasion.BRUNCH},
        formality_score=4,
        keywords=['shirt', 'button-down', 'button-up', 'dress shirt', 'oxford']
    ),
    'blouse': GarmentType(
        name='blouse',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.FORMAL, StyleFamily.BOHEMIAN},
        occasions={Occasion.WORK, Occasion.DATE, Occasion.BRUNCH, Occasion.FORMAL_EVENT},
        formality_score=3,
        keywords=['blouse', 'formal top', 'silk top']
    ),
    'tunic': GarmentType(
        name='tunic',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.CASUAL, StyleFamily.BOHEMIAN, StyleFamily.ETHNIC},
        occasions={Occasion.DAILY, Occasion.BRUNCH},
        formality_score=2,
        keywords=['tunic', 'long top', 'tunic top']
    ),
    
    # Ethnic Tops
    'kurta': GarmentType(
        name='kurta',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.ETHNIC},
        occasions={Occasion.DAILY, Occasion.FESTIVE, Occasion.WEDDING, Occasion.WORK},
        formality_score=3,
        keywords=['kurta', 'kurti', 'kurtis', 'ethnic top']
    ),
    'kurti': GarmentType(
        name='kurti',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.ETHNIC, StyleFamily.CASUAL},
        occasions={Occasion.DAILY, Occasion.WORK, Occasion.FESTIVE},
        formality_score=2,
        keywords=['kurti', 'short kurta']
    ),
    'choli': GarmentType(
        name='choli',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.ETHNIC},
        occasions={Occasion.WEDDING, Occasion.FESTIVE, Occasion.PARTY},
        formality_score=4,
        keywords=['choli', 'blouse for lehenga', 'saree blouse']
    ),
    
    # Knitwear
    'sweater': GarmentType(
        name='sweater',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.CASUAL, StyleFamily.PREPPY, StyleFamily.MINIMALIST},
        occasions={Occasion.DAILY, Occasion.WORK, Occasion.DATE},
        formality_score=2,
        keywords=['sweater', 'jumper', 'pullover', 'knit']
    ),
    'cardigan': GarmentType(
        name='cardigan',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.CASUAL, StyleFamily.PREPPY, StyleFamily.MINIMALIST},
        occasions={Occasion.DAILY, Occasion.WORK, Occasion.BRUNCH},
        formality_score=2,
        keywords=['cardigan', 'knit cardigan', 'open front']
    ),
    'turtleneck': GarmentType(
        name='turtleneck',
        category=GarmentCategory.TOP,
        style_families={StyleFamily.MINIMALIST, StyleFamily.FORMAL},
        occasions={Occasion.WORK, Occasion.DATE, Occasion.FORMAL_EVENT},
        formality_score=3,
        keywords=['turtleneck', 'roll neck', 'mock neck', 'high neck']
    ),
}


# =============================================================================
# BOTTOMS
# =============================================================================

BOTTOMS = {
    # Casual Bottoms
    'jeans': GarmentType(
        name='jeans',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.CASUAL, StyleFamily.STREETWEAR},
        occasions={Occasion.DAILY, Occasion.BRUNCH, Occasion.DATE},
        formality_score=2,
        keywords=['jeans', 'denim', 'skinny jeans', 'straight jeans', 'wide leg jeans']
    ),
    'shorts': GarmentType(
        name='shorts',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.CASUAL, StyleFamily.ATHLEISURE},
        occasions={Occasion.DAILY, Occasion.BEACH},
        formality_score=1,
        keywords=['shorts', 'denim shorts', 'bermuda']
    ),
    'joggers': GarmentType(
        name='joggers',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.CASUAL, StyleFamily.ATHLEISURE, StyleFamily.STREETWEAR},
        occasions={Occasion.DAILY, Occasion.WORKOUT},
        formality_score=1,
        keywords=['joggers', 'sweatpants', 'track pants']
    ),
    'leggings': GarmentType(
        name='leggings',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.CASUAL, StyleFamily.ATHLEISURE},
        occasions={Occasion.DAILY, Occasion.WORKOUT},
        formality_score=1,
        keywords=['leggings', 'tights', 'yoga pants']
    ),
    'cargo_pants': GarmentType(
        name='cargo_pants',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.CASUAL, StyleFamily.STREETWEAR},
        occasions={Occasion.DAILY},
        formality_score=1,
        keywords=['cargo', 'cargo pants', 'utility pants']
    ),
    
    # Smart/Formal Bottoms
    'trousers': GarmentType(
        name='trousers',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.FORMAL, StyleFamily.MINIMALIST},
        occasions={Occasion.WORK, Occasion.FORMAL_EVENT},
        formality_score=4,
        keywords=['trousers', 'dress pants', 'formal pants', 'slacks']
    ),
    'chinos': GarmentType(
        name='chinos',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.CASUAL, StyleFamily.PREPPY},
        occasions={Occasion.WORK, Occasion.BRUNCH, Occasion.DATE},
        formality_score=3,
        keywords=['chinos', 'khakis', 'cotton pants']
    ),
    
    # Skirts
    'skirt': GarmentType(
        name='skirt',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.CASUAL, StyleFamily.FORMAL, StyleFamily.BOHEMIAN},
        occasions={Occasion.DAILY, Occasion.WORK, Occasion.DATE, Occasion.BRUNCH},
        formality_score=3,
        keywords=['skirt', 'a-line skirt', 'pencil skirt', 'midi skirt', 'mini skirt']
    ),
    'maxi_skirt': GarmentType(
        name='maxi_skirt',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.BOHEMIAN, StyleFamily.ETHNIC},
        occasions={Occasion.DAILY, Occasion.BEACH, Occasion.FESTIVE},
        formality_score=2,
        keywords=['maxi skirt', 'long skirt', 'floor length skirt']
    ),
    'pleated_skirt': GarmentType(
        name='pleated_skirt',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.FORMAL, StyleFamily.PREPPY},
        occasions={Occasion.WORK, Occasion.DATE, Occasion.FORMAL_EVENT},
        formality_score=3,
        keywords=['pleated skirt', 'accordion skirt']
    ),
    
    # Ethnic Bottoms
    'palazzos': GarmentType(
        name='palazzos',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.ETHNIC, StyleFamily.BOHEMIAN, StyleFamily.CASUAL},
        occasions={Occasion.DAILY, Occasion.FESTIVE, Occasion.WORK},
        formality_score=2,
        keywords=['palazzos', 'palazzo pants', 'wide leg']
    ),
    'salwar': GarmentType(
        name='salwar',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.ETHNIC},
        occasions={Occasion.DAILY, Occasion.FESTIVE, Occasion.WEDDING},
        formality_score=3,
        keywords=['salwar', 'shalwar', 'patiala']
    ),
    'churidar': GarmentType(
        name='churidar',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.ETHNIC},
        occasions={Occasion.FESTIVE, Occasion.WEDDING, Occasion.WORK},
        formality_score=3,
        keywords=['churidar', 'fitted bottom']
    ),
    'dhoti_pants': GarmentType(
        name='dhoti_pants',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.ETHNIC},
        occasions={Occasion.FESTIVE, Occasion.WEDDING},
        formality_score=3,
        keywords=['dhoti pants', 'dhoti', 'draped pants']
    ),
    'lehenga_skirt': GarmentType(
        name='lehenga_skirt',
        category=GarmentCategory.BOTTOM,
        style_families={StyleFamily.ETHNIC},
        occasions={Occasion.WEDDING, Occasion.FESTIVE, Occasion.PARTY},
        formality_score=5,
        keywords=['lehenga', 'ghagra', 'chaniya']
    ),
}


# =============================================================================
# OUTERWEAR
# =============================================================================

OUTERWEAR = {
    'blazer': GarmentType(
        name='blazer',
        category=GarmentCategory.OUTERWEAR,
        style_families={StyleFamily.FORMAL, StyleFamily.PREPPY, StyleFamily.MINIMALIST},
        occasions={Occasion.WORK, Occasion.FORMAL_EVENT, Occasion.DATE},
        formality_score=4,
        keywords=['blazer', 'sport coat', 'suit jacket']
    ),
    'jacket': GarmentType(
        name='jacket',
        category=GarmentCategory.OUTERWEAR,
        style_families={StyleFamily.CASUAL, StyleFamily.STREETWEAR},
        occasions={Occasion.DAILY},
        formality_score=2,
        keywords=['jacket', 'bomber', 'varsity']
    ),
    'denim_jacket': GarmentType(
        name='denim_jacket',
        category=GarmentCategory.OUTERWEAR,
        style_families={StyleFamily.CASUAL, StyleFamily.STREETWEAR},
        occasions={Occasion.DAILY, Occasion.BRUNCH},
        formality_score=2,
        keywords=['denim jacket', 'jean jacket']
    ),
    'leather_jacket': GarmentType(
        name='leather_jacket',
        category=GarmentCategory.OUTERWEAR,
        style_families={StyleFamily.STREETWEAR, StyleFamily.CASUAL},
        occasions={Occasion.DAILY, Occasion.PARTY, Occasion.DATE},
        formality_score=2,
        keywords=['leather jacket', 'biker jacket', 'moto jacket']
    ),
    'coat': GarmentType(
        name='coat',
        category=GarmentCategory.OUTERWEAR,
        style_families={StyleFamily.FORMAL, StyleFamily.MINIMALIST},
        occasions={Occasion.WORK, Occasion.FORMAL_EVENT},
        formality_score=4,
        keywords=['coat', 'overcoat', 'trench', 'peacoat', 'wool coat']
    ),
    'vest': GarmentType(
        name='vest',
        category=GarmentCategory.OUTERWEAR,
        style_families={StyleFamily.FORMAL, StyleFamily.PREPPY},
        occasions={Occasion.WORK, Occasion.FORMAL_EVENT},
        formality_score=4,
        keywords=['vest', 'waistcoat', 'gilet']
    ),
    'shrug': GarmentType(
        name='shrug',
        category=GarmentCategory.OUTERWEAR,
        style_families={StyleFamily.CASUAL, StyleFamily.ETHNIC},
        occasions={Occasion.DAILY, Occasion.FESTIVE},
        formality_score=2,
        keywords=['shrug', 'bolero']
    ),
    'nehru_jacket': GarmentType(
        name='nehru_jacket',
        category=GarmentCategory.OUTERWEAR,
        style_families={StyleFamily.ETHNIC},
        occasions={Occasion.FESTIVE, Occasion.WEDDING, Occasion.FORMAL_EVENT},
        formality_score=4,
        keywords=['nehru jacket', 'bandhgala', 'modi jacket', 'ethnic jacket']
    ),
}


# =============================================================================
# FULL BODY GARMENTS
# =============================================================================

FULL_BODY = {
    'dress': GarmentType(
        name='dress',
        category=GarmentCategory.FULL_BODY,
        style_families={StyleFamily.CASUAL, StyleFamily.FORMAL, StyleFamily.BOHEMIAN},
        occasions={Occasion.DAILY, Occasion.WORK, Occasion.DATE, Occasion.PARTY, Occasion.BRUNCH},
        formality_score=3,
        keywords=['dress', 'frock', 'midi dress', 'maxi dress', 'mini dress']
    ),
    'gown': GarmentType(
        name='gown',
        category=GarmentCategory.FULL_BODY,
        style_families={StyleFamily.FORMAL},
        occasions={Occasion.FORMAL_EVENT, Occasion.WEDDING, Occasion.PARTY},
        formality_score=5,
        keywords=['gown', 'evening gown', 'ball gown']
    ),
    'jumpsuit': GarmentType(
        name='jumpsuit',
        category=GarmentCategory.FULL_BODY,
        style_families={StyleFamily.CASUAL, StyleFamily.FORMAL},
        occasions={Occasion.DAILY, Occasion.PARTY, Occasion.DATE},
        formality_score=3,
        keywords=['jumpsuit', 'romper', 'playsuit']
    ),
    'saree': GarmentType(
        name='saree',
        category=GarmentCategory.FULL_BODY,
        style_families={StyleFamily.ETHNIC},
        occasions={Occasion.WEDDING, Occasion.FESTIVE, Occasion.FORMAL_EVENT, Occasion.WORK},
        formality_score=4,
        keywords=['saree', 'sari', 'designer saree']
    ),
    'lehenga': GarmentType(
        name='lehenga',
        category=GarmentCategory.FULL_BODY,
        style_families={StyleFamily.ETHNIC},
        occasions={Occasion.WEDDING, Occasion.FESTIVE, Occasion.PARTY},
        formality_score=5,
        keywords=['lehenga', 'lehenga choli', 'chaniya choli', 'ghagra choli']
    ),
    'anarkali': GarmentType(
        name='anarkali',
        category=GarmentCategory.FULL_BODY,
        style_families={StyleFamily.ETHNIC},
        occasions={Occasion.FESTIVE, Occasion.WEDDING, Occasion.PARTY},
        formality_score=4,
        keywords=['anarkali', 'anarkali suit', 'floor length kurta']
    ),
    'salwar_suit': GarmentType(
        name='salwar_suit',
        category=GarmentCategory.FULL_BODY,
        style_families={StyleFamily.ETHNIC},
        occasions={Occasion.DAILY, Occasion.FESTIVE, Occasion.WORK},
        formality_score=3,
        keywords=['salwar suit', 'salwar kameez', 'punjabi suit']
    ),
}


# =============================================================================
# COMBINED GARMENT REGISTRY
# =============================================================================

ALL_GARMENTS = {
    **TOPS,
    **BOTTOMS,
    **OUTERWEAR,
    **FULL_BODY,
}


# =============================================================================
# COMPATIBILITY RULES
# =============================================================================

# Which categories can be combined
CATEGORY_COMPATIBILITY = {
    GarmentCategory.TOP: {GarmentCategory.BOTTOM, GarmentCategory.OUTERWEAR},
    GarmentCategory.BOTTOM: {GarmentCategory.TOP, GarmentCategory.OUTERWEAR},
    GarmentCategory.FULL_BODY: {GarmentCategory.OUTERWEAR},  # Dresses only need outerwear
    GarmentCategory.OUTERWEAR: {GarmentCategory.TOP, GarmentCategory.BOTTOM, GarmentCategory.FULL_BODY},
}


# =============================================================================
# OUTFIT REQUIREMENTS BY ANCHOR CATEGORY
# =============================================================================

# What categories are needed to complete an outfit based on anchor type
OUTFIT_REQUIREMENTS = {
    'outerwear': {
        'required': ['top', 'bottom'],
        'optional': [],
        'description': 'Outerwear needs tops underneath and bottoms'
    },
    'top': {
        'required': ['bottom'],
        'optional': ['outerwear'],
        'description': 'Tops need bottoms, optionally outerwear'
    },
    'bottom': {
        'required': ['top'],
        'optional': ['outerwear'],
        'description': 'Bottoms need tops, optionally outerwear'
    },
    'full_body': {
        'required': [],
        'optional': ['outerwear'],
        'description': 'Full body items (dresses) only need optional outerwear'
    }
}

# Default garments per category for queries
CATEGORY_GARMENT_DEFAULTS = {
    'top': {
        'western_casual': ['tshirt', 'polo', 'henley', 'tank_top'],
        'western_formal': ['shirt', 'blouse', 'turtleneck'],
        'ethnic': ['kurta', 'kurti', 'choli'],
        'default': ['shirt', 'blouse', 'top', 'tshirt']
    },
    'bottom': {
        'western_casual': ['jeans', 'shorts', 'joggers', 'chinos'],
        'western_formal': ['trousers', 'skirt', 'pleated_skirt'],
        'ethnic': ['palazzos', 'salwar', 'churidar', 'dhoti_pants'],
        'default': ['trousers', 'pants', 'jeans', 'skirt']
    },
    'outerwear': {
        'western_casual': ['denim_jacket', 'hoodie', 'cardigan', 'jacket'],
        'western_formal': ['blazer', 'coat', 'vest'],
        'ethnic': ['nehru_jacket', 'shrug'],
        'default': ['blazer', 'jacket', 'cardigan']
    }
}

# Style to sub-category mapping
STYLE_MAPPING = {
    'casual': 'western_casual',
    'formal': 'western_formal',
    'work': 'western_formal',
    'western': 'western_casual',
    'ethnic': 'ethnic',
    'festive': 'ethnic',
    'wedding': 'ethnic',
    'party': 'western_casual',
    'date': 'western_casual',
    'brunch': 'western_casual',
}


# Specific garment compatibility (what goes well together)
GARMENT_COMPATIBILITY = {
    # Tops with Bottoms
    'tshirt': ['jeans', 'shorts', 'joggers', 'skirt', 'cargo_pants', 'chinos'],
    'shirt': ['trousers', 'chinos', 'jeans', 'skirt', 'pleated_skirt'],
    'blouse': ['trousers', 'skirt', 'pleated_skirt', 'jeans', 'palazzos'],
    'polo': ['chinos', 'jeans', 'shorts', 'trousers'],
    'kurta': ['palazzos', 'salwar', 'churidar', 'jeans', 'dhoti_pants', 'leggings'],
    'kurti': ['palazzos', 'jeans', 'leggings', 'churidar'],
    'sweater': ['jeans', 'trousers', 'skirt', 'chinos'],
    'turtleneck': ['trousers', 'jeans', 'skirt', 'pleated_skirt'],
    'hoodie': ['jeans', 'joggers', 'cargo_pants', 'shorts'],
    'crop_top': ['jeans', 'skirt', 'palazzos', 'maxi_skirt', 'shorts'],
    'tank_top': ['shorts', 'jeans', 'skirt', 'joggers'],
    
    # Outerwear with Tops
    'blazer': ['shirt', 'blouse', 'turtleneck', 'tshirt'],
    'denim_jacket': ['tshirt', 'blouse', 'tank_top', 'hoodie'],
    'leather_jacket': ['tshirt', 'turtleneck', 'hoodie'],
    'cardigan': ['shirt', 'blouse', 'tshirt', 'turtleneck'],
    'nehru_jacket': ['kurta'],
    'shrug': ['kurta', 'kurti', 'blouse'],
    
    # Ethnic specific
    'choli': ['lehenga_skirt'],
    'salwar': ['kurta', 'kurti'],
    'churidar': ['kurta', 'anarkali'],
    'palazzos': ['kurta', 'kurti', 'crop_top', 'blouse'],
    'dhoti_pants': ['kurta', 'kurti'],
}


# =============================================================================
# ENSEMBLE TEMPLATES
# =============================================================================

@dataclass
class EnsembleTemplate:
    """Template for building an outfit"""
    name: str
    required_categories: List[GarmentCategory]
    optional_categories: List[GarmentCategory]
    style_families: Set[StyleFamily]
    occasions: Set[Occasion]
    suggested_garments: Dict[GarmentCategory, List[str]]


ENSEMBLE_TEMPLATES = {
    # Casual Western
    'casual_daily': EnsembleTemplate(
        name='Casual Daily',
        required_categories=[GarmentCategory.TOP, GarmentCategory.BOTTOM],
        optional_categories=[GarmentCategory.OUTERWEAR],
        style_families={StyleFamily.CASUAL},
        occasions={Occasion.DAILY, Occasion.BRUNCH},
        suggested_garments={
            GarmentCategory.TOP: ['tshirt', 'polo', 'henley', 'sweater'],
            GarmentCategory.BOTTOM: ['jeans', 'chinos', 'shorts'],
            GarmentCategory.OUTERWEAR: ['denim_jacket', 'cardigan', 'hoodie'],
        }
    ),
    
    'smart_casual': EnsembleTemplate(
        name='Smart Casual',
        required_categories=[GarmentCategory.TOP, GarmentCategory.BOTTOM],
        optional_categories=[GarmentCategory.OUTERWEAR],
        style_families={StyleFamily.CASUAL, StyleFamily.PREPPY},
        occasions={Occasion.BRUNCH, Occasion.DATE, Occasion.WORK},
        suggested_garments={
            GarmentCategory.TOP: ['shirt', 'blouse', 'polo', 'turtleneck'],
            GarmentCategory.BOTTOM: ['chinos', 'jeans', 'skirt', 'trousers'],
            GarmentCategory.OUTERWEAR: ['blazer', 'cardigan'],
        }
    ),
    
    'formal_western': EnsembleTemplate(
        name='Formal Western',
        required_categories=[GarmentCategory.TOP, GarmentCategory.BOTTOM, GarmentCategory.OUTERWEAR],
        optional_categories=[],
        style_families={StyleFamily.FORMAL},
        occasions={Occasion.WORK, Occasion.FORMAL_EVENT},
        suggested_garments={
            GarmentCategory.TOP: ['shirt', 'blouse', 'turtleneck'],
            GarmentCategory.BOTTOM: ['trousers', 'pleated_skirt', 'skirt'],
            GarmentCategory.OUTERWEAR: ['blazer', 'vest', 'coat'],
        }
    ),
    
    'streetwear': EnsembleTemplate(
        name='Streetwear',
        required_categories=[GarmentCategory.TOP, GarmentCategory.BOTTOM],
        optional_categories=[GarmentCategory.OUTERWEAR],
        style_families={StyleFamily.STREETWEAR},
        occasions={Occasion.DAILY, Occasion.PARTY},
        suggested_garments={
            GarmentCategory.TOP: ['tshirt', 'hoodie', 'sweatshirt', 'crop_top'],
            GarmentCategory.BOTTOM: ['jeans', 'joggers', 'cargo_pants'],
            GarmentCategory.OUTERWEAR: ['denim_jacket', 'leather_jacket', 'jacket'],
        }
    ),
    
    # Ethnic Indian
    'ethnic_daily': EnsembleTemplate(
        name='Ethnic Daily',
        required_categories=[GarmentCategory.TOP, GarmentCategory.BOTTOM],
        optional_categories=[],
        style_families={StyleFamily.ETHNIC},
        occasions={Occasion.DAILY, Occasion.WORK},
        suggested_garments={
            GarmentCategory.TOP: ['kurti', 'kurta'],
            GarmentCategory.BOTTOM: ['palazzos', 'jeans', 'leggings', 'churidar'],
        }
    ),
    
    'ethnic_festive': EnsembleTemplate(
        name='Ethnic Festive',
        required_categories=[GarmentCategory.TOP, GarmentCategory.BOTTOM],
        optional_categories=[GarmentCategory.OUTERWEAR],
        style_families={StyleFamily.ETHNIC},
        occasions={Occasion.FESTIVE, Occasion.WEDDING, Occasion.PARTY},
        suggested_garments={
            GarmentCategory.TOP: ['kurta', 'choli'],
            GarmentCategory.BOTTOM: ['palazzos', 'salwar', 'churidar', 'lehenga_skirt', 'dhoti_pants'],
            GarmentCategory.OUTERWEAR: ['nehru_jacket', 'shrug'],
        }
    ),
    
    'fusion': EnsembleTemplate(
        name='Indo-Western Fusion',
        required_categories=[GarmentCategory.TOP, GarmentCategory.BOTTOM],
        optional_categories=[GarmentCategory.OUTERWEAR],
        style_families={StyleFamily.ETHNIC, StyleFamily.CASUAL},
        occasions={Occasion.DAILY, Occasion.BRUNCH, Occasion.DATE},
        suggested_garments={
            GarmentCategory.TOP: ['kurti', 'kurta', 'crop_top', 'blouse'],
            GarmentCategory.BOTTOM: ['jeans', 'palazzos', 'skirt', 'dhoti_pants'],
            GarmentCategory.OUTERWEAR: ['denim_jacket', 'shrug'],
        }
    ),
    
    # Special occasions
    'party': EnsembleTemplate(
        name='Party Outfit',
        required_categories=[GarmentCategory.TOP, GarmentCategory.BOTTOM],
        optional_categories=[GarmentCategory.OUTERWEAR],
        style_families={StyleFamily.CASUAL, StyleFamily.STREETWEAR},
        occasions={Occasion.PARTY, Occasion.DATE},
        suggested_garments={
            GarmentCategory.TOP: ['blouse', 'crop_top', 'tshirt'],
            GarmentCategory.BOTTOM: ['jeans', 'skirt', 'trousers'],
            GarmentCategory.OUTERWEAR: ['blazer', 'leather_jacket'],
        }
    ),
    
    'date_night': EnsembleTemplate(
        name='Date Night',
        required_categories=[GarmentCategory.TOP, GarmentCategory.BOTTOM],
        optional_categories=[GarmentCategory.OUTERWEAR],
        style_families={StyleFamily.CASUAL, StyleFamily.MINIMALIST},
        occasions={Occasion.DATE},
        suggested_garments={
            GarmentCategory.TOP: ['blouse', 'shirt', 'turtleneck', 'kurti'],
            GarmentCategory.BOTTOM: ['jeans', 'skirt', 'trousers', 'palazzos'],
            GarmentCategory.OUTERWEAR: ['blazer', 'cardigan', 'leather_jacket'],
        }
    ),
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_garment_type(name: str) -> Optional[GarmentType]:
    """Get garment type definition by name"""
    return ALL_GARMENTS.get(name.lower())


def identify_garment_from_text(text: str) -> Optional[GarmentType]:
    """
    Identify garment type from text description.
    
    Args:
        text: Text containing garment description
    
    Returns:
        Most likely GarmentType or None
    """
    text_lower = text.lower()
    
    best_match = None
    best_score = 0
    
    for garment_name, garment_type in ALL_GARMENTS.items():
        score = 0
        
        # Check direct name match
        if garment_name in text_lower:
            score += 10
        
        # Check keywords
        for keyword in garment_type.keywords:
            if keyword.lower() in text_lower:
                score += 5
        
        if score > best_score:
            best_score = score
            best_match = garment_type
    
    return best_match if best_score > 0 else None


def get_compatible_garments(anchor_garment: str) -> List[str]:
    """
    Get list of garment types compatible with anchor.
    
    Args:
        anchor_garment: Name of the anchor garment
    
    Returns:
        List of compatible garment names
    """
    return GARMENT_COMPATIBILITY.get(anchor_garment.lower(), [])


def get_ensemble_template(
    occasion: Optional[Occasion] = None,
    style: Optional[StyleFamily] = None
) -> Optional[EnsembleTemplate]:
    """
    Get best matching ensemble template.
    
    Args:
        occasion: Target occasion
        style: Target style family
    
    Returns:
        Best matching EnsembleTemplate
    """
    best_match = None
    best_score = 0
    
    for template_name, template in ENSEMBLE_TEMPLATES.items():
        score = 0
        
        if occasion and occasion in template.occasions:
            score += 10
        
        if style and style in template.style_families:
            score += 5
        
        if score > best_score:
            best_score = score
            best_match = template
    
    return best_match


def check_style_coherence(garment1: str, garment2: str) -> Dict:
    """
    Check if two garments have coherent styles.
    
    Returns:
        Dict with compatibility score and details
    """
    g1 = get_garment_type(garment1)
    g2 = get_garment_type(garment2)
    
    if not g1 or not g2:
        return {'score': 0, 'compatible': False, 'reason': 'Unknown garment type'}
    
    result = {
        'score': 0,
        'compatible': False,
        'shared_styles': [],
        'shared_occasions': [],
        'formality_diff': 0,
    }
    
    # Check style family overlap
    shared_styles = g1.style_families & g2.style_families
    result['shared_styles'] = [s.value for s in shared_styles]
    
    if shared_styles:
        result['score'] += 40
    
    # Check occasion overlap
    shared_occasions = g1.occasions & g2.occasions
    result['shared_occasions'] = [o.value for o in shared_occasions]
    
    if shared_occasions:
        result['score'] += 30
    
    # Check formality compatibility (within 2 levels)
    formality_diff = abs(g1.formality_score - g2.formality_score)
    result['formality_diff'] = formality_diff
    
    if formality_diff <= 1:
        result['score'] += 30
    elif formality_diff == 2:
        result['score'] += 15
    
    # Overall compatibility
    result['compatible'] = result['score'] >= 50
    
    return result


def check_category_compatibility(garment1: str, garment2: str) -> bool:
    """
    Check if two garments can be worn together based on category.
    """
    g1 = get_garment_type(garment1)
    g2 = get_garment_type(garment2)
    
    if not g1 or not g2:
        return False
    
    # Same category usually not compatible (can't wear two tops)
    if g1.category == g2.category:
        return False
    
    # Check if categories are compatible
    if g2.category in CATEGORY_COMPATIBILITY.get(g1.category, set()):
        return True
    
    return False


# =============================================================================
# ENSEMBLE BUILDER (COMPREHENSIVE)
# =============================================================================

def get_style_subcategory(style: str = None, occasion: str = None) -> str:
    """Map style/occasion to subcategory for garment selection."""
    if style:
        mapped = STYLE_MAPPING.get(style.lower())
        if mapped:
            return mapped
    if occasion:
        mapped = STYLE_MAPPING.get(occasion.lower())
        if mapped:
            return mapped
    return 'default'


def get_garments_for_category(category: str, style: str = None, occasion: str = None) -> List[str]:
    """Get appropriate garment types for a category based on style."""
    subcategory = get_style_subcategory(style, occasion)
    category_defaults = CATEGORY_GARMENT_DEFAULTS.get(category, {})
    return category_defaults.get(subcategory, category_defaults.get('default', []))


def build_ensemble_queries(
    anchor_garment: str,
    anchor_color: str,
    occasion: Optional[str] = None,
    style: Optional[str] = None,
    complementary_colors: Optional[List[str]] = None
) -> List[Dict]:
    """
    Generate search queries for building a complete ensemble.
    
    This function is category-aware and ensures all required
    outfit components are included (tops, bottoms, outerwear as needed).
    
    Args:
        anchor_garment: The main garment type (e.g., 'kurta', 'blazer')
        anchor_color: Color of the anchor garment (e.g., 'yellow')
        occasion: Optional occasion (e.g., 'festive', 'work')
        style: Optional style preference (e.g., 'ethnic', 'formal')
        complementary_colors: List of colors that complement anchor
    
    Returns:
        List of query dicts with garment type, category, color, and search text
    """
    queries = []
    
    # Get anchor garment info
    anchor_type = get_garment_type(anchor_garment)
    
    # Determine anchor category
    if anchor_type:
        anchor_category = anchor_type.category.value
    else:
        # Try to infer from common mappings
        if anchor_garment.lower() in ['blazer', 'jacket', 'coat', 'cardigan', 'hoodie', 'nehru_jacket', 'shrug']:
            anchor_category = 'outerwear'
        elif anchor_garment.lower() in ['jeans', 'pants', 'trousers', 'skirt', 'palazzos', 'shorts']:
            anchor_category = 'bottom'
        elif anchor_garment.lower() in ['dress', 'gown', 'saree', 'lehenga', 'jumpsuit']:
            anchor_category = 'full_body'
        else:
            anchor_category = 'top'
    
    # Get outfit requirements for this anchor category
    requirements = OUTFIT_REQUIREMENTS.get(anchor_category, {'required': ['bottom'], 'optional': []})
    needed_categories = requirements['required'] + requirements['optional']
    
    # Use complementary colors if provided, else use neutrals
    if not complementary_colors:
        complementary_colors = ['white', 'black', 'grey', 'beige', 'navy']
    
    # Generate queries for each needed category
    color_index = 0
    
    for category in needed_categories:
        # Get appropriate garments for this category and style
        garment_options = get_garments_for_category(category, style, occasion)
        
        # Create 2 queries per category (variety)
        for i, garment_name in enumerate(garment_options[:2]):
            # Cycle through complementary colors
            color = complementary_colors[color_index % len(complementary_colors)]
            color_index += 1
            
            # Build search query text
            query_parts = [color, garment_name]
            
            # Add occasion if provided
            if occasion:
                query_parts.append(occasion)
            
            # Add style hint for better matching
            if style and style.lower() in ['ethnic', 'festive', 'wedding']:
                query_parts.append('ethnic')
            elif style and style.lower() in ['formal', 'work']:
                query_parts.append('formal')
            
            queries.append({
                'garment_type': garment_name,
                'garment_category': category,
                'color': color,
                'search_text': ' '.join(query_parts),
            })
    
    return queries


def build_ensemble_queries_balanced(
    anchor_garment: str,
    anchor_color: str,
    occasion: Optional[str] = None,
    style: Optional[str] = None,
    n_safe: int = 3,
    n_bold: int = 1
) -> List[Dict]:
    """
    Generate balanced search queries: safe neutrals + bold statement pieces.
    
    Follows fashion principles:
    - Bottoms: Primarily neutrals (60% rule)
    - Tops: Mix of neutrals and pops (30% rule)
    - Bold options: Color-theory based for fashionable risk-takers
    
    Args:
        anchor_garment: The main garment type
        anchor_color: Color of the anchor garment
        occasion: Optional occasion
        style: Optional style preference
        n_safe: Number of safe/neutral queries per category
        n_bold: Number of bold/statement queries per category
    
    Returns:
        List of query dicts with is_bold flag
    """
    queries = []
    
    # Get anchor garment info
    anchor_type = get_garment_type(anchor_garment)
    
    # Determine anchor category
    if anchor_type:
        anchor_category = anchor_type.category.value
    else:
        if anchor_garment.lower() in ['blazer', 'jacket', 'coat', 'cardigan', 'hoodie', 'nehru_jacket', 'shrug']:
            anchor_category = 'outerwear'
        elif anchor_garment.lower() in ['jeans', 'pants', 'trousers', 'skirt', 'palazzos', 'shorts']:
            anchor_category = 'bottom'
        elif anchor_garment.lower() in ['dress', 'gown', 'saree', 'lehenga', 'jumpsuit']:
            anchor_category = 'full_body'
        else:
            anchor_category = 'top'
    
    # Get outfit requirements
    requirements = OUTFIT_REQUIREMENTS.get(anchor_category, {'required': ['bottom'], 'optional': []})
    needed_categories = requirements['required'] + requirements['optional']
    
    # Import color functions (will be available when modules are loaded together)
    from color_theory import get_safe_colors_for_category, get_bold_complementary
    
    # Determine style for color selection
    style_for_colors = style or occasion or 'default'
    
    for category in needed_categories:
        # Get garment options for this category
        garment_options = get_garments_for_category(category, style, occasion)
        
        # Get safe colors for this category
        safe_colors = get_safe_colors_for_category(category, style, occasion)
        
        # Get bold colors based on anchor
        bold_colors = get_bold_complementary(anchor_color, n_bold + 1)
        
        # Generate SAFE queries (3 by default)
        for i in range(min(n_safe, len(safe_colors))):
            color = safe_colors[i]
            garment = garment_options[i % len(garment_options)]
            
            query_parts = [color, garment]
            if occasion:
                query_parts.append(occasion)
            if style and style.lower() in ['formal', 'work', 'office']:
                query_parts.append('formal')
            elif style and style.lower() in ['ethnic', 'festive']:
                query_parts.append('ethnic')
            
            queries.append({
                'garment_type': garment,
                'garment_category': category,
                'color': color,
                'search_text': ' '.join(query_parts),
                'is_bold': False,
                'style_note': 'safe_neutral'
            })
        
        # Generate BOLD queries (1 by default)
        for i in range(n_bold):
            bold_color = bold_colors[i % len(bold_colors)]
            # Use a different garment for variety
            garment_idx = (n_safe + i) % len(garment_options)
            garment = garment_options[garment_idx]
            
            query_parts = [bold_color, garment, 'statement']
            if occasion:
                query_parts.append(occasion)
            
            queries.append({
                'garment_type': garment,
                'garment_category': category,
                'color': bold_color,
                'search_text': ' '.join(query_parts),
                'is_bold': True,
                'style_note': 'bold_statement'
            })
    
    return queries


def build_ensemble_queries_extended(
    anchor_garment: str,
    anchor_color: str,
    occasion: Optional[str] = None,
    style: Optional[str] = None,
    complementary_colors: Optional[List[str]] = None,
    items_per_category: int = 4
) -> Dict[str, List[Dict]]:
    """
    Generate multiple search queries per category for richer results.
    
    Returns queries grouped by category for easier processing.
    
    Args:
        anchor_garment: The main garment type
        anchor_color: Color of the anchor garment
        occasion: Optional occasion
        style: Optional style preference
        complementary_colors: List of complementary colors
        items_per_category: Number of different queries per category
    
    Returns:
        Dict mapping category to list of query dicts
    """
    # Get anchor garment info
    anchor_type = get_garment_type(anchor_garment)
    
    # Determine anchor category
    if anchor_type:
        anchor_category = anchor_type.category.value
    else:
        if anchor_garment.lower() in ['blazer', 'jacket', 'coat', 'cardigan', 'hoodie', 'nehru_jacket', 'shrug']:
            anchor_category = 'outerwear'
        elif anchor_garment.lower() in ['jeans', 'pants', 'trousers', 'skirt', 'palazzos', 'shorts']:
            anchor_category = 'bottom'
        elif anchor_garment.lower() in ['dress', 'gown', 'saree', 'lehenga', 'jumpsuit']:
            anchor_category = 'full_body'
        else:
            anchor_category = 'top'
    
    # Get outfit requirements
    requirements = OUTFIT_REQUIREMENTS.get(anchor_category, {'required': ['bottom'], 'optional': []})
    needed_categories = requirements['required'] + requirements['optional']
    
    # Use complementary colors if provided
    if not complementary_colors:
        complementary_colors = ['white', 'black', 'grey', 'beige', 'navy', 'cream']
    
    # Generate queries grouped by category
    queries_by_category = {}
    
    for category in needed_categories:
        queries_by_category[category] = []
        
        # Get appropriate garments for this category and style
        garment_options = get_garments_for_category(category, style, occasion)
        
        # Create multiple queries for variety
        for i in range(min(items_per_category, len(garment_options) * len(complementary_colors))):
            garment_name = garment_options[i % len(garment_options)]
            color = complementary_colors[i % len(complementary_colors)]
            
            # Build search query text
            query_parts = [color, garment_name]
            
            if occasion:
                query_parts.append(occasion)
            
            queries_by_category[category].append({
                'garment_type': garment_name,
                'garment_category': category,
                'color': color,
                'search_text': ' '.join(query_parts),
            })
    
    return queries_by_category


def suggest_outfit_structure(
    input_text: str,
    has_anchor_image: bool = False
) -> Dict:
    """
    Analyze input and suggest outfit structure.
    
    Args:
        input_text: User's text input
        has_anchor_image: Whether user provided an image
    
    Returns:
        Dict with detected intent and suggested structure
    """
    text_lower = input_text.lower()
    
    result = {
        'intent': 'similar',  # Default: find similar items
        'needs_full_ensemble': False,
        'detected_occasion': None,
        'detected_style': None,
        'detected_garment': None,
        'suggested_categories': [],
    }
    
    # Detect intent keywords
    ensemble_keywords = ['outfit', 'ensemble', 'complete', 'full look', 'match', 'pair with', 'goes with', 'coordinate']
    for keyword in ensemble_keywords:
        if keyword in text_lower:
            result['intent'] = 'ensemble'
            result['needs_full_ensemble'] = True
            break
    
    # Detect occasion
    for occasion in Occasion:
        if occasion.value in text_lower:
            result['detected_occasion'] = occasion.value
            result['needs_full_ensemble'] = True
            break
    
    # Detect style
    for style in StyleFamily:
        if style.value in text_lower:
            result['detected_style'] = style.value
            break
    
    # Detect garment type
    detected_garment = identify_garment_from_text(text_lower)
    if detected_garment:
        result['detected_garment'] = detected_garment.name
        
        # Suggest categories based on garment
        if detected_garment.category == GarmentCategory.TOP:
            result['suggested_categories'] = ['bottom', 'outerwear']
        elif detected_garment.category == GarmentCategory.BOTTOM:
            result['suggested_categories'] = ['top', 'outerwear']
        elif detected_garment.category == GarmentCategory.FULL_BODY:
            result['suggested_categories'] = ['outerwear']
    
    return result


# =============================================================================
# DIVERSE AND CONTEXT-AWARE QUERY BUILDERS
# =============================================================================

# More diverse safe palettes (avoids too-similar colors)
SAFE_COLORS_DIVERSE = {
    'bottom': {
        'formal': ['black', 'navy', 'camel', 'burgundy'],
        'casual': ['black', 'olive', 'khaki', 'brown'],
        'ethnic': ['white', 'beige', 'maroon', 'navy'],
        'default': ['black', 'navy', 'camel', 'olive']
    },
    'top': {
        'formal': ['white', 'light blue', 'blush', 'sage'],
        'casual': ['white', 'olive', 'burgundy', 'denim'],
        'ethnic': ['white', 'gold', 'pink', 'mint'],
        'default': ['white', 'light blue', 'blush', 'olive']
    },
    'outerwear': {
        'formal': ['black', 'camel', 'burgundy', 'grey'],
        'casual': ['olive', 'tan', 'navy', 'brown'],
        'ethnic': ['black', 'maroon', 'gold', 'cream'],
        'default': ['black', 'camel', 'navy', 'olive']
    }
}

# Garment variety per category (different silhouettes)
GARMENT_VARIETY = {
    'top': {
        'formal': ['shirt', 'blouse', 'turtleneck', 'silk top'],
        'casual': ['tshirt', 'henley', 'polo', 'sweater'],
    },
    'bottom': {
        'formal': ['trousers', 'pencil skirt', 'wide leg pants', 'A-line skirt'],
        'casual': ['jeans', 'chinos', 'joggers', 'shorts'],
    }
}

# Context-aware modifiers per occasion
OCCASION_MODIFIERS = {
    'office': ['formal', 'professional', 'workwear'],
    'work': ['formal', 'professional', 'office'],
    'formal': ['elegant', 'sophisticated', 'formal'],
    'party': ['stylish', 'evening', 'statement'],
    'date': ['elegant', 'chic', 'flattering'],
    'casual': ['relaxed', 'comfortable', 'everyday'],
    'brunch': ['smart casual', 'relaxed', 'daytime'],
    'festive': ['traditional', 'embroidered', 'ethnic'],
    'wedding': ['elegant', 'traditional', 'festive'],
}

# Garments to AVOID per occasion
OCCASION_EXCLUSIONS = {
    'office': ['cropped', 'crop top', 'sleeveless', 'backless'],
    'work': ['cropped', 'crop top', 'sleeveless', 'party'],
    'formal': ['casual', 'sporty', 'cropped'],
}

# Better garment descriptors per occasion
GARMENT_DESCRIPTORS = {
    'top': {
        'office': ['formal shirt', 'blouse', 'turtleneck', 'silk blouse'],
        'work': ['button-down shirt', 'formal blouse', 'mock neck top', 'tailored top'],
        'casual': ['tshirt', 'henley', 'casual top', 'sweater'],
        'party': ['silk top', 'statement blouse', 'elegant top', 'dressy top'],
        'festive': ['embroidered top', 'silk kurta', 'traditional blouse', 'festive top'],
        'date': ['elegant blouse', 'chic top', 'stylish shirt', 'flattering top'],
    },
    'bottom': {
        'office': ['formal trousers', 'pencil skirt', 'tailored pants', 'midi skirt'],
        'work': ['dress pants', 'formal skirt', 'wide leg trousers', 'A-line skirt'],
        'casual': ['jeans', 'chinos', 'casual pants', 'shorts'],
        'party': ['statement skirt', 'elegant pants', 'dressy trousers', 'midi skirt'],
        'festive': ['palazzos', 'ethnic skirt', 'traditional pants', 'lehenga skirt'],
        'date': ['elegant pants', 'chic skirt', 'flattering trousers', 'midi skirt'],
    }
}


def build_diverse_queries(
    anchor_garment: str,
    anchor_color: str,
    occasion: Optional[str] = None,
    style: Optional[str] = None,
    n_safe: int = 3,
    n_bold: int = 1
) -> List[Dict]:
    """
    Build queries with diverse colors AND silhouettes.
    
    Ensures variety by pairing different colors with different garment types.
    """
    from color_theory import get_bold_complementary
    
    queries = []
    
    # Determine anchor category
    anchor_type = get_garment_type(anchor_garment)
    if anchor_type:
        anchor_category = anchor_type.category.value
    else:
        if anchor_garment.lower() in ['blazer', 'jacket', 'coat', 'cardigan']:
            anchor_category = 'outerwear'
        elif anchor_garment.lower() in ['jeans', 'pants', 'trousers', 'skirt']:
            anchor_category = 'bottom'
        else:
            anchor_category = 'top'
    
    # Get needed categories
    requirements = OUTFIT_REQUIREMENTS.get(anchor_category, {'required': ['bottom'], 'optional': []})
    needed_categories = requirements['required'] + requirements['optional']
    
    # Style key
    style_key = 'formal' if occasion in ['office', 'work', 'formal'] else 'casual'
    if style and style.lower() in ['ethnic', 'festive']:
        style_key = 'ethnic'
    
    for category in needed_categories:
        # Get diverse colors
        safe_colors = SAFE_COLORS_DIVERSE.get(category, {}).get(style_key, ['white', 'black', 'grey', 'navy'])
        
        # Get diverse garments
        garment_options = GARMENT_VARIETY.get(category, {}).get(style_key, ['item'])
        
        # SAFE queries: pair each color with different garment
        for i in range(min(n_safe, len(safe_colors))):
            color = safe_colors[i]
            garment = garment_options[i % len(garment_options)]
            
            query_parts = [color, garment]
            if occasion:
                query_parts.append(occasion)
            
            queries.append({
                'garment_type': garment,
                'garment_category': category,
                'color': color,
                'search_text': ' '.join(query_parts),
                'is_bold': False
            })
        
        # BOLD query
        bold_colors = get_bold_complementary(anchor_color, 2)
        bold_garment = garment_options[-1] if garment_options else 'item'
        
        queries.append({
            'garment_type': bold_garment,
            'garment_category': category,
            'color': bold_colors[0],
            'search_text': f"{bold_colors[0]} {bold_garment} statement",
            'is_bold': True
        })
    
    return queries


# Accessory queries by occasion
ACCESSORY_QUERIES = {
    'office': [
        ('black', 'leather belt', 'formal accessory'),
        ('silver', 'formal watch', 'accessory'),
        ('black', 'leather laptop bag', 'professional'),
        ('black', 'oxford shoes', 'formal footwear'),
    ],
    'work': [
        ('brown', 'leather belt', 'formal accessory'),
        ('gold', 'watch', 'accessory'),
        ('tan', 'tote bag', 'professional'),
        ('nude', 'heels', 'formal footwear'),
    ],
    'party': [
        ('gold', 'statement earrings', 'accessory'),
        ('black', 'clutch bag', 'evening'),
        ('black', 'heels', 'party footwear'),
        ('silver', 'bracelet', 'jewelry'),
    ],
    'casual': [
        ('brown', 'canvas belt', 'casual accessory'),
        ('white', 'sneakers', 'casual footwear'),
        ('tan', 'crossbody bag', 'casual'),
    ],
    'festive': [
        ('gold', 'statement jewelry', 'ethnic accessory'),
        ('gold', 'embroidered clutch', 'festive'),
        ('gold', 'juttis', 'ethnic footwear'),
    ],
    'date': [
        ('gold', 'delicate jewelry', 'accessory'),
        ('black', 'clutch', 'evening'),
        ('nude', 'heels', 'elegant footwear'),
    ],
}


def build_context_aware_queries(
    anchor_garment: str,
    anchor_color: str,
    occasion: Optional[str] = None,
    style: Optional[str] = None,
    n_safe: int = 3,
    n_bold: int = 1
) -> List[Dict]:
    """
    Build queries with context-aware fit and formality modifiers.
    
    Ensures office queries get professional items, party queries get stylish items, etc.
    """
    from color_theory import get_bold_complementary
    
    queries = []
    
    # Normalize occasion
    occasion_key = (occasion or 'casual').lower()
    if occasion_key not in OCCASION_MODIFIERS:
        occasion_key = 'casual'
    
    # Get modifiers for this occasion
    modifiers = OCCASION_MODIFIERS.get(occasion_key, [])
    primary_modifier = modifiers[0] if modifiers else ''
    
    # Determine anchor category
    anchor_type = get_garment_type(anchor_garment)
    if anchor_type:
        anchor_category = anchor_type.category.value
    else:
        if anchor_garment.lower() in ['blazer', 'jacket', 'coat', 'cardigan']:
            anchor_category = 'outerwear'
        elif anchor_garment.lower() in ['jeans', 'pants', 'trousers', 'skirt']:
            anchor_category = 'bottom'
        else:
            anchor_category = 'top'
    
    # Get needed categories
    requirements = OUTFIT_REQUIREMENTS.get(anchor_category, {'required': ['bottom'], 'optional': []})
    needed_categories = requirements['required'] + requirements['optional']
    
    # Style key for colors
    style_key = 'formal' if occasion_key in ['office', 'work', 'formal'] else 'casual'
    
    for category in needed_categories:
        # Get context-appropriate garment descriptors
        garment_options = GARMENT_DESCRIPTORS.get(category, {}).get(occasion_key, ['item'])
        if not garment_options:
            garment_options = GARMENT_DESCRIPTORS.get(category, {}).get('casual', ['item'])
        
        # Get category-specific colors
        cat_colors = SAFE_COLORS_DIVERSE.get(category, {}).get(style_key, ['white', 'black', 'grey', 'navy'])
        
        # SAFE queries with context modifiers
        for i in range(min(n_safe, len(cat_colors))):
            color = cat_colors[i]
            garment = garment_options[i % len(garment_options)]
            
            # Build query with modifier
            query_parts = [color, garment]
            if primary_modifier:
                query_parts.append(primary_modifier)
            
            queries.append({
                'garment_type': garment,
                'garment_category': category,
                'color': color,
                'search_text': ' '.join(query_parts),
                'is_bold': False,
                'occasion': occasion_key
            })
        
        # BOLD query (still respects occasion)
        bold_colors = get_bold_complementary(anchor_color, 2)
        bold_garment = garment_options[-1] if garment_options else 'statement piece'
        
        # For office, even bold should be professional
        if occasion_key in ['office', 'work']:
            bold_modifier = 'elegant statement'
        else:
            bold_modifier = 'statement'
        
        queries.append({
            'garment_type': bold_garment,
            'garment_category': category,
            'color': bold_colors[0],
            'search_text': f"{bold_colors[0]} {bold_garment} {bold_modifier}",
            'is_bold': True,
            'occasion': occasion_key
        })
    
    return queries


def build_context_aware_queries_v2(
    anchor_garment: str,
    anchor_color: str,
    occasion: Optional[str] = None,
    style: Optional[str] = None,
    n_safe: int = 3,
    n_bold: int = 1,
    include_accessories: bool = False
) -> List[Dict]:
    """
    Build queries with context-aware modifiers + 'full view' for complete garment images.
    
    This is the production version that ensures:
    - Full garment images (not cropped/partial)
    - Context-appropriate formality
    - Optional accessories
    """
    from color_theory import get_bold_complementary
    
    queries = []
    
    # Normalize occasion
    occasion_key = (occasion or 'casual').lower()
    if occasion_key not in OCCASION_MODIFIERS:
        occasion_key = 'casual'
    
    # Get modifiers
    modifiers = OCCASION_MODIFIERS.get(occasion_key, [])
    primary_modifier = modifiers[0] if modifiers else ''
    
    # Determine anchor category
    anchor_type = get_garment_type(anchor_garment)
    if anchor_type:
        anchor_category = anchor_type.category.value
    else:
        if anchor_garment.lower() in ['blazer', 'jacket', 'coat', 'cardigan']:
            anchor_category = 'outerwear'
        elif anchor_garment.lower() in ['jeans', 'pants', 'trousers', 'skirt']:
            anchor_category = 'bottom'
        else:
            anchor_category = 'top'
    
    # Get needed categories
    requirements = OUTFIT_REQUIREMENTS.get(anchor_category, {'required': ['bottom'], 'optional': []})
    needed_categories = requirements['required'] + requirements['optional']
    
    # Style key for colors
    style_key = 'formal' if occasion_key in ['office', 'work', 'formal'] else 'casual'
    
    for category in needed_categories:
        # Get context-appropriate garments
        garment_options = GARMENT_DESCRIPTORS.get(category, {}).get(occasion_key, ['item'])
        if not garment_options:
            garment_options = GARMENT_DESCRIPTORS.get(category, {}).get('casual', ['item'])
        
        # Get colors
        cat_colors = SAFE_COLORS_DIVERSE.get(category, {}).get(style_key, ['white', 'black', 'grey', 'navy'])
        
        # SAFE queries with "full view" modifier
        for i in range(min(n_safe, len(cat_colors))):
            color = cat_colors[i]
            garment = garment_options[i % len(garment_options)]
            
            # Build query with "full view" modifier
            query_parts = [color, garment, primary_modifier, 'full view']
            query_text = ' '.join([p for p in query_parts if p]).strip()
            
            queries.append({
                'garment_type': garment,
                'garment_category': category,
                'color': color,
                'search_text': query_text,
                'is_bold': False,
                'occasion': occasion_key
            })
        
        # BOLD query
        bold_colors = get_bold_complementary(anchor_color, 2)
        bold_garment = garment_options[-1] if garment_options else 'statement piece'
        
        if occasion_key in ['office', 'work']:
            bold_modifier = 'elegant statement full view'
        else:
            bold_modifier = 'statement full view'
        
        queries.append({
            'garment_type': bold_garment,
            'garment_category': category,
            'color': bold_colors[0],
            'search_text': f"{bold_colors[0]} {bold_garment} {bold_modifier}",
            'is_bold': True,
            'occasion': occasion_key
        })
    
    # ACCESSORIES (optional)
    if include_accessories:
        acc_list = ACCESSORY_QUERIES.get(occasion_key, ACCESSORY_QUERIES.get('casual', []))
        for color, item, modifier in acc_list:
            queries.append({
                'garment_type': item,
                'garment_category': 'accessory',
                'color': color,
                'search_text': f"{color} {item} {modifier} full view",
                'is_bold': False,
                'occasion': occasion_key
            })
    
    return queries


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'GarmentCategory',
    'StyleFamily',
    'Occasion',
    
    # Data classes
    'GarmentType',
    'EnsembleTemplate',
    
    # Registries
    'TOPS',
    'BOTTOMS',
    'OUTERWEAR',
    'FULL_BODY',
    'ALL_GARMENTS',
    'ENSEMBLE_TEMPLATES',
    'GARMENT_COMPATIBILITY',
    'OUTFIT_REQUIREMENTS',
    'CATEGORY_GARMENT_DEFAULTS',
    'STYLE_MAPPING',
    
    # Diverse/Context-aware constants
    'SAFE_COLORS_DIVERSE',
    'GARMENT_VARIETY',
    'OCCASION_MODIFIERS',
    'OCCASION_EXCLUSIONS',
    'GARMENT_DESCRIPTORS',
    'ACCESSORY_QUERIES',
    
    # Functions
    'get_garment_type',
    'identify_garment_from_text',
    'get_compatible_garments',
    'get_ensemble_template',
    'check_style_coherence',
    'check_category_compatibility',
    'get_style_subcategory',
    'get_garments_for_category',
    'build_ensemble_queries',
    'build_ensemble_queries_balanced',
    'build_ensemble_queries_extended',
    'build_diverse_queries',
    'build_context_aware_queries',
    'build_context_aware_queries_v2',
    'suggest_outfit_structure',
]