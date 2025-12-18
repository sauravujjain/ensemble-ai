"""
Garment Normalizer Module for Fashion Ensemble AI

Handles:
1. Garment name normalization (spelling variations, regional terms)
2. Category detection with fuzzy matching
3. British/American/Indian terminology
4. Compound term parsing

Author: Fashion Ensemble Builder
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re


# =============================================================================
# GARMENT CATEGORIES
# =============================================================================

class GarmentCategory(Enum):
    TOP = "top"
    BOTTOM = "bottom"
    OUTERWEAR = "outerwear"
    FULL_BODY = "full_body"
    ACCESSORY = "accessory"
    FOOTWEAR = "footwear"
    UNKNOWN = "unknown"


# =============================================================================
# COMPREHENSIVE GARMENT NORMALIZATION MAP
# =============================================================================

# Maps variations -> canonical form
# Organized by category for clarity

GARMENT_NORMALIZATION = {
    # =========================================================================
    # TOPS - T-SHIRTS & CASUAL
    # =========================================================================
    # T-shirt variations (VERY common issue)
    'tshirt': 't-shirt',
    't shirt': 't-shirt',
    'tee': 't-shirt',
    'tee shirt': 't-shirt',
    'teeshirt': 't-shirt',
    't-shirts': 't-shirt',
    'tshirts': 't-shirt',
    'graphic tee': 't-shirt',
    'plain tee': 't-shirt',
    'crew neck tee': 't-shirt',
    'v neck tee': 't-shirt',
    'lounge tshirt': 't-shirt',
    'lounge t-shirt': 't-shirt',
    'lounge tee': 't-shirt',
    'casual tshirt': 't-shirt',
    'casual tee': 't-shirt',
    'basic tee': 't-shirt',
    'basic tshirt': 't-shirt',
    'cotton tee': 't-shirt',
    'fitted tee': 't-shirt',
    'relaxed tee': 't-shirt',
    'oversized tee': 't-shirt',
    'longline tee': 't-shirt',
    'pocket tee': 't-shirt',
    'printed tee': 't-shirt',
    'striped tee': 't-shirt',
    
    # Tank tops
    'tank': 'tank_top',
    'tank top': 'tank_top',
    'tanktop': 'tank_top',
    'sleeveless top': 'tank_top',
    'vest top': 'tank_top',
    'camisole': 'tank_top',
    'cami': 'tank_top',
    'singlet': 'tank_top',
    'muscle tee': 'tank_top',
    'muscle tank': 'tank_top',
    'racerback': 'tank_top',
    'racerback tank': 'tank_top',
    
    # Polo shirts
    'polo': 'polo',
    'polo shirt': 'polo',
    'golf shirt': 'polo',
    'collar tee': 'polo',
    'collared tee': 'polo',
    'pique polo': 'polo',
    
    # Henley
    'henley': 'henley',
    'henley shirt': 'henley',
    'button placket': 'henley',
    
    # Hoodies
    'hoodie': 'hoodie',
    'hoody': 'hoodie',
    'hooded sweatshirt': 'hoodie',
    'pullover hoodie': 'hoodie',
    'zip hoodie': 'hoodie',
    'zip-up hoodie': 'hoodie',
    'zipup hoodie': 'hoodie',
    
    # Sweatshirts
    'sweatshirt': 'sweatshirt',
    'sweat shirt': 'sweatshirt',
    'crewneck': 'sweatshirt',
    'crew neck sweatshirt': 'sweatshirt',
    'pullover': 'sweatshirt',  # Could also be sweater
    'fleece': 'sweatshirt',
    
    # Crop tops
    'crop top': 'crop_top',
    'croptop': 'crop_top',
    'cropped top': 'crop_top',
    'cropped tee': 'crop_top',
    'midriff top': 'crop_top',
    'belly top': 'crop_top',
    
    # =========================================================================
    # TOPS - FORMAL/SMART
    # =========================================================================
    # Shirts
    'shirt': 'shirt',
    'dress shirt': 'shirt',
    'formal shirt': 'shirt',
    'button down': 'shirt',
    'button-down': 'shirt',
    'button up': 'shirt',
    'button-up': 'shirt',
    'oxford': 'shirt',
    'oxford shirt': 'shirt',
    'chambray': 'shirt',
    'chambray shirt': 'shirt',
    'linen shirt': 'shirt',
    'cotton shirt': 'shirt',
    'casual shirt': 'shirt',
    'work shirt': 'shirt',
    'office shirt': 'shirt',
    'poplin shirt': 'shirt',
    
    # Blouse
    'blouse': 'blouse',
    'formal top': 'blouse',
    'silk top': 'blouse',
    'silk blouse': 'blouse',
    'chiffon top': 'blouse',
    'chiffon blouse': 'blouse',
    'dressy top': 'blouse',
    'elegant top': 'blouse',
    'office blouse': 'blouse',
    'work blouse': 'blouse',
    'peplum top': 'blouse',
    'wrap top': 'blouse',
    'pussy bow blouse': 'blouse',
    
    # Tunic
    'tunic': 'tunic',
    'long top': 'tunic',
    'tunic top': 'tunic',
    'tunic blouse': 'tunic',
    
    # =========================================================================
    # TOPS - KNITWEAR
    # =========================================================================
    # Sweater / Jumper (UK)
    'sweater': 'sweater',
    'jumper': 'sweater',  # British term
    'pullover sweater': 'sweater',
    'knit': 'sweater',
    'knit sweater': 'sweater',
    'knitwear': 'sweater',
    'woolen': 'sweater',
    'wool sweater': 'sweater',
    'cable knit': 'sweater',
    'v-neck sweater': 'sweater',
    'crew neck sweater': 'sweater',
    
    # Cardigan
    'cardigan': 'cardigan',
    'cardie': 'cardigan',  # British informal
    'knit cardigan': 'cardigan',
    'button cardigan': 'cardigan',
    'open front cardigan': 'cardigan',
    'longline cardigan': 'cardigan',
    
    # Turtleneck
    'turtleneck': 'turtleneck',
    'turtle neck': 'turtleneck',
    'roll neck': 'turtleneck',  # British term
    'mock neck': 'turtleneck',
    'high neck': 'turtleneck',
    'polo neck': 'turtleneck',  # British term
    
    # =========================================================================
    # TOPS - INDIAN/ETHNIC
    # =========================================================================
    # Kurta
    'kurta': 'kurta',
    'kurtaa': 'kurta',
    'ethnic top': 'kurta',
    'indian top': 'kurta',
    'long kurta': 'kurta',
    'short kurta': 'kurti',
    'cotton kurta': 'kurta',
    'silk kurta': 'kurta',
    'embroidered kurta': 'kurta',
    'printed kurta': 'kurta',
    'mens kurta': 'kurta',
    "men's kurta": 'kurta',
    'womens kurta': 'kurta',
    "women's kurta": 'kurta',
    
    # Kurti
    'kurti': 'kurti',
    'kurtis': 'kurti',
    'kurtee': 'kurti',
    'short kurti': 'kurti',
    'straight kurti': 'kurti',
    'anarkali kurti': 'kurti',
    'a-line kurti': 'kurti',
    
    # Choli (blouse for lehenga/saree)
    'choli': 'choli',
    'saree blouse': 'choli',
    'lehenga blouse': 'choli',
    'blouse for lehenga': 'choli',
    'blouse for saree': 'choli',
    
    # Kameez
    'kameez': 'kameez',
    'qameez': 'kameez',
    
    # =========================================================================
    # BOTTOMS - CASUAL
    # =========================================================================
    # Jeans
    'jeans': 'jeans',
    'jean': 'jeans',
    'denim': 'jeans',
    'denim jeans': 'jeans',
    'denims': 'jeans',
    'skinny jeans': 'jeans',
    'slim jeans': 'jeans',
    'straight jeans': 'jeans',
    'straight leg jeans': 'jeans',
    'bootcut jeans': 'jeans',
    'wide leg jeans': 'jeans',
    'mom jeans': 'jeans',
    'boyfriend jeans': 'jeans',
    'ripped jeans': 'jeans',
    'distressed jeans': 'jeans',
    'high waisted jeans': 'jeans',
    'high rise jeans': 'jeans',
    'low rise jeans': 'jeans',
    'cropped jeans': 'jeans',
    'ankle jeans': 'jeans',
    'flared jeans': 'jeans',
    
    # Shorts
    'shorts': 'shorts',
    'short': 'shorts',
    'denim shorts': 'shorts',
    'jean shorts': 'shorts',
    'bermuda': 'shorts',
    'bermuda shorts': 'shorts',
    'bermudas': 'shorts',
    'chino shorts': 'shorts',
    'cargo shorts': 'shorts',
    'athletic shorts': 'shorts',
    'running shorts': 'shorts',
    'gym shorts': 'shorts',
    'board shorts': 'shorts',
    'swim shorts': 'shorts',
    
    # Joggers / Sweatpants
    'joggers': 'joggers',
    'jogger': 'joggers',
    'jogging bottoms': 'joggers',  # British
    'jogging pants': 'joggers',
    'sweatpants': 'joggers',
    'sweat pants': 'joggers',
    'track pants': 'joggers',
    'tracksuit bottoms': 'joggers',  # British
    'trackie bottoms': 'joggers',  # British informal
    
    # Leggings
    'leggings': 'leggings',
    'legging': 'leggings',
    'tights': 'leggings',
    'yoga pants': 'leggings',
    'yoga leggings': 'leggings',
    'workout leggings': 'leggings',
    'gym leggings': 'leggings',
    'jeggings': 'leggings',
    
    # Cargo pants
    'cargo pants': 'cargo_pants',
    'cargo': 'cargo_pants',
    'cargos': 'cargo_pants',
    'utility pants': 'cargo_pants',
    
    # =========================================================================
    # BOTTOMS - FORMAL/SMART
    # =========================================================================
    # Trousers / Pants
    'trousers': 'trousers',
    'trouser': 'trousers',
    'pants': 'trousers',  # American term
    'pant': 'trousers',
    'dress pants': 'trousers',
    'formal pants': 'trousers',
    'formal trousers': 'trousers',
    'slacks': 'trousers',
    'dress trousers': 'trousers',
    'suit pants': 'trousers',
    'suit trousers': 'trousers',
    'tailored trousers': 'trousers',
    'tailored pants': 'trousers',
    'work pants': 'trousers',
    'work trousers': 'trousers',
    'office pants': 'trousers',
    'office trousers': 'trousers',
    'cigarette pants': 'trousers',
    'ankle pants': 'trousers',
    'cropped trousers': 'trousers',
    'wide leg trousers': 'trousers',
    'straight leg trousers': 'trousers',
    
    # Chinos
    'chinos': 'chinos',
    'chino': 'chinos',
    'chino pants': 'chinos',
    'khakis': 'chinos',
    'khaki': 'chinos',
    'khaki pants': 'chinos',
    'cotton pants': 'chinos',
    'cotton trousers': 'chinos',
    
    # =========================================================================
    # BOTTOMS - SKIRTS
    # =========================================================================
    'skirt': 'skirt',
    'a-line skirt': 'skirt',
    'a line skirt': 'skirt',
    'pencil skirt': 'skirt',
    'midi skirt': 'skirt',
    'mini skirt': 'skirt',
    'miniskirt': 'skirt',
    'tube skirt': 'skirt',
    'wrap skirt': 'skirt',
    'flared skirt': 'skirt',
    'circle skirt': 'skirt',
    'denim skirt': 'skirt',
    'jean skirt': 'skirt',
    
    'maxi skirt': 'maxi_skirt',
    'long skirt': 'maxi_skirt',
    'floor length skirt': 'maxi_skirt',
    
    'pleated skirt': 'pleated_skirt',
    'accordion skirt': 'pleated_skirt',
    
    # =========================================================================
    # BOTTOMS - INDIAN/ETHNIC
    # =========================================================================
    # Palazzos
    'palazzos': 'palazzos',
    'palazzo': 'palazzos',
    'palazzo pants': 'palazzos',
    'wide leg pants': 'palazzos',
    'flared pants': 'palazzos',
    
    # Salwar
    'salwar': 'salwar',
    'shalwar': 'salwar',
    'salwaar': 'salwar',
    'patiala': 'salwar',
    'patiala salwar': 'salwar',
    'patiyala': 'salwar',
    
    # Churidar
    'churidar': 'churidar',
    'churidaar': 'churidar',
    'churidars': 'churidar',
    'fitted bottom': 'churidar',
    
    # Dhoti pants
    'dhoti pants': 'dhoti_pants',
    'dhoti': 'dhoti_pants',
    'draped pants': 'dhoti_pants',
    
    # Lehenga skirt
    'lehenga': 'lehenga_skirt',
    'lehanga': 'lehenga_skirt',
    'ghagra': 'lehenga_skirt',
    'ghaghra': 'lehenga_skirt',
    'chaniya': 'lehenga_skirt',
    
    # =========================================================================
    # OUTERWEAR
    # =========================================================================
    # Blazer
    'blazer': 'blazer',
    'sport coat': 'blazer',
    'sports coat': 'blazer',
    'suit jacket': 'blazer',
    'formal jacket': 'blazer',
    'office blazer': 'blazer',
    'work blazer': 'blazer',
    
    # Jacket (casual)
    'jacket': 'jacket',
    'bomber': 'jacket',
    'bomber jacket': 'jacket',
    'varsity': 'jacket',
    'varsity jacket': 'jacket',
    'flight jacket': 'jacket',
    'windbreaker': 'jacket',
    'rain jacket': 'jacket',
    'casual jacket': 'jacket',
    'light jacket': 'jacket',
    'utility jacket': 'jacket',
    'field jacket': 'jacket',
    'anorak': 'jacket',
    'parka': 'jacket',
    
    # Denim jacket
    'denim jacket': 'denim_jacket',
    'jean jacket': 'denim_jacket',
    'jeans jacket': 'denim_jacket',
    
    # Leather jacket
    'leather jacket': 'leather_jacket',
    'biker jacket': 'leather_jacket',
    'moto jacket': 'leather_jacket',
    'motorcycle jacket': 'leather_jacket',
    'faux leather jacket': 'leather_jacket',
    
    # Coat
    'coat': 'coat',
    'overcoat': 'coat',
    'over coat': 'coat',
    'trench': 'coat',
    'trench coat': 'coat',
    'trenchcoat': 'coat',
    'peacoat': 'coat',
    'pea coat': 'coat',
    'wool coat': 'coat',
    'winter coat': 'coat',
    'long coat': 'coat',
    'mac': 'coat',  # British - mackintosh/raincoat
    'macintosh': 'coat',
    'mackintosh': 'coat',
    'cagoule': 'coat',  # British - thin windproof jacket
    
    # Vest / Waistcoat
    'vest': 'vest',
    'waistcoat': 'vest',  # British term
    'gilet': 'vest',
    'bodywarmer': 'vest',  # British term
    'puffer vest': 'vest',
    'quilted vest': 'vest',
    'suit vest': 'vest',
    'formal vest': 'vest',
    
    # Shrug
    'shrug': 'shrug',
    'bolero': 'shrug',
    'bolero jacket': 'shrug',
    
    # Nehru jacket (Indian)
    'nehru jacket': 'nehru_jacket',
    'bandhgala': 'nehru_jacket',
    'bandh gala': 'nehru_jacket',
    'modi jacket': 'nehru_jacket',
    'ethnic jacket': 'nehru_jacket',
    'indian jacket': 'nehru_jacket',
    
    # =========================================================================
    # FULL BODY
    # =========================================================================
    # Dress
    'dress': 'dress',
    'frock': 'dress',
    'midi dress': 'dress',
    'maxi dress': 'dress',
    'mini dress': 'dress',
    'shift dress': 'dress',
    'sheath dress': 'dress',
    'a-line dress': 'dress',
    'wrap dress': 'dress',
    'shirt dress': 'dress',
    'bodycon dress': 'dress',
    'cocktail dress': 'dress',
    'party dress': 'dress',
    'casual dress': 'dress',
    'summer dress': 'dress',
    'sundress': 'dress',
    'sun dress': 'dress',
    'little black dress': 'dress',
    'lbd': 'dress',
    
    # Gown
    'gown': 'gown',
    'evening gown': 'gown',
    'ball gown': 'gown',
    'formal gown': 'gown',
    'wedding gown': 'gown',
    'prom dress': 'gown',
    
    # Jumpsuit
    'jumpsuit': 'jumpsuit',
    'jump suit': 'jumpsuit',
    'romper': 'jumpsuit',
    'playsuit': 'jumpsuit',
    'onepiece': 'jumpsuit',
    'one-piece': 'jumpsuit',
    'one piece': 'jumpsuit',
    'boiler suit': 'jumpsuit',
    'coverall': 'jumpsuit',
    'coveralls': 'jumpsuit',
    'dungarees': 'jumpsuit',  # British - overalls
    'overalls': 'jumpsuit',
    
    # Saree
    'saree': 'saree',
    'sari': 'saree',
    'saari': 'saree',
    'designer saree': 'saree',
    'silk saree': 'saree',
    'cotton saree': 'saree',
    
    # Lehenga (full outfit)
    'lehenga choli': 'lehenga',
    'lehanga choli': 'lehenga',
    'chaniya choli': 'lehenga',
    'ghagra choli': 'lehenga',
    
    # Anarkali
    'anarkali': 'anarkali',
    'anarkali suit': 'anarkali',
    'floor length kurta': 'anarkali',
    'anarkali dress': 'anarkali',
    
    # Salwar suit
    'salwar suit': 'salwar_suit',
    'shalwar suit': 'salwar_suit',
    'salwar kameez': 'salwar_suit',
    'shalwar kameez': 'salwar_suit',
    'punjabi suit': 'salwar_suit',
    'suit': 'salwar_suit',  # In Indian context, often means salwar suit
    
    # Sherwani (Indian men's formal)
    'sherwani': 'sherwani',
    'sherwaani': 'sherwani',
}


# =============================================================================
# CATEGORY KEYWORDS FOR FUZZY MATCHING
# =============================================================================

CATEGORY_KEYWORDS = {
    GarmentCategory.TOP: [
        'shirt', 'tshirt', 't-shirt', 'tee', 'top', 'blouse', 'kurta', 'kurti',
        'polo', 'henley', 'hoodie', 'sweatshirt', 'sweater', 'jumper', 'cardigan',
        'turtleneck', 'tank', 'cami', 'camisole', 'crop', 'tunic', 'kameez',
        'choli', 'vest', 'singlet', 'pullover', 'crewneck', 'jersey'
    ],
    GarmentCategory.BOTTOM: [
        'jeans', 'denim', 'trouser', 'pant', 'short', 'skirt', 'jogger', 
        'legging', 'chino', 'khaki', 'cargo', 'palazzo', 'salwar', 'churidar',
        'dhoti', 'lehenga', 'ghagra', 'slack', 'bottom', 'jegging'
    ],
    GarmentCategory.OUTERWEAR: [
        'blazer', 'jacket', 'coat', 'cardigan', 'sweater', 'jumper', 'hoodie',
        'vest', 'waistcoat', 'gilet', 'shrug', 'bolero', 'parka', 'anorak',
        'trench', 'overcoat', 'peacoat', 'nehru', 'bandhgala', 'windbreaker'
    ],
    GarmentCategory.FULL_BODY: [
        'dress', 'gown', 'jumpsuit', 'romper', 'playsuit', 'saree', 'sari',
        'lehenga', 'anarkali', 'salwar suit', 'salwar kameez', 'frock',
        'onepiece', 'coverall', 'dungaree', 'overall', 'sherwani'
    ],
    GarmentCategory.ACCESSORY: [
        'belt', 'watch', 'bag', 'handbag', 'purse', 'clutch', 'wallet',
        'scarf', 'muffler', 'tie', 'bow tie', 'earring', 'necklace', 'bracelet',
        'ring', 'sunglasses', 'glasses', 'hat', 'cap', 'beanie', 'glove',
        'jewelry', 'jewellery', 'bangle', 'anklet', 'dupatta'
    ],
    GarmentCategory.FOOTWEAR: [
        'shoe', 'sneaker', 'trainer', 'boot', 'sandal', 'heel', 'flat',
        'loafer', 'oxford', 'derby', 'brogue', 'slipper', 'mule', 'pump',
        'stiletto', 'wedge', 'espadrille', 'flip flop', 'jutti', 'mojari'
    ]
}


# =============================================================================
# MAIN NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_garment(garment: str) -> str:
    """
    Normalize garment name to canonical form.
    
    Args:
        garment: Raw garment name from user input or LLM parsing
        
    Returns:
        Canonical garment name
    
    Examples:
        'lounge tshirt' -> 't-shirt'
        'formal pants' -> 'trousers'
        'jumper' -> 'sweater'
    """
    if not garment:
        return 'top'  # Default
    
    # Clean input
    garment_clean = garment.lower().strip()
    garment_clean = re.sub(r'\s+', ' ', garment_clean)  # Normalize whitespace
    
    # Direct lookup first
    if garment_clean in GARMENT_NORMALIZATION:
        return GARMENT_NORMALIZATION[garment_clean]
    
    # Try partial matching - check if any key is contained in the garment
    for variant, canonical in GARMENT_NORMALIZATION.items():
        if variant in garment_clean:
            return canonical
    
    # Try reverse - check if garment is contained in any key
    for variant, canonical in GARMENT_NORMALIZATION.items():
        if garment_clean in variant:
            return canonical
    
    # Check for category keywords
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in garment_clean:
                # Return the keyword as a reasonable canonical form
                if keyword in GARMENT_NORMALIZATION:
                    return GARMENT_NORMALIZATION[keyword]
                return keyword
    
    # Return original if no match found
    return garment_clean


def get_garment_category(garment: str) -> GarmentCategory:
    """
    Determine garment category using fuzzy keyword matching.
    
    Args:
        garment: Garment name (raw or normalized)
        
    Returns:
        GarmentCategory enum value
    """
    if not garment:
        return GarmentCategory.UNKNOWN
    
    # Normalize first
    normalized = normalize_garment(garment)
    garment_lower = normalized.lower()
    
    # Check each category's keywords
    match_scores = {}
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in garment_lower or garment_lower in keyword:
                # Exact match gets higher score
                if keyword == garment_lower:
                    score += 10
                else:
                    score += 1
        match_scores[category] = score
    
    # Return category with highest score
    if max(match_scores.values()) > 0:
        return max(match_scores, key=match_scores.get)
    
    return GarmentCategory.UNKNOWN


def get_needed_categories(anchor_garment: str) -> Dict[str, bool]:
    """
    Determine what categories are needed based on anchor garment.
    
    Args:
        anchor_garment: The main garment the user has/wants
        
    Returns:
        Dict with 'need_tops', 'need_bottoms', 'need_outerwear'
    """
    category = get_garment_category(anchor_garment)
    normalized = normalize_garment(anchor_garment)
    
    result = {
        'need_tops': False,
        'need_bottoms': False,
        'need_outerwear': False
    }
    
    if category == GarmentCategory.OUTERWEAR:
        # Have outerwear -> need tops and bottoms
        result['need_tops'] = True
        result['need_bottoms'] = True
        
    elif category == GarmentCategory.TOP:
        # Have top -> need bottoms
        result['need_bottoms'] = True
        # Optionally suggest outerwear for formal occasions
        
    elif category == GarmentCategory.BOTTOM:
        # Have bottom -> need tops
        result['need_tops'] = True
        
    elif category == GarmentCategory.FULL_BODY:
        # Full body (dress, jumpsuit) -> might need outerwear only
        result['need_outerwear'] = True  # Optional
        
    else:
        # Unknown or accessory - default to suggesting both
        result['need_tops'] = True
        result['need_bottoms'] = True
    
    return result


def parse_compound_garment(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse compound garment descriptions to extract garment type and modifiers.
    
    Args:
        text: Compound description like "navy lounge tshirt" or "formal black blazer"
        
    Returns:
        Tuple of (garment_type, modifiers_string)
    
    Example:
        "navy lounge tshirt" -> ("t-shirt", "navy lounge")
        "formal black blazer" -> ("blazer", "formal black")
    """
    if not text:
        return None, None
    
    text_clean = text.lower().strip()
    words = text_clean.split()
    
    # Find the garment keyword
    garment_word = None
    garment_idx = -1
    
    for i, word in enumerate(words):
        # Check if this word matches a garment
        test_garment = normalize_garment(word)
        if test_garment != word:  # It was normalized, so it's a garment
            garment_word = word
            garment_idx = i
            break
        
        # Also check if word is in any category keywords
        for category, keywords in CATEGORY_KEYWORDS.items():
            if word in keywords:
                garment_word = word
                garment_idx = i
                break
        
        if garment_word:
            break
    
    if garment_word:
        normalized = normalize_garment(garment_word)
        modifiers = ' '.join(words[:garment_idx] + words[garment_idx+1:])
        return normalized, modifiers.strip() if modifiers else None
    
    # Try full text normalization as fallback
    normalized = normalize_garment(text_clean)
    if normalized != text_clean:
        return normalized, None
    
    return None, text_clean


# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def validate_garment_normalization(test_inputs: List[str]) -> List[Dict]:
    """
    Test normalization on a list of inputs.
    
    Returns list of dicts with input, normalized, category.
    """
    results = []
    for inp in test_inputs:
        normalized = normalize_garment(inp)
        category = get_garment_category(inp)
        needed = get_needed_categories(inp)
        
        results.append({
            'input': inp,
            'normalized': normalized,
            'category': category.value,
            'need_tops': needed['need_tops'],
            'need_bottoms': needed['need_bottoms'],
            'need_outerwear': needed['need_outerwear']
        })
    
    return results


# Quick test if run directly
if __name__ == "__main__":
    test_cases = [
        "lounge tshirt",
        "Men's Lounge Tshirt",
        "navy blazer",
        "formal pants",
        "jumper",  # British for sweater
        "waistcoat",  # British for vest
        "kurta",
        "salwar kameez",
        "graphic tee",
        "joggers",
        "tracksuit bottoms",  # British
        "trousers",
        "dress shirt",
        "polo",
        "skinny jeans",
        "maxi dress",
        "anarkali suit",
    ]
    
    print("Garment Normalization Test Results:")
    print("=" * 80)
    
    results = validate_garment_normalization(test_cases)
    for r in results:
        print(f"Input: '{r['input']}'")
        print(f"  → Normalized: '{r['normalized']}' | Category: {r['category']}")
        print(f"  → Need: tops={r['need_tops']}, bottoms={r['need_bottoms']}, outerwear={r['need_outerwear']}")
        print()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'GarmentCategory',
    'GARMENT_NORMALIZATION',
    'CATEGORY_KEYWORDS',
    'normalize_garment',
    'get_garment_category',
    'get_needed_categories',
    'parse_compound_garment',
    'validate_garment_normalization',
]