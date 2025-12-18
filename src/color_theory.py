"""
Color Theory Module for Fashion Ensemble Matching

Based on:
- Traditional color wheel theory (Itten, Munsell)
- Fashion industry practices
- Seasonal color analysis principles
- Indian ethnic wear conventions

Author: Fashion Ensemble Builder
"""

from typing import List, Tuple, Dict, Optional
from enum import Enum
import colorsys
import math


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class HarmonyType(Enum):
    COMPLEMENTARY = "complementary"       # Opposite on wheel (high contrast)
    ANALOGOUS = "analogous"               # Adjacent colors (harmonious)
    TRIADIC = "triadic"                   # 3 equidistant (vibrant)
    SPLIT_COMPLEMENTARY = "split_comp"    # Complement + neighbors (balanced)
    TETRADIC = "tetradic"                 # 4 colors rectangle (rich)
    MONOCHROMATIC = "monochromatic"       # Same hue, different L/S (safe)
    NEUTRAL_ACCENT = "neutral_accent"     # Neutrals + one pop color


class ColorTemperature(Enum):
    WARM = "warm"
    COOL = "cool"
    NEUTRAL = "neutral"


class SeasonalPalette(Enum):
    """Seasonal color analysis - affects which colors suit together"""
    SPRING = "spring"   # Warm, bright, clear
    SUMMER = "summer"   # Cool, muted, soft
    AUTUMN = "autumn"   # Warm, muted, rich
    WINTER = "winter"   # Cool, bright, clear


# =============================================================================
# NEUTRAL COLORS (Fashion's Foundation)
# =============================================================================

NEUTRALS_HSL = {
    # Color: (H, S%, L%) - approximate values
    'black': (0, 0, 5),
    'white': (0, 0, 98),
    'off_white': (40, 20, 95),
    'cream': (45, 40, 90),
    'ivory': (50, 30, 93),
    'grey': (0, 0, 50),
    'charcoal': (0, 0, 25),
    'silver': (0, 5, 75),
    'navy': (220, 60, 20),
    'beige': (40, 30, 75),
    'tan': (35, 40, 55),
    'camel': (35, 50, 50),
    'brown': (30, 50, 30),
    'chocolate': (25, 60, 20),
    'khaki': (50, 35, 55),
    'olive': (80, 40, 35),
    'taupe': (30, 15, 50),
}

NEUTRAL_NAMES = set(NEUTRALS_HSL.keys())


# =============================================================================
# FASHION COLOR DEFINITIONS (HSL)
# =============================================================================

FASHION_COLORS_HSL = {
    # Reds
    'red': (0, 85, 50),
    'crimson': (348, 85, 45),
    'scarlet': (5, 90, 50),
    'burgundy': (345, 70, 25),
    'maroon': (0, 65, 25),
    'coral': (16, 80, 65),
    'rust': (15, 70, 40),
    'terracotta': (15, 55, 45),
    
    # Oranges
    'orange': (30, 90, 55),
    'peach': (25, 70, 75),
    'apricot': (30, 65, 70),
    'tangerine': (25, 90, 55),
    'burnt_orange': (25, 80, 40),
    
    # Yellows
    'yellow': (55, 90, 55),
    'gold': (45, 85, 50),
    'mustard': (45, 75, 45),
    'lemon': (55, 85, 70),
    'amber': (40, 90, 50),
    'saffron': (45, 95, 55),  # Important in Indian fashion
    
    # Greens
    'green': (120, 60, 40),
    'emerald': (140, 70, 40),
    'sage': (100, 25, 55),
    'olive': (80, 40, 35),
    'mint': (150, 50, 75),
    'teal': (175, 60, 35),
    'forest': (130, 50, 25),
    'lime': (90, 70, 50),
    'seafoam': (160, 45, 65),
    
    # Blues
    'blue': (220, 70, 50),
    'navy': (220, 60, 20),
    'royal_blue': (225, 80, 45),
    'sky_blue': (200, 70, 70),
    'powder_blue': (200, 50, 80),
    'cobalt': (215, 85, 45),
    'turquoise': (175, 70, 50),
    'aqua': (180, 60, 55),
    'indigo': (240, 50, 30),
    'denim': (215, 50, 45),
    
    # Purples
    'purple': (280, 60, 45),
    'lavender': (270, 50, 75),
    'violet': (270, 70, 50),
    'plum': (300, 45, 35),
    'mauve': (310, 30, 60),
    'magenta': (300, 80, 50),
    'wine': (340, 60, 30),
    'aubergine': (290, 50, 25),  # Eggplant
    
    # Pinks
    'pink': (340, 70, 75),
    'hot_pink': (330, 85, 55),
    'blush': (350, 50, 85),
    'rose': (350, 55, 60),
    'fuchsia': (320, 80, 50),
    'salmon': (5, 60, 70),
    'dusty_pink': (350, 35, 65),
    'millennial_pink': (5, 40, 80),
    
    # Indian Fashion Specific
    'rani_pink': (335, 85, 45),    # Bright magenta-pink
    'peacock_blue': (185, 80, 35),
    'parrot_green': (100, 75, 45),
    'turmeric': (45, 90, 55),
    'vermillion': (5, 90, 50),     # Sindoor color
}

# Merge all colors
ALL_COLORS_HSL = {**NEUTRALS_HSL, **FASHION_COLORS_HSL}


# =============================================================================
# COLOR CONVERSION UTILITIES
# =============================================================================

def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB (0-255) to HSL (H: 0-360, S: 0-100, L: 0-100)"""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (int(h * 360), int(s * 100), int(l * 100))


def hsl_to_rgb(h: int, s: int, l: int) -> Tuple[int, int, int]:
    """Convert HSL to RGB (0-255)"""
    h, s, l = h / 360.0, s / 100.0, l / 100.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def hex_to_hsl(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex (#RRGGBB) to HSL"""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb_to_hsl(r, g, b)


def hue_distance(h1: int, h2: int) -> int:
    """Calculate shortest distance between two hues on 360° wheel"""
    diff = abs(h1 - h2)
    return min(diff, 360 - diff)


# =============================================================================
# COLOR CLASSIFICATION
# =============================================================================

def is_neutral(h: int, s: int, l: int = None) -> bool:
    """
    Determine if a color is neutral.
    Neutrals: very low saturation OR very high/low lightness
    """
    if s < 15:  # Very desaturated
        return True
    if l is not None and (l < 15 or l > 90):  # Very dark or very light
        return True
    # Special case: navy and olive are neutrals despite saturation
    if s < 65 and h in range(210, 230):  # Navy range
        return True
    if s < 50 and h in range(70, 100):  # Olive range
        return True
    return False


def get_color_temperature(h: int, s: int) -> ColorTemperature:
    """
    Determine if color is warm, cool, or neutral.
    
    Warm: Red, Orange, Yellow (0-60°, 300-360°)
    Cool: Green, Blue, Purple (120-270°)
    Neutral: Low saturation or boundary hues
    """
    if s < 15:
        return ColorTemperature.NEUTRAL
    
    # Warm hues
    if h <= 60 or h >= 300:
        return ColorTemperature.WARM
    
    # Cool hues
    if 150 <= h <= 270:
        return ColorTemperature.COOL
    
    # Boundary zones - consider neutral
    return ColorTemperature.NEUTRAL


def get_color_season(h: int, s: int, l: int) -> SeasonalPalette:
    """
    Classify color into seasonal palette.
    
    Spring: Warm + Bright + Clear (high S, medium-high L, warm H)
    Summer: Cool + Muted + Soft (low-medium S, medium-high L, cool H)
    Autumn: Warm + Muted + Rich (medium S, low-medium L, warm H)
    Winter: Cool + Bright + Clear (high S, extreme L, cool H)
    """
    temp = get_color_temperature(h, s)
    
    is_bright = s > 60
    is_muted = s < 45
    is_light = l > 60
    is_dark = l < 40
    
    if temp == ColorTemperature.WARM:
        if is_bright and is_light:
            return SeasonalPalette.SPRING
        else:
            return SeasonalPalette.AUTUMN
    else:  # Cool or Neutral
        if is_bright or is_dark:
            return SeasonalPalette.WINTER
        else:
            return SeasonalPalette.SUMMER


def classify_saturation(s: int) -> str:
    """Classify saturation level"""
    if s < 20:
        return "desaturated"
    elif s < 45:
        return "muted"
    elif s < 70:
        return "medium"
    else:
        return "vibrant"


def classify_lightness(l: int) -> str:
    """Classify lightness level"""
    if l < 20:
        return "very_dark"
    elif l < 40:
        return "dark"
    elif l < 60:
        return "medium"
    elif l < 80:
        return "light"
    else:
        return "very_light"


# =============================================================================
# COLOR HARMONY FUNCTIONS
# =============================================================================

def get_complementary(h: int) -> int:
    """Get complementary hue (opposite on wheel)"""
    return (h + 180) % 360


def get_analogous(h: int, spread: int = 30) -> List[int]:
    """Get analogous hues (adjacent on wheel)"""
    return [
        (h - spread) % 360,
        h,
        (h + spread) % 360
    ]


def get_triadic(h: int) -> List[int]:
    """Get triadic hues (3 equidistant)"""
    return [h, (h + 120) % 360, (h + 240) % 360]


def get_split_complementary(h: int, spread: int = 30) -> List[int]:
    """Get split-complementary (complement's neighbors)"""
    comp = get_complementary(h)
    return [h, (comp - spread) % 360, (comp + spread) % 360]


def get_tetradic(h: int) -> List[int]:
    """Get tetradic/rectangle (4 colors)"""
    return [h, (h + 90) % 360, (h + 180) % 360, (h + 270) % 360]


def check_hue_harmony(h1: int, h2: int, harmony_type: HarmonyType, tolerance: int = 15) -> bool:
    """
    Check if two hues are in harmony according to specified type.
    
    Args:
        h1, h2: Hue values (0-360)
        harmony_type: Type of harmony to check
        tolerance: Degrees of tolerance for matching
    
    Returns:
        True if hues are harmonious
    """
    distance = hue_distance(h1, h2)
    
    if harmony_type == HarmonyType.COMPLEMENTARY:
        return abs(distance - 180) <= tolerance
    
    elif harmony_type == HarmonyType.ANALOGOUS:
        return distance <= 60
    
    elif harmony_type == HarmonyType.TRIADIC:
        return abs(distance - 120) <= tolerance or abs(distance - 240) <= tolerance
    
    elif harmony_type == HarmonyType.SPLIT_COMPLEMENTARY:
        return abs(distance - 150) <= tolerance or abs(distance - 210) <= tolerance
    
    elif harmony_type == HarmonyType.TETRADIC:
        return distance <= tolerance or abs(distance - 90) <= tolerance or \
               abs(distance - 180) <= tolerance or abs(distance - 270) <= tolerance
    
    elif harmony_type == HarmonyType.MONOCHROMATIC:
        return distance <= tolerance
    
    return False


# =============================================================================
# FASHION-SPECIFIC COMPATIBILITY
# =============================================================================

def saturation_compatible(s1: int, s2: int, tolerance: int = 25) -> bool:
    """
    Check if saturations are compatible.
    
    Fashion rule: Muted with muted, vibrant with vibrant.
    Exception: Neutrals go with anything.
    """
    # If either is very desaturated (neutral), always compatible
    if s1 < 20 or s2 < 20:
        return True
    
    # Both muted (< 45) or both vibrant (> 55)
    both_muted = s1 < 45 and s2 < 45
    both_vibrant = s1 > 55 and s2 > 55
    
    # Within tolerance
    within_range = abs(s1 - s2) <= tolerance
    
    return both_muted or both_vibrant or within_range


def temperature_compatible(h1: int, s1: int, h2: int, s2: int) -> bool:
    """
    Check if color temperatures are compatible.
    
    Fashion rule: Warm with warm, cool with cool.
    Exception: Neutrals bridge everything.
    """
    temp1 = get_color_temperature(h1, s1)
    temp2 = get_color_temperature(h2, s2)
    
    # Neutrals are wildcards
    if temp1 == ColorTemperature.NEUTRAL or temp2 == ColorTemperature.NEUTRAL:
        return True
    
    return temp1 == temp2


def lightness_contrast_score(l1: int, l2: int) -> float:
    """
    Score the lightness contrast between two colors.
    
    Fashion rule: Some contrast is good (dark + light),
    but extreme contrast can be jarring.
    
    Returns: 0.0 to 1.0 (higher = better contrast)
    """
    diff = abs(l1 - l2)
    
    if diff < 10:
        return 0.3  # Too similar, might blend
    elif diff < 30:
        return 0.6  # Subtle contrast
    elif diff < 50:
        return 1.0  # Ideal contrast
    elif diff < 70:
        return 0.8  # Strong contrast
    else:
        return 0.6  # Extreme contrast (can work but bold)


def season_compatible(h1: int, s1: int, l1: int, h2: int, s2: int, l2: int) -> bool:
    """
    Check if colors belong to compatible seasonal palettes.
    
    Compatible seasons:
    - Spring + Autumn (both warm)
    - Summer + Winter (both cool)
    - Any season + Neutrals
    """
    if is_neutral(h1, s1, l1) or is_neutral(h2, s2, l2):
        return True
    
    season1 = get_color_season(h1, s1, l1)
    season2 = get_color_season(h2, s2, l2)
    
    warm_seasons = {SeasonalPalette.SPRING, SeasonalPalette.AUTUMN}
    cool_seasons = {SeasonalPalette.SUMMER, SeasonalPalette.WINTER}
    
    both_warm = season1 in warm_seasons and season2 in warm_seasons
    both_cool = season1 in cool_seasons and season2 in cool_seasons
    
    return both_warm or both_cool


# =============================================================================
# MAIN COMPATIBILITY SCORING
# =============================================================================

def calculate_color_compatibility(
    color1_hsl: Tuple[int, int, int],
    color2_hsl: Tuple[int, int, int],
    preferred_harmony: Optional[HarmonyType] = None
) -> Dict:
    """
    Calculate comprehensive compatibility score between two colors.
    
    Args:
        color1_hsl: (H, S, L) of first color
        color2_hsl: (H, S, L) of second color
        preferred_harmony: Optional specific harmony to check
    
    Returns:
        Dict with overall score (0-100) and component scores
    """
    h1, s1, l1 = color1_hsl
    h2, s2, l2 = color2_hsl
    
    result = {
        'overall_score': 0,
        'is_compatible': False,
        'harmony_type': None,
        'details': {}
    }
    
    # ===================
    # NEUTRAL FAST PATH
    # ===================
    if is_neutral(h1, s1, l1) or is_neutral(h2, s2, l2):
        result['overall_score'] = 85
        result['is_compatible'] = True
        result['harmony_type'] = HarmonyType.NEUTRAL_ACCENT
        result['details']['note'] = "Neutral color - universally compatible"
        
        # Bonus for good lightness contrast
        l_score = lightness_contrast_score(l1, l2)
        result['overall_score'] = min(100, 85 + int(l_score * 15))
        return result
    
    # ===================
    # COMPONENT SCORES
    # ===================
    scores = {}
    
    # 1. Hue Harmony (40 points max)
    harmonies_to_check = [
        (HarmonyType.COMPLEMENTARY, 40),
        (HarmonyType.ANALOGOUS, 35),
        (HarmonyType.SPLIT_COMPLEMENTARY, 38),
        (HarmonyType.TRIADIC, 32),
        (HarmonyType.MONOCHROMATIC, 35),
    ]
    
    best_harmony_score = 0
    best_harmony_type = None
    
    for harmony_type, max_score in harmonies_to_check:
        if preferred_harmony and harmony_type != preferred_harmony:
            continue
        if check_hue_harmony(h1, h2, harmony_type):
            if max_score > best_harmony_score:
                best_harmony_score = max_score
                best_harmony_type = harmony_type
    
    scores['hue_harmony'] = best_harmony_score
    result['harmony_type'] = best_harmony_type
    
    # 2. Saturation Compatibility (25 points max)
    if saturation_compatible(s1, s2):
        scores['saturation'] = 25
    else:
        # Partial score based on how close
        diff = abs(s1 - s2)
        scores['saturation'] = max(0, 25 - diff // 3)
    
    # 3. Temperature Compatibility (20 points max)
    if temperature_compatible(h1, s1, h2, s2):
        scores['temperature'] = 20
    else:
        scores['temperature'] = 5  # Small penalty, not dealbreaker
    
    # 4. Lightness Contrast (15 points max)
    l_score = lightness_contrast_score(l1, l2)
    scores['lightness_contrast'] = int(l_score * 15)
    
    # 5. Seasonal Compatibility (bonus: up to 10 points)
    if season_compatible(h1, s1, l1, h2, s2, l2):
        scores['seasonal_bonus'] = 10
    else:
        scores['seasonal_bonus'] = 0
    
    # ===================
    # CALCULATE TOTAL
    # ===================
    total = sum(scores.values())
    result['overall_score'] = min(100, total)
    result['is_compatible'] = total >= 60
    result['details'] = scores
    
    return result


# =============================================================================
# 60-30-10 RULE FOR OUTFIT
# =============================================================================

class ColorRole(Enum):
    DOMINANT = 60      # 60% - largest visual area
    SECONDARY = 30     # 30% - supporting color
    ACCENT = 10        # 10% - pop of color


def assign_color_roles(colors_hsl: List[Tuple[int, int, int]]) -> Dict[ColorRole, Tuple[int, int, int]]:
    """
    Assign colors to 60-30-10 roles based on fashion principles.
    
    Rules:
    - Neutrals prefer DOMINANT (60%) role
    - Brightest color usually ACCENT (10%)
    - Most saturated non-neutral is SECONDARY (30%)
    
    Args:
        colors_hsl: List of 2-3 colors as (H, S, L) tuples
    
    Returns:
        Dict mapping ColorRole to color
    """
    if len(colors_hsl) < 2:
        raise ValueError("Need at least 2 colors for 60-30-10 rule")
    
    colors_with_info = []
    for hsl in colors_hsl:
        h, s, l = hsl
        colors_with_info.append({
            'hsl': hsl,
            'is_neutral': is_neutral(h, s, l),
            'saturation': s,
            'lightness': l
        })
    
    roles = {}
    remaining = colors_with_info.copy()
    
    # 1. DOMINANT (60%): Prefer neutral, else darkest/most muted
    neutrals = [c for c in remaining if c['is_neutral']]
    if neutrals:
        # Pick the neutral closest to medium lightness
        dominant = min(neutrals, key=lambda c: abs(c['lightness'] - 50))
    else:
        # Pick most muted (lowest saturation)
        dominant = min(remaining, key=lambda c: c['saturation'])
    
    roles[ColorRole.DOMINANT] = dominant['hsl']
    remaining.remove(dominant)
    
    # 2. If only 2 colors, second is SECONDARY
    if len(remaining) == 1:
        roles[ColorRole.SECONDARY] = remaining[0]['hsl']
        return roles
    
    # 3. ACCENT (10%): Most saturated/vibrant
    accent = max(remaining, key=lambda c: c['saturation'])
    roles[ColorRole.ACCENT] = accent['hsl']
    remaining.remove(accent)
    
    # 4. SECONDARY (30%): Whatever's left
    if remaining:
        roles[ColorRole.SECONDARY] = remaining[0]['hsl']
    
    return roles


def suggest_role_for_garment(garment_type: str) -> ColorRole:
    """
    Suggest which color role a garment type typically takes.
    
    Based on visual area the garment occupies.
    """
    dominant_garments = {
        'pants', 'trousers', 'jeans', 'palazzos', 'skirt', 'dress', 
        'saree', 'lehenga', 'gown', 'coat', 'overcoat'
    }
    
    secondary_garments = {
        'shirt', 'blouse', 'top', 'kurta', 'kurti', 'tshirt', 
        'sweater', 'blazer', 'jacket', 'cardigan', 'hoodie'
    }
    
    accent_garments = {
        'scarf', 'dupatta', 'tie', 'belt', 'pocket_square',
        'shoes', 'bag', 'jewelry', 'watch', 'hat'
    }
    
    garment_lower = garment_type.lower()
    
    if garment_lower in dominant_garments:
        return ColorRole.DOMINANT
    elif garment_lower in secondary_garments:
        return ColorRole.SECONDARY
    elif garment_lower in accent_garments:
        return ColorRole.ACCENT
    else:
        return ColorRole.SECONDARY  # Default


# =============================================================================
# FIND MATCHING COLORS
# =============================================================================

def find_complementary_colors(
    anchor_hsl: Tuple[int, int, int],
    harmony_type: HarmonyType = HarmonyType.COMPLEMENTARY,
    include_neutrals: bool = True
) -> List[Dict]:
    """
    Find colors that complement the anchor color.
    
    Args:
        anchor_hsl: The base color (H, S, L)
        harmony_type: Type of harmony to use
        include_neutrals: Whether to include neutral suggestions
    
    Returns:
        List of suggested colors with scores
    """
    h, s, l = anchor_hsl
    suggestions = []
    
    # Get harmonious hues based on type
    if harmony_type == HarmonyType.COMPLEMENTARY:
        target_hues = [get_complementary(h)]
    elif harmony_type == HarmonyType.ANALOGOUS:
        target_hues = get_analogous(h)
    elif harmony_type == HarmonyType.TRIADIC:
        target_hues = get_triadic(h)[1:]  # Exclude original
    elif harmony_type == HarmonyType.SPLIT_COMPLEMENTARY:
        target_hues = get_split_complementary(h)[1:]  # Exclude original
    else:
        target_hues = [get_complementary(h)]
    
    # Find named colors close to target hues
    for color_name, (ch, cs, cl) in FASHION_COLORS_HSL.items():
        for target_h in target_hues:
            if hue_distance(ch, target_h) <= 20:
                # Check saturation compatibility
                if saturation_compatible(s, cs):
                    score = calculate_color_compatibility(anchor_hsl, (ch, cs, cl))
                    suggestions.append({
                        'name': color_name,
                        'hsl': (ch, cs, cl),
                        'score': score['overall_score'],
                        'harmony': harmony_type.value
                    })
    
    # Add neutrals if requested
    if include_neutrals:
        for color_name, (nh, ns, nl) in NEUTRALS_HSL.items():
            # Good lightness contrast with anchor
            if abs(l - nl) > 25:
                suggestions.append({
                    'name': color_name,
                    'hsl': (nh, ns, nl),
                    'score': 85,
                    'harmony': 'neutral'
                })
    
    # Sort by score descending
    suggestions.sort(key=lambda x: x['score'], reverse=True)
    
    return suggestions[:10]  # Top 10


def get_safe_combinations(anchor_color_name: str) -> List[str]:
    """
    Get pre-defined safe color combinations for common colors.
    
    Based on classic fashion pairings.
    """
    SAFE_COMBOS = {
        # Neutrals
        'black': ['white', 'grey', 'red', 'pink', 'gold', 'camel'],
        'white': ['navy', 'black', 'denim', 'any'],
        'navy': ['white', 'cream', 'coral', 'gold', 'burgundy', 'pink'],
        'grey': ['pink', 'yellow', 'purple', 'blue', 'burgundy'],
        'beige': ['navy', 'brown', 'white', 'burgundy', 'forest'],
        'brown': ['cream', 'blue', 'orange', 'green', 'gold'],
        
        # Colors
        'red': ['navy', 'white', 'black', 'denim', 'camel', 'grey'],
        'yellow': ['navy', 'grey', 'white', 'denim', 'purple', 'brown'],
        'orange': ['navy', 'white', 'brown', 'teal', 'cream'],
        'green': ['white', 'cream', 'brown', 'navy', 'pink', 'gold'],
        'blue': ['white', 'cream', 'orange', 'coral', 'brown', 'grey'],
        'purple': ['grey', 'white', 'gold', 'pink', 'navy'],
        'pink': ['grey', 'navy', 'white', 'denim', 'olive', 'brown'],
        
        # Indian fashion specific
        'rani_pink': ['gold', 'white', 'navy', 'emerald'],
        'mustard': ['navy', 'teal', 'burgundy', 'white', 'brown'],
        'teal': ['coral', 'mustard', 'gold', 'cream', 'burgundy'],
        'maroon': ['gold', 'cream', 'pink', 'peach', 'mint'],
        'saffron': ['white', 'navy', 'green', 'maroon'],
    }
    
    return SAFE_COMBOS.get(anchor_color_name.lower(), ['white', 'black', 'grey'])


# =============================================================================
# COLOR NAME DETECTION
# =============================================================================

def closest_color_name(hsl: Tuple[int, int, int]) -> str:
    """
    Find the closest named color for an HSL value.
    """
    h, s, l = hsl
    
    # Check neutrals first
    if is_neutral(h, s, l):
        best_match = None
        best_dist = float('inf')
        for name, (nh, ns, nl) in NEUTRALS_HSL.items():
            dist = math.sqrt((s - ns)**2 + (l - nl)**2)
            if dist < best_dist:
                best_dist = dist
                best_match = name
        return best_match
    
    # Find closest in fashion colors
    best_match = None
    best_dist = float('inf')
    
    for name, (ch, cs, cl) in FASHION_COLORS_HSL.items():
        h_dist = hue_distance(h, ch)
        s_dist = abs(s - cs)
        l_dist = abs(l - cl)
        
        # Weighted distance (hue matters most)
        dist = math.sqrt((h_dist * 2)**2 + s_dist**2 + l_dist**2)
        
        if dist < best_dist:
            best_dist = dist
            best_match = name
    
    return best_match


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def get_outfit_colors(
    anchor_color: str,
    num_colors: int = 3,
    style: str = "balanced"
) -> Dict[str, List[str]]:
    """
    Get recommended colors for an outfit based on anchor color.
    
    Args:
        anchor_color: Name of the main color
        num_colors: Total colors in outfit (2-4)
        style: "bold" (complementary), "subtle" (analogous), "balanced" (mixed)
    
    Returns:
        Dict with color suggestions by role
    """
    anchor_hsl = ALL_COLORS_HSL.get(anchor_color.lower())
    
    if anchor_hsl is None:
        return {'error': f'Unknown color: {anchor_color}'}
    
    # Choose harmony based on style
    if style == "bold":
        harmony = HarmonyType.COMPLEMENTARY
    elif style == "subtle":
        harmony = HarmonyType.ANALOGOUS
    else:
        harmony = HarmonyType.SPLIT_COMPLEMENTARY
    
    suggestions = find_complementary_colors(anchor_hsl, harmony)
    
    # Build outfit
    result = {
        'anchor': anchor_color,
        'dominant': [],
        'secondary': [],
        'accent': []
    }
    
    # Assign anchor to its natural role
    anchor_role = suggest_role_for_garment('top')  # Assume anchor is a top
    
    # Fill other roles from suggestions
    neutral_suggestions = [s for s in suggestions if s['hsl'][1] < 20]
    color_suggestions = [s for s in suggestions if s['hsl'][1] >= 20]
    
    # Dominant: prefer neutral
    if neutral_suggestions:
        result['dominant'] = [neutral_suggestions[0]['name']]
    elif color_suggestions:
        result['dominant'] = [color_suggestions[0]['name']]
    
    # Secondary and accent from color suggestions
    for i, sug in enumerate(color_suggestions[:3]):
        if i == 0 and not neutral_suggestions:
            continue  # Already used for dominant
        if len(result['secondary']) < 1:
            result['secondary'].append(sug['name'])
        elif len(result['accent']) < 1:
            result['accent'].append(sug['name'])
    
    return result


# =============================================================================
# BOLD COLOR RECOMMENDATIONS (Color Theory Based)
# =============================================================================

# Bold colors that work with each anchor - based on complementary, triadic, split-comp
BOLD_COLOR_PAIRINGS = {
    # Neutrals - can go bold with almost anything
    'black': ['red', 'emerald', 'gold', 'cobalt', 'fuchsia'],
    'white': ['navy', 'emerald', 'burgundy', 'cobalt', 'coral'],
    'grey': ['burgundy', 'emerald', 'mustard', 'coral', 'teal'],
    'charcoal': ['burgundy', 'teal', 'gold', 'coral', 'emerald'],
    'navy': ['coral', 'gold', 'rust', 'mustard', 'blush'],
    'beige': ['burgundy', 'teal', 'cobalt', 'rust', 'emerald'],
    'cream': ['burgundy', 'navy', 'emerald', 'rust', 'plum'],
    'brown': ['teal', 'coral', 'turquoise', 'gold', 'rust'],
    'tan': ['burgundy', 'teal', 'navy', 'emerald', 'cobalt'],
    'khaki': ['burgundy', 'navy', 'coral', 'teal', 'rust'],
    'olive': ['burgundy', 'coral', 'rust', 'gold', 'plum'],
    'camel': ['burgundy', 'navy', 'teal', 'emerald', 'cobalt'],
    
    # Colors - complementary and triadic based
    'red': ['emerald', 'teal', 'gold', 'navy', 'turquoise'],
    'burgundy': ['emerald', 'teal', 'gold', 'mint', 'turquoise'],
    'maroon': ['teal', 'gold', 'emerald', 'mustard', 'mint'],
    'coral': ['teal', 'navy', 'emerald', 'turquoise', 'cobalt'],
    'orange': ['navy', 'teal', 'cobalt', 'turquoise', 'indigo'],
    'rust': ['teal', 'navy', 'emerald', 'turquoise', 'cobalt'],
    'mustard': ['navy', 'purple', 'burgundy', 'teal', 'plum'],
    'yellow': ['purple', 'navy', 'burgundy', 'indigo', 'plum'],
    'gold': ['navy', 'burgundy', 'purple', 'teal', 'plum'],
    'green': ['burgundy', 'coral', 'magenta', 'rust', 'plum'],
    'emerald': ['burgundy', 'coral', 'gold', 'rust', 'magenta'],
    'teal': ['coral', 'burgundy', 'rust', 'gold', 'orange'],
    'mint': ['burgundy', 'coral', 'plum', 'rust', 'magenta'],
    'blue': ['orange', 'coral', 'gold', 'rust', 'mustard'],
    'cobalt': ['orange', 'coral', 'gold', 'rust', 'mustard'],
    'turquoise': ['coral', 'rust', 'burgundy', 'orange', 'gold'],
    'purple': ['gold', 'mustard', 'yellow', 'coral', 'orange'],
    'lavender': ['gold', 'mustard', 'olive', 'coral', 'rust'],
    'plum': ['gold', 'mustard', 'mint', 'coral', 'olive'],
    'pink': ['emerald', 'teal', 'olive', 'navy', 'forest'],
    'blush': ['emerald', 'teal', 'navy', 'forest', 'olive'],
    'magenta': ['emerald', 'teal', 'gold', 'mint', 'olive'],
    'fuchsia': ['emerald', 'teal', 'gold', 'mint', 'lime'],
}

# Safe neutral colors per category and occasion
SAFE_COLORS_BY_CATEGORY = {
    'bottom': {
        'formal': ['black', 'charcoal', 'grey', 'navy', 'beige'],
        'casual': ['black', 'grey', 'khaki', 'olive', 'beige'],
        'ethnic': ['white', 'cream', 'beige', 'black', 'navy'],
        'default': ['black', 'grey', 'navy', 'beige', 'khaki']
    },
    'top': {
        'formal': ['white', 'cream', 'light_blue', 'blush', 'lavender'],
        'casual': ['white', 'black', 'grey', 'navy', 'cream'],
        'ethnic': ['white', 'cream', 'gold', 'beige', 'pink'],
        'default': ['white', 'cream', 'black', 'grey', 'navy']
    },
    'outerwear': {
        'formal': ['black', 'navy', 'charcoal', 'camel', 'grey'],
        'casual': ['black', 'navy', 'olive', 'brown', 'grey'],
        'ethnic': ['black', 'navy', 'maroon', 'gold', 'cream'],
        'default': ['black', 'navy', 'grey', 'charcoal', 'camel']
    }
}


def get_bold_complementary(anchor_color: str, n: int = 3) -> List[str]:
    """
    Get bold color recommendations based on color theory.
    
    Args:
        anchor_color: The anchor piece color
        n: Number of bold colors to return
    
    Returns:
        List of bold colors that complement the anchor
    """
    anchor_lower = anchor_color.lower().replace(' ', '_')
    
    # Direct lookup
    if anchor_lower in BOLD_COLOR_PAIRINGS:
        return BOLD_COLOR_PAIRINGS[anchor_lower][:n]
    
    # Try to find closest match
    for key in BOLD_COLOR_PAIRINGS:
        if key in anchor_lower or anchor_lower in key:
            return BOLD_COLOR_PAIRINGS[key][:n]
    
    # Fallback: use color wheel
    if anchor_lower in ALL_COLORS_HSL:
        h, s, l = ALL_COLORS_HSL[anchor_lower]
        comp_hue = get_complementary(h)
        
        # Find named colors near complementary hue
        bold_options = []
        for name, (ch, cs, cl) in FASHION_COLORS_HSL.items():
            if hue_distance(ch, comp_hue) < 30 and cs > 50:  # Saturated colors
                bold_options.append(name)
        
        if bold_options:
            return bold_options[:n]
    
    # Ultimate fallback
    return ['coral', 'emerald', 'gold'][:n]


def get_safe_colors_for_category(
    category: str, 
    style: str = None, 
    occasion: str = None
) -> List[str]:
    """
    Get safe neutral colors appropriate for a garment category.
    
    Args:
        category: 'top', 'bottom', or 'outerwear'
        style: Style preference (formal, casual, ethnic)
        occasion: Occasion type
    
    Returns:
        List of safe colors for this category
    """
    category_lower = category.lower()
    
    if category_lower not in SAFE_COLORS_BY_CATEGORY:
        return ['black', 'white', 'grey', 'navy', 'beige']
    
    category_colors = SAFE_COLORS_BY_CATEGORY[category_lower]
    
    # Determine style key
    style_key = 'default'
    if style:
        style_lower = style.lower()
        if style_lower in ['formal', 'work', 'office', 'professional']:
            style_key = 'formal'
        elif style_lower in ['casual', 'daily', 'weekend', 'brunch']:
            style_key = 'casual'
        elif style_lower in ['ethnic', 'festive', 'wedding', 'traditional']:
            style_key = 'ethnic'
    elif occasion:
        occasion_lower = occasion.lower()
        if occasion_lower in ['work', 'office', 'meeting', 'interview', 'formal']:
            style_key = 'formal'
        elif occasion_lower in ['casual', 'daily', 'brunch', 'weekend']:
            style_key = 'casual'
        elif occasion_lower in ['festive', 'wedding', 'ethnic', 'traditional']:
            style_key = 'ethnic'
    
    return category_colors.get(style_key, category_colors['default'])


def get_balanced_colors_for_category(
    category: str,
    anchor_color: str,
    style: str = None,
    occasion: str = None,
    n_safe: int = 3,
    n_bold: int = 1
) -> List[Dict]:
    """
    Get balanced color mix: safe neutrals + bold statement options.
    
    Args:
        category: 'top', 'bottom', or 'outerwear'
        anchor_color: Color of anchor piece
        style: Style preference
        occasion: Occasion type
        n_safe: Number of safe colors
        n_bold: Number of bold colors
    
    Returns:
        List of dicts with color and is_bold flag
    """
    result = []
    
    # Get safe colors
    safe_colors = get_safe_colors_for_category(category, style, occasion)
    for color in safe_colors[:n_safe]:
        result.append({
            'color': color,
            'is_bold': False,
            'reasoning': 'neutral_palette'
        })
    
    # Get bold colors
    bold_colors = get_bold_complementary(anchor_color, n_bold)
    for color in bold_colors[:n_bold]:
        result.append({
            'color': color,
            'is_bold': True,
            'reasoning': 'color_theory_complementary'
        })
    
    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'HarmonyType',
    'ColorTemperature', 
    'SeasonalPalette',
    'ColorRole',
    
    # Constants
    'NEUTRALS_HSL',
    'FASHION_COLORS_HSL',
    'ALL_COLORS_HSL',
    'NEUTRAL_NAMES',
    'BOLD_COLOR_PAIRINGS',
    'SAFE_COLORS_BY_CATEGORY',
    
    # Conversion
    'rgb_to_hsl',
    'hsl_to_rgb',
    'hex_to_hsl',
    'hue_distance',
    
    # Classification
    'is_neutral',
    'get_color_temperature',
    'get_color_season',
    'classify_saturation',
    'classify_lightness',
    'closest_color_name',
    
    # Harmony
    'get_complementary',
    'get_analogous',
    'get_triadic',
    'get_split_complementary',
    'get_tetradic',
    'check_hue_harmony',
    
    # Compatibility
    'saturation_compatible',
    'temperature_compatible',
    'lightness_contrast_score',
    'season_compatible',
    'calculate_color_compatibility',
    
    # 60-30-10
    'assign_color_roles',
    'suggest_role_for_garment',
    
    # High-level
    'find_complementary_colors',
    'get_safe_combinations',
    'get_outfit_colors',
    
    # Bold/Safe balanced
    'get_bold_complementary',
    'get_safe_colors_for_category',
    'get_balanced_colors_for_category',
]