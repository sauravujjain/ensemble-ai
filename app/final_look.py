"""
Final Look Module for Ensemble AI

Creates:
1. Final outfit collage from user-selected items
2. Shopping prompt for external AI assistants (ChatGPT, Gemini, Perplexity)

Author: Fashion Ensemble Builder
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap


# =============================================================================
# FINAL LOOK COLLAGE
# =============================================================================

def create_final_look_collage(
    selected_items: Dict[str, List[Dict]],
    anchor_item: Optional[Dict] = None,
    parsed_query: Optional[Dict] = None,
    image_size: Tuple[int, int] = (320, 400),
    anchor_size: Tuple[int, int] = (360, 450),
) -> Image.Image:
    """
    Create a clean 'outfit card' collage from user-selected items.
    
    Args:
        selected_items: Dict with keys 'top', 'bottom', 'accessory', 'coordinating', each containing
                       list of item dicts with 'path', 'query', 'description'
        anchor_item: Dict with 'path' and optional 'description' for anchor piece
        parsed_query: Parsed user query with occasion, style, gender, etc.
        image_size: Size for ensemble items (width, height)
        anchor_size: Size for anchor piece (width, height)
    
    Returns:
        PIL Image of the collage
    """
    # Safely handle None parsed_query
    if parsed_query is None:
        parsed = {}
    else:
        parsed = parsed_query
    
    # Collect all items to display
    all_items = []
    
    # Add anchor first if present
    if anchor_item and anchor_item.get('path'):
        anchor_color = parsed.get('anchor_color') or ''
        anchor_garment = parsed.get('anchor_garment') or ''
        all_items.append({
            'path': anchor_item['path'],
            'label': 'YOUR PIECE',
            'sublabel': f"{anchor_color} {anchor_garment}".strip().title(),
            'is_anchor': True,
            'is_bold': False
        })
    
    # Add selected items by category
    category_order = ['coordinating', 'top', 'bottom', 'accessory']
    category_labels = {'top': 'TOP', 'bottom': 'BOTTOM', 'accessory': 'ACCESSORY', 'coordinating': 'PAIRING'}
    
    for category in category_order:
        items = selected_items.get(category, [])
        for i, item in enumerate(items):
            query = item.get('query', {})
            color = query.get('color', '')
            garment = query.get('garment_type', '')
            is_bold = query.get('is_bold', False)
            
            all_items.append({
                'path': item.get('path'),
                'label': f"{category_labels.get(category, category.upper())} {i+1}",
                'sublabel': f"{color} {garment}".strip().title()[:25],
                'is_anchor': False,
                'is_bold': is_bold,
                'description': item.get('description', '')
            })
    
    if not all_items:
        # Return empty placeholder
        empty = Image.new('RGB', (400, 300), color='#f5f5f5')
        draw = ImageDraw.Draw(empty)
        draw.text((100, 140), "No items selected", fill='#999999')
        return empty
    
    # Layout calculation - larger for better visibility
    padding = 30
    header_height = 100
    item_label_height = 70
    
    # Determine grid layout
    n_items = len(all_items)
    
    # For 1-4 items: single row
    # For 5-7 items: two rows (4 top, rest bottom centered)
    if n_items <= 4:
        cols = n_items
        rows = 1
    else:
        cols = 4
        rows = 2
    
    # Calculate canvas size
    item_width = image_size[0]
    item_height = image_size[1]
    
    # Anchor gets special treatment if present
    has_anchor = any(item.get('is_anchor') for item in all_items)
    
    canvas_width = padding + cols * (item_width + padding)
    canvas_height = header_height + rows * (item_height + item_label_height + padding) + padding
    
    # Minimum width for header
    canvas_width = max(canvas_width, 900)
    
    # Create canvas with light background
    collage = Image.new('RGB', (int(canvas_width), int(canvas_height)), color='#FFFFFF')
    draw = ImageDraw.Draw(collage)
    
    # Try to load a nicer font, fall back to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Draw header background
    draw.rectangle([0, 0, canvas_width, header_height], fill='#667eea')
    
    # Title - handle None values safely
    occasion = (parsed.get('occasion') or 'casual').title()
    gender = parsed.get('gender') or 'womens'
    gender_display = "Women's" if gender == 'womens' else "Men's" if gender == 'mens' else ""
    
    title = f"👗 {gender_display} {occasion} Ensemble".strip()
    draw.text((padding, 25), title, fill='white', font=title_font)
    
    # Subtitle with style info
    style = (parsed.get('style') or 'western').title()
    subtitle = f"Style: {style} | Items: {n_items}"
    draw.text((padding, 65), subtitle, fill='#E8E8FF', font=subtitle_font)
    
    # Draw items
    y_start = header_height + padding
    
    # First row
    row1_items = all_items[:4]
    row1_start_x = (canvas_width - len(row1_items) * (item_width + padding) + padding) // 2
    
    for i, item in enumerate(row1_items):
        x = row1_start_x + i * (item_width + padding)
        y = y_start
        
        # Draw item
        _draw_item(collage, draw, item, x, y, item_width, item_height, 
                   label_font, subtitle_font)
    
    # Second row (if needed)
    if n_items > 4:
        row2_items = all_items[4:]
        row2_start_x = (canvas_width - len(row2_items) * (item_width + padding) + padding) // 2
        
        for i, item in enumerate(row2_items):
            x = row2_start_x + i * (item_width + padding)
            y = y_start + item_height + item_label_height + padding
            
            _draw_item(collage, draw, item, x, y, item_width, item_height,
                       label_font, subtitle_font)
    
    # Footer
    footer_y = canvas_height - 30
    draw.text((padding, footer_y), "Created with Ensemble AI", fill='#CCCCCC', font=label_font)
    
    return collage


def _draw_item(
    collage: Image.Image,
    draw: ImageDraw.Draw,
    item: Dict,
    x: int,
    y: int,
    width: int,
    height: int,
    label_font,
    subtitle_font
):
    """Helper to draw a single item in the collage."""
    
    path = item.get('path')
    is_anchor = item.get('is_anchor', False)
    is_bold = item.get('is_bold', False)
    
    # Draw image border/background
    border_color = '#667eea' if is_anchor else '#FFD700' if is_bold else '#E0E0E0'
    border_width = 3 if is_anchor or is_bold else 1
    
    # Draw border rectangle
    draw.rectangle(
        [x - border_width, y - border_width, 
         x + width + border_width, y + height + border_width],
        outline=border_color,
        width=border_width
    )
    
    # Load and paste image
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((width, height), Image.LANCZOS)
        collage.paste(img, (int(x), int(y)))
    except Exception as e:
        # Draw placeholder
        draw.rectangle([x, y, x + width, y + height], fill='#F5F5F5')
        draw.text((x + 10, y + height//2), "Image\nUnavailable", fill='#999999')
    
    # Draw labels below image
    label_y = y + height + 8
    
    # Label (category or "YOUR PIECE")
    label = item.get('label', '')
    if is_anchor:
        draw.text((x, label_y), "⭐ " + label, fill='#667eea', font=label_font)
    elif is_bold:
        draw.text((x, label_y), "⚡ " + label, fill='#D4AF37', font=label_font)
    else:
        draw.text((x, label_y), label, fill='#666666', font=label_font)
    
    # Sublabel (color + garment type)
    sublabel = item.get('sublabel', '')
    if sublabel:
        draw.text((x, label_y + 20), sublabel[:25], fill='#333333', font=subtitle_font)
    
    # Bold indicator
    if is_bold and not is_anchor:
        draw.text((x, label_y + 42), "Statement piece", fill='#D4AF37', font=label_font)


# =============================================================================
# SHOPPING PROMPT GENERATOR
# =============================================================================

def generate_shopping_prompt(
    selected_items: Dict[str, List[Dict]],
    anchor_item: Optional[Dict] = None,
    parsed_query: Optional[Dict] = None,
    include_retailers: bool = True
) -> str:
    """
    Generate a detailed shopping prompt for external AI assistants.
    
    Args:
        selected_items: Dict with 'top', 'bottom', 'accessory' lists
        anchor_item: Anchor piece info
        parsed_query: Parsed user query
        include_retailers: Whether to suggest specific retailers
    
    Returns:
        Formatted prompt string for ChatGPT/Gemini/Perplexity
    """
    parsed = parsed_query or {}
    
    # Extract context - handle None values safely
    occasion = (parsed.get('occasion') or 'casual').title()
    style = (parsed.get('style') or 'western').title()
    gender = parsed.get('gender') or 'womens'
    age_group = parsed.get('age_group') or 'adult'
    anchor_color = parsed.get('anchor_color') or ''
    anchor_garment = parsed.get('anchor_garment') or ''
    
    # Build gender/age display
    if age_group == 'kids':
        demographic = "Kids'"
    elif gender == 'womens':
        demographic = "Women's"
    elif gender == 'mens':
        demographic = "Men's"
    else:
        demographic = ""
    
    # Start building prompt
    lines = []
    
    # Header
    lines.append("=" * 50)
    lines.append("🛍️ SHOPPING RECOMMENDATION REQUEST")
    lines.append("=" * 50)
    lines.append("")
    
    # Context section
    lines.append("**OUTFIT CONTEXT:**")
    lines.append(f"• Demographic: {demographic}")
    lines.append(f"• Occasion: {occasion}")
    lines.append(f"• Style: {style}")
    if anchor_color and anchor_garment:
        lines.append(f"• Building around: {anchor_color.title()} {anchor_garment.title()}")
    lines.append("")
    
    # Items section
    lines.append("**ITEMS I'M LOOKING FOR:**")
    lines.append("")
    
    item_count = 1
    
    # Anchor piece (if they need to find a similar one)
    if anchor_item:
        lines.append(f"{item_count}. **ANCHOR PIECE** (already own or looking for similar)")
        lines.append(f"   • Type: {anchor_garment.title()}")
        if anchor_color:
            lines.append(f"   • Color: {anchor_color.title()}")
        lines.append("")
        item_count += 1
    
    # Category items
    category_names = {
        'top': 'TOP/BLOUSE/SHIRT',
        'bottom': 'BOTTOM/PANTS/SKIRT', 
        'accessory': 'ACCESSORY',
        'coordinating': 'COORDINATING PIECE'
    }
    
    for category in ['coordinating', 'top', 'bottom', 'accessory']:
        items = selected_items.get(category, [])
        for item in items:
            query = item.get('query', {})
            color = query.get('color', 'neutral')
            garment = query.get('garment_type', 'item')
            is_bold = query.get('is_bold', False)
            description = item.get('description', '')
            
            bold_tag = " ⚡ (STATEMENT PIECE)" if is_bold else ""
            
            lines.append(f"{item_count}. **{color.upper()} {garment.upper()}**{bold_tag}")
            lines.append(f"   • Category: {category_names.get(category, category.upper())}")
            lines.append(f"   • Color: {color.title()}")
            lines.append(f"   • Style: {garment.title()}")
            if description:
                lines.append(f"   • Vibe: {description}")
            if is_bold:
                lines.append(f"   • Note: This is a color-theory complementary piece for visual interest")
            lines.append("")
            item_count += 1
    
    # Requirements section
    lines.append("**WHAT I NEED FROM YOU:**")
    lines.append("")
    lines.append("1. **Specific product recommendations** for each item above")
    lines.append("2. **Direct shopping links** where I can buy them")
    lines.append("3. **Price range options**: Budget-friendly + Premium choices")
    lines.append("4. **Styling tips** for putting the outfit together")
    lines.append("")
    
    # Retailers section
    if include_retailers:
        lines.append("**PREFERRED RETAILERS (if available):**")
        lines.append("• Global: Amazon, H&M, Zara, ASOS, Uniqlo, Nordstrom")
        lines.append("• Fashion: Mango, & Other Stories, COS, Massimo Dutti")
        lines.append("• Premium: NET-A-PORTER, SSENSE, Matches Fashion")
        lines.append("• Any other reputable online fashion retailers")
        lines.append("")
    
    # Image reference note
    lines.append("**NOTE:** I've attached an image showing the outfit concept.")
    lines.append("Please use both this description AND the image for recommendations.")
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)


def generate_shopping_prompt_compact(
    selected_items: Dict[str, List[Dict]],
    parsed_query: Optional[Dict] = None,
) -> str:
    """
    Generate a shorter, more conversational shopping prompt.
    Alternative to the detailed version.
    """
    parsed = parsed_query or {}
    
    # Handle None values safely
    occasion = parsed.get('occasion') or 'casual'
    style = parsed.get('style') or 'western'
    gender = parsed.get('gender') or 'womens'
    anchor_color = parsed.get('anchor_color') or ''
    anchor_garment = parsed.get('anchor_garment') or ''
    
    gender_text = "women's" if gender == 'womens' else "men's" if gender == 'mens' else ""
    
    # Build items list
    items_list = []
    
    for category in ['coordinating', 'top', 'bottom', 'accessory']:
        for item in selected_items.get(category, []):
            query = item.get('query', {})
            color = query.get('color', '')
            garment = query.get('garment_type', '')
            is_bold = query.get('is_bold', False)
            
            if is_bold:
                items_list.append(f"• {color.title()} {garment} (statement piece)")
            else:
                items_list.append(f"• {color.title()} {garment}")
    
    items_text = "\n".join(items_list)
    
    prompt = f"""Hi! I'm building a {gender_text} {occasion} outfit in a {style} style.

I'm looking for shopping recommendations for these items:

{items_text}

{"I already have a " + anchor_color + " " + anchor_garment + " that I'm building this outfit around." if anchor_color and anchor_garment else ""}

Can you suggest specific products with links from major retailers (Amazon, H&M, Zara, ASOS, etc.)? 
Please include both budget-friendly and premium options.

I've attached an image showing what I'm going for - please use it as reference!"""

    return prompt


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'create_final_look_collage',
    'generate_shopping_prompt',
    'generate_shopping_prompt_compact',
]