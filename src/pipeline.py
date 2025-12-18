"""
Fashion Ensemble Pipeline

Complete pipeline for:
1. LLM query parsing
2. Color theory integration
3. Category-aware ensemble building
4. FAISS retrieval
5. Collage generation

Author: Fashion Ensemble Builder
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# These will be imported in notebook after loading
# import ollama
# from color_theory import get_safe_combinations, find_complementary_colors
# from ensemble_rules import build_ensemble_queries, build_ensemble_queries_extended, suggest_outfit_structure


# =============================================================================
# LLM SYSTEM PROMPT
# =============================================================================

FASHION_SYSTEM_PROMPT = """You are a fashion styling assistant. Your job is to parse user requests and output structured JSON for outfit building.

Given a user's fashion request, extract:
1. intent: "single_item" (find similar) or "ensemble" (complete outfit)
2. anchor_garment: the main piece mentioned (kurta, dress, jeans, blazer, etc.)
3. anchor_color: color of the main piece (if mentioned)
4. occasion: event type (daily, festive, wedding, party, work, casual, date, brunch, office)
5. style: style preference (ethnic, western, casual, formal, fusion)
6. specific_requests: any specific items user wants (e.g., "matching palazzos")

RESPOND WITH ONLY VALID JSON. No explanations, no markdown.

Example input: "I have a yellow kurta, need a complete festive look"
Example output:
{"intent": "ensemble", "anchor_garment": "kurta", "anchor_color": "yellow", "occasion": "festive", "style": "ethnic", "specific_requests": []}

Example input: "navy blazer for office, need complete outfit"
Example output:
{"intent": "ensemble", "anchor_garment": "blazer", "anchor_color": "navy", "occasion": "work", "style": "formal", "specific_requests": []}
"""

STYLE_DESCRIPTOR_PROMPT = """You are a fashion stylist. Generate SHORT style descriptors (max 6 words) for outfit items.

Context:
- Anchor piece: {anchor_color} {anchor_garment}
- Occasion: {occasion}
- Overall style: {style}

For each item below, provide a concise descriptor explaining why it works with the anchor:

Items:
{items_list}

Output JSON array format:
[
  {{"item": "white shirt", "is_bold": false, "descriptor": "Timeless professional choice"}},
  {{"item": "coral pants", "is_bold": true, "descriptor": "Bold complementary statement"}}
]

Rules:
- Safe items (is_bold=false): Emphasize versatility, classic appeal, everyday wearability
- Bold items (is_bold=true): Highlight fashion-forward choice, color theory, confidence
- Max 6 words per descriptor
- Be specific to the occasion
- NO generic phrases like "looks good" or "nice option"
- Start with adjective or action word

RESPOND WITH ONLY JSON ARRAY. No explanations.
"""


# =============================================================================
# LLM QUERY PARSER
# =============================================================================

def parse_user_query(user_text: str, ollama_model: str = 'phi3') -> dict:
    """Use LLM to parse user fashion query into structured format."""
    import ollama
    
    try:
        response = ollama.chat(
            model=ollama_model,
            messages=[
                {'role': 'system', 'content': FASHION_SYSTEM_PROMPT},
                {'role': 'user', 'content': user_text}
            ],
            options={'temperature': 0.1, 'num_predict': 200}
        )
        
        content = response['message']['content'].strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {'success': True, 'data': parsed}
        else:
            return {'success': False, 'error': 'No JSON found', 'raw': content}
            
    except json.JSONDecodeError as e:
        return {'success': False, 'error': f'JSON parse error: {e}', 'raw': content}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def parse_user_query_robust(user_text: str, fallback_func=None, max_retries: int = 2) -> dict:
    """Robust LLM parser with retry and fallback."""
    import ollama
    
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model='phi3',
                messages=[
                    {'role': 'system', 'content': FASHION_SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_text}
                ],
                options={'temperature': 0.1, 'num_predict': 200}
            )
            
            content = response['message']['content'].strip()
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                
                # Validate required fields
                required = ['intent', 'anchor_garment']
                if all(k in parsed for k in required):
                    return {'success': True, 'data': parsed}
            
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            
    # Fallback: use rule-based parsing
    print(f"  [Fallback to rule-based parsing]")
    
    if fallback_func:
        fallback = fallback_func(user_text)
    else:
        fallback = {'needs_full_ensemble': True, 'detected_garment': 'top'}
    
    return {
        'success': True,
        'data': {
            'intent': 'ensemble' if fallback.get('needs_full_ensemble') else 'single_item',
            'anchor_garment': fallback.get('detected_garment', 'top'),
            'anchor_color': extract_color_from_text(user_text),
            'occasion': fallback.get('detected_occasion'),
            'style': fallback.get('detected_style'),
            'specific_requests': []
        },
        'fallback': True
    }


def extract_color_from_text(text: str) -> str:
    """Extract color from text using keyword matching."""
    colors = [
        'red', 'blue', 'yellow', 'green', 'black', 'white', 'pink', 
        'navy', 'maroon', 'gold', 'grey', 'gray', 'beige', 'orange', 
        'purple', 'cream', 'brown', 'teal', 'coral', 'mustard',
        'burgundy', 'olive', 'peach', 'lavender', 'mint'
    ]
    text_lower = text.lower()
    for color in colors:
        if color in text_lower:
            return color
    return 'neutral'


# =============================================================================
# LLM STYLE DESCRIPTORS
# =============================================================================

def generate_style_descriptors(
    outfit_results: Dict,
    ollama_model: str = 'phi3'
) -> Dict[str, str]:
    """
    Use LLM to generate contextual style descriptors for each item.
    
    Args:
        outfit_results: Results from build_outfit_pipeline
        ollama_model: Ollama model to use
    
    Returns:
        Dict mapping "category_index" to descriptor string
    """
    import ollama
    
    parsed = outfit_results.get('parsed', {})
    anchor_color = parsed.get('anchor_color', 'neutral')
    anchor_garment = parsed.get('anchor_garment', 'item')
    occasion = parsed.get('occasion', 'casual')
    style = parsed.get('style', 'versatile')
    
    # Build items list for LLM
    items_list = []
    item_keys = []
    
    for item in outfit_results.get('ensemble_items', []):
        query = item['query']
        color = query.get('color', 'neutral')
        garment = query.get('garment_type', 'item')
        is_bold = query.get('is_bold', False)
        category = query.get('garment_category', 'item')
        
        items_list.append(f"- {color} {garment} (is_bold: {is_bold})")
        item_keys.append(f"{category}_{color}_{garment}")
    
    if not items_list:
        return {}
    
    # Build prompt
    prompt = STYLE_DESCRIPTOR_PROMPT.format(
        anchor_color=anchor_color,
        anchor_garment=anchor_garment,
        occasion=occasion,
        style=style,
        items_list='\n'.join(items_list)
    )
    
    try:
        response = ollama.chat(
            model=ollama_model,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            options={'temperature': 0.3, 'num_predict': 500}
        )
        
        content = response['message']['content'].strip()
        
        # Extract JSON array
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            descriptors_list = json.loads(json_match.group())
            
            # Map back to our keys
            result = {}
            for i, desc in enumerate(descriptors_list):
                if i < len(item_keys):
                    key = item_keys[i]
                    result[key] = {
                        'descriptor': desc.get('descriptor', ''),
                        'is_bold': desc.get('is_bold', False)
                    }
            
            return result
    
    except Exception as e:
        print(f"  [LLM descriptor error: {e}]")
    
    # Fallback: generate basic descriptors
    return generate_fallback_descriptors(outfit_results)


def generate_fallback_descriptors(outfit_results: Dict) -> Dict[str, str]:
    """Generate basic descriptors without LLM."""
    
    SAFE_DESCRIPTORS = {
        'top': [
            "Versatile office essential",
            "Classic professional choice",
            "Everyday wardrobe staple",
            "Clean, polished look"
        ],
        'bottom': [
            "Reliable wardrobe foundation",
            "Goes with everything",
            "Timeless neutral base",
            "Professional everyday wear"
        ],
        'outerwear': [
            "Polished finishing layer",
            "Elevates any outfit",
            "Smart layering piece",
            "Professional outer layer"
        ]
    }
    
    BOLD_DESCRIPTORS = {
        'top': "⚡ Statement piece - color pop",
        'bottom': "⚡ Bold choice - confident style",
        'outerwear': "⚡ Fashion-forward accent"
    }
    
    result = {}
    category_counts = {}
    
    for item in outfit_results.get('ensemble_items', []):
        query = item['query']
        color = query.get('color', 'neutral')
        garment = query.get('garment_type', 'item')
        is_bold = query.get('is_bold', False)
        category = query.get('garment_category', 'item')
        
        key = f"{category}_{color}_{garment}"
        
        if is_bold:
            descriptor = BOLD_DESCRIPTORS.get(category, "⚡ Bold statement")
        else:
            count = category_counts.get(category, 0)
            descriptors = SAFE_DESCRIPTORS.get(category, ["Versatile choice"])
            descriptor = descriptors[count % len(descriptors)]
            category_counts[category] = count + 1
        
        result[key] = {
            'descriptor': descriptor,
            'is_bold': is_bold
        }
    
    return result


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def build_outfit_pipeline(
    user_text: str,
    anchor_image_path: str = None,
    k: int = 5,
    use_balanced_queries: bool = True,
    generate_descriptors: bool = True,
    # Injected dependencies
    index=None,
    valid_paths=None,
    model=None,
    tokenizer=None,
    preprocess=None,
    device=None,
    search_by_text_func=None,
    get_safe_combinations_func=None,
    build_queries_func=None,
    build_balanced_queries_func=None,
    suggest_structure_func=None
) -> Dict:
    """
    Full LLM-orchestrated pipeline for outfit building.
    
    Returns results grouped by category with multiple items each.
    
    Args:
        user_text: User's fashion request
        anchor_image_path: Optional path to anchor image
        k: Number of results per query
        use_balanced_queries: Use 3 safe + 1 bold per category
        generate_descriptors: Generate LLM style descriptors
    """
    
    print("="*60)
    print(f"User: '{user_text}'")
    print("="*60)
    
    # Step 1: LLM Parse
    print("\n[1] LLM Parsing...")
    parsed = parse_user_query_robust(user_text, fallback_func=suggest_structure_func)
    data = parsed['data']
    print(f"    Intent: {data['intent']}")
    print(f"    Anchor: {data.get('anchor_color', 'unknown')} {data['anchor_garment']}")
    print(f"    Occasion: {data.get('occasion')}")
    print(f"    Style: {data.get('style')}")
    
    # Step 2: Get complementary colors (for fallback)
    print("\n[2] Color Theory...")
    anchor_color = data.get('anchor_color', 'white')
    if anchor_color == 'neutral':
        anchor_color = 'white'
    complementary = get_safe_combinations_func(anchor_color)
    print(f"    Complementary colors for {anchor_color}: {complementary}")
    
    # Step 3: Build ensemble queries
    print("\n[3] Building Queries...")
    
    if use_balanced_queries and build_balanced_queries_func:
        queries = build_balanced_queries_func(
            anchor_garment=data['anchor_garment'],
            anchor_color=anchor_color,
            occasion=data.get('occasion'),
            style=data.get('style'),
            n_safe=3,
            n_bold=1
        )
    else:
        queries = build_queries_func(
            anchor_garment=data['anchor_garment'],
            anchor_color=anchor_color,
            occasion=data.get('occasion'),
            style=data.get('style'),
            complementary_colors=complementary[:6]
        )
    
    for q in queries:
        bold_tag = "⚡" if q.get('is_bold') else "  "
        print(f"    {bold_tag} '{q['search_text']}' ({q['garment_category']})")
    
    # Step 4: FAISS Search
    print("\n[4] Searching FAISS...")
    results = {
        'parsed': data,
        'anchor_image': anchor_image_path,
        'ensemble_items': [],
        'by_category': {}
    }
    
    for q in queries:
        distances, indices = search_by_text_func(
            q['search_text'], index, valid_paths, model, tokenizer, device, k=k
        )
        
        item_results = {
            'query': q,
            'results': [
                {'path': valid_paths[i], 'score': float(d)} 
                for i, d in zip(indices, distances)
            ]
        }
        
        results['ensemble_items'].append(item_results)
        
        # Group by category
        category = q['garment_category']
        if category not in results['by_category']:
            results['by_category'][category] = []
        
        # Add query info to each result
        for r in item_results['results']:
            r['query'] = q
        
        results['by_category'][category].extend(item_results['results'])
        
        print(f"    '{q['search_text']}' → {len(indices)} results")
    
    # Deduplicate within each category (keep highest score)
    for category in results['by_category']:
        seen_paths = set()
        unique_results = []
        for r in sorted(results['by_category'][category], key=lambda x: -x['score']):
            if r['path'] not in seen_paths:
                seen_paths.add(r['path'])
                unique_results.append(r)
        results['by_category'][category] = unique_results
    
    # Step 5: Generate style descriptors
    if generate_descriptors:
        print("\n[5] Generating Style Descriptors...")
        try:
            results['descriptors'] = generate_style_descriptors(results)
            print(f"    Generated {len(results['descriptors'])} descriptors")
        except Exception as e:
            print(f"    Fallback descriptors (LLM error: {e})")
            results['descriptors'] = generate_fallback_descriptors(results)
    
    print("\n[6] Done!")
    return results


# =============================================================================
# COLLAGE GENERATION
# =============================================================================

def create_ensemble_collage(
    outfit_results: Dict,
    anchor_image_path: str = None,
    items_per_category: int = 4,
    image_size: Tuple[int, int] = (180, 240),
    show_descriptors: bool = True
) -> Image.Image:
    """
    Create a collage with anchor + items grouped by category.
    Includes LLM-generated style descriptors.
    
    Layout:
    [ANCHOR] | [TOP 1] [TOP 2] [TOP 3] [TOP 4 ⚡]
             | [BOT 1] [BOT 2] [BOT 3] [BOT 4 ⚡]
    """
    parsed = outfit_results.get('parsed', {})
    by_category = outfit_results.get('by_category', {})
    descriptors = outfit_results.get('descriptors', {})
    
    # Use provided anchor or from results
    if anchor_image_path is None:
        anchor_image_path = outfit_results.get('anchor_image')
    
    # Layout calculations
    padding = 15
    label_height = 45 if show_descriptors else 25
    category_label_width = 100
    anchor_size = (200, 280)
    
    n_categories = len(by_category)
    if n_categories == 0:
        n_categories = 1
    
    # Canvas size
    has_anchor = anchor_image_path is not None
    
    canvas_width = (
        (anchor_size[0] + padding * 2 if has_anchor else 0) +
        category_label_width +
        items_per_category * (image_size[0] + padding) + padding
    )
    
    canvas_height = (
        60 +  # Title
        n_categories * (image_size[1] + label_height + padding) + padding
    )
    
    # Ensure minimum height for anchor
    if has_anchor:
        canvas_height = max(canvas_height, anchor_size[1] + 120)
    
    collage = Image.new('RGB', (int(canvas_width), int(canvas_height)), color='white')
    draw = ImageDraw.Draw(collage)
    
    # Title
    title = f"OUTFIT: {parsed.get('anchor_color', '')} {parsed.get('anchor_garment', '')} | {parsed.get('occasion', 'casual')}".upper().strip()
    draw.text((padding, 15), title, fill='black')
    
    y_offset = 55
    x_items_start = padding
    
    # Draw anchor if provided
    if has_anchor:
        try:
            anchor_img = Image.open(anchor_image_path).convert('RGB')
            anchor_img = anchor_img.resize(anchor_size, Image.LANCZOS)
            collage.paste(anchor_img, (padding, y_offset))
            draw.text((padding, y_offset + anchor_size[1] + 5), "YOUR ITEM", fill='darkgreen')
        except Exception as e:
            print(f"Error loading anchor: {e}")
        
        x_items_start = padding + anchor_size[0] + padding
    
    # Draw items by category
    category_y = y_offset
    
    category_display_names = {
        'top': 'TOPS',
        'bottom': 'BOTTOMS',
        'outerwear': 'LAYERS'
    }
    
    for category, items in by_category.items():
        # Category label
        cat_label = category_display_names.get(category, category.upper())
        draw.text(
            (x_items_start, category_y + image_size[1] // 2 - 10),
            cat_label,
            fill='grey'
        )
        
        # Draw items (up to items_per_category)
        x = x_items_start + category_label_width
        
        for i, item in enumerate(items[:items_per_category]):
            try:
                img = Image.open(item['path']).convert('RGB')
                img = img.resize(image_size, Image.LANCZOS)
                collage.paste(img, (int(x), int(category_y)))
                
                # Get query info if available
                query = item.get('query', {})
                color = query.get('color', '')
                garment = query.get('garment_type', '')
                is_bold = query.get('is_bold', False)
                
                # Build label
                item_label = f"{color} {garment}".strip().title()
                
                # Get descriptor
                desc_key = f"{category}_{color}_{garment}"
                desc_info = descriptors.get(desc_key, {})
                descriptor = desc_info.get('descriptor', '') if show_descriptors else ''
                
                # Draw labels
                text_y = int(category_y + image_size[1] + 3)
                
                if is_bold:
                    # Bold item - highlight
                    draw.text((int(x), text_y), "⚡ BOLD", fill='#D4AF37')  # Gold color
                    if descriptor:
                        draw.text((int(x), text_y + 12), descriptor[:22], fill='#666666')
                else:
                    # Regular item
                    if descriptor:
                        draw.text((int(x), text_y), descriptor[:25], fill='#666666')
                    else:
                        draw.text((int(x), text_y), f"{item['score']:.2f}", fill='grey')
                
            except Exception as e:
                print(f"Error loading image: {e}")
            
            x += image_size[0] + padding
        
        category_y += image_size[1] + label_height + padding
    
    return collage


def create_simple_collage(
    outfit_results: Dict,
    anchor_image_path: str = None,
    picks_per_query: int = 1,
    image_size: Tuple[int, int] = (200, 280)
) -> Image.Image:
    """
    Create a simple collage showing one pick per query.
    
    Layout: [ANCHOR] [PICK1] [PICK2] [PICK3] [PICK4]
    """
    parsed = outfit_results.get('parsed', {})
    ensemble_items = outfit_results.get('ensemble_items', [])
    descriptors = outfit_results.get('descriptors', {})
    
    # Collect picks
    selected = []
    labels = []
    query_info = []
    
    for item in ensemble_items:
        query = item['query']
        results = item['results'][:picks_per_query]
        for r in results:
            selected.append(r['path'])
            labels.append(f"{query['color']} {query['garment_type']}")
            query_info.append(query)
    
    # Layout
    padding = 15
    label_height = 50
    anchor_size = (220, 300)
    
    has_anchor = anchor_image_path is not None
    n_items = len(selected)
    
    canvas_width = (
        (anchor_size[0] + padding if has_anchor else 0) +
        n_items * (image_size[0] + padding) + padding
    )
    canvas_height = max(anchor_size[1] if has_anchor else 0, image_size[1]) + label_height + 80
    
    collage = Image.new('RGB', (int(canvas_width), int(canvas_height)), color='white')
    draw = ImageDraw.Draw(collage)
    
    # Title
    title = f"OUTFIT: {parsed.get('anchor_color', '')} {parsed.get('anchor_garment', '')}".upper().strip()
    draw.text((padding, 15), title, fill='black')
    
    y_offset = 50
    x = padding
    
    # Anchor
    if has_anchor:
        try:
            anchor_img = Image.open(anchor_image_path).convert('RGB')
            anchor_img = anchor_img.resize(anchor_size, Image.LANCZOS)
            collage.paste(anchor_img, (x, y_offset))
            draw.text((x, y_offset + anchor_size[1] + 5), "YOUR ITEM", fill='darkgreen')
        except:
            pass
        x += anchor_size[0] + padding
    
    # Items
    for img_path, label, query in zip(selected, labels, query_info):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(image_size, Image.LANCZOS)
            collage.paste(img, (int(x), y_offset))
            
            # Labels
            is_bold = query.get('is_bold', False)
            category = query.get('garment_category', '')
            color = query.get('color', '')
            garment = query.get('garment_type', '')
            
            text_y = y_offset + image_size[1] + 5
            
            # Get descriptor
            desc_key = f"{category}_{color}_{garment}"
            desc_info = descriptors.get(desc_key, {})
            descriptor = desc_info.get('descriptor', '')
            
            if is_bold:
                draw.text((int(x), text_y), "⚡ BOLD", fill='#D4AF37')
                if descriptor:
                    draw.text((int(x), text_y + 15), descriptor[:20], fill='#666666')
            else:
                label_short = label[:18].title()
                draw.text((int(x), text_y), label_short, fill='grey')
                if descriptor:
                    draw.text((int(x), text_y + 15), descriptor[:20], fill='#666666')
        except:
            pass
        
        x += image_size[0] + padding
    
    return collage


# =============================================================================
# PROMPT GENERATOR FOR EXTERNAL AI
# =============================================================================

def generate_shopping_prompt(outfit_results: Dict, anchor_description: str = None) -> str:
    """
    Generate a prompt for external LLM/shopping assistant.
    
    Can be used with ChatGPT, Claude, or shopping sites.
    """
    parsed = outfit_results.get('parsed', {})
    by_category = outfit_results.get('by_category', {})
    
    anchor_color = parsed.get('anchor_color', 'neutral')
    anchor_garment = parsed.get('anchor_garment', 'item')
    occasion = parsed.get('occasion', 'casual')
    style = parsed.get('style', 'versatile')
    
    prompt_parts = [
        f"I have a {anchor_color} {anchor_garment} and I'm looking for a complete {occasion} outfit.",
        f"Style preference: {style}.",
        "",
        "I need recommendations for:"
    ]
    
    for category, items in by_category.items():
        colors_seen = []
        for item in items[:4]:
            # Extract info from query if available
            pass
        
        if category == 'top':
            prompt_parts.append(f"- TOPS: Shirts, blouses, or tops that complement {anchor_color}")
        elif category == 'bottom':
            prompt_parts.append(f"- BOTTOMS: Pants, trousers, or skirts in complementary colors")
        elif category == 'outerwear':
            prompt_parts.append(f"- OUTERWEAR: Jackets or blazers for layering")
    
    prompt_parts.extend([
        "",
        "Please suggest specific items with:",
        "1. Color recommendations that complement my anchor piece",
        "2. Fabric suggestions appropriate for the occasion",
        "3. Style tips for putting the outfit together",
        "4. Where I might find these items (brands/stores)"
    ])
    
    return "\n".join(prompt_parts)


def generate_image_gen_prompt(outfit_results: Dict) -> str:
    """
    Generate a prompt for AI image generation (Midjourney, DALL-E, etc.)
    
    Creates a detailed description of the complete outfit.
    """
    parsed = outfit_results.get('parsed', {})
    by_category = outfit_results.get('by_category', {})
    
    anchor_color = parsed.get('anchor_color', '')
    anchor_garment = parsed.get('anchor_garment', '')
    occasion = parsed.get('occasion', 'casual')
    style = parsed.get('style', '')
    
    # Build outfit description
    outfit_parts = [f"{anchor_color} {anchor_garment}"]
    
    category_templates = {
        'top': 'complementary {color} top',
        'bottom': 'matching {color} bottoms',
        'outerwear': '{color} layering piece'
    }
    
    colors_used = set()
    for category, items in by_category.items():
        if items and category in category_templates:
            # Get the top-scored item's color from query
            first_query = outfit_results['ensemble_items'][0]['query'] if outfit_results.get('ensemble_items') else {}
            color = first_query.get('color', 'neutral')
            if color not in colors_used:
                outfit_parts.append(category_templates[category].format(color=color))
                colors_used.add(color)
    
    prompt = f"""Fashion photograph of a stylish {occasion} outfit:
    
Main pieces: {', '.join(outfit_parts)}

Style: {style if style else 'modern and sophisticated'}
Occasion: {occasion}

Photography style: Clean white background, professional fashion photography, 
full body shot, model wearing the complete outfit, high-end editorial look,
soft natural lighting, 8k resolution, fashion magazine quality."""
    
    return prompt


# =============================================================================
# FIXED COLLAGE (1 result per query, not pooled)
# =============================================================================

def create_ensemble_collage_fixed(
    outfit_results: Dict,
    anchor_image_path: str = None,
    image_size: Tuple[int, int] = (180, 240),
    show_descriptors: bool = True
) -> Image.Image:
    """
    Fixed collage: Show 1 best result per query (not pooled by category).
    
    This ensures diversity - each query slot shows its own best result,
    rather than pooling all results and showing top-N which may be similar.
    """
    parsed = outfit_results.get('parsed', {})
    ensemble_items = outfit_results.get('ensemble_items', [])
    descriptors = outfit_results.get('descriptors', {})
    
    if anchor_image_path is None:
        anchor_image_path = outfit_results.get('anchor_image')
    
    # Group queries by category
    queries_by_category = {}
    for item in ensemble_items:
        cat = item['query']['garment_category']
        if cat not in queries_by_category:
            queries_by_category[cat] = []
        queries_by_category[cat].append(item)
    
    # Layout
    padding = 15
    label_height = 50
    category_label_width = 100
    anchor_size = (200, 280)
    
    n_categories = len(queries_by_category)
    items_per_category = max(len(v) for v in queries_by_category.values()) if queries_by_category else 4
    
    has_anchor = anchor_image_path is not None
    
    canvas_width = (
        (anchor_size[0] + padding * 2 if has_anchor else 0) +
        category_label_width +
        items_per_category * (image_size[0] + padding) + padding
    )
    
    canvas_height = (
        60 +
        n_categories * (image_size[1] + label_height + padding) + padding
    )
    
    if has_anchor:
        canvas_height = max(canvas_height, anchor_size[1] + 120)
    
    collage = Image.new('RGB', (int(canvas_width), int(canvas_height)), color='white')
    draw = ImageDraw.Draw(collage)
    
    # Title
    title = f"OUTFIT: {parsed.get('anchor_color', '')} {parsed.get('anchor_garment', '')} | {parsed.get('occasion', '')}".upper().strip()
    draw.text((padding, 15), title, fill='black')
    
    y_offset = 55
    x_items_start = padding
    
    # Draw anchor
    if has_anchor:
        try:
            anchor_img = Image.open(anchor_image_path).convert('RGB')
            anchor_img = anchor_img.resize(anchor_size, Image.LANCZOS)
            collage.paste(anchor_img, (padding, y_offset))
            draw.text((padding, y_offset + anchor_size[1] + 5), "YOUR ITEM", fill='darkgreen')
        except Exception as e:
            print(f"Anchor error: {e}")
        x_items_start = padding + anchor_size[0] + padding
    
    category_names = {'top': 'TOPS', 'bottom': 'BOTTOMS', 'outerwear': 'LAYERS'}
    category_y = y_offset
    
    # Draw items BY QUERY (not pooled)
    for category, query_items in queries_by_category.items():
        # Category label
        draw.text(
            (x_items_start, category_y + image_size[1] // 2 - 10),
            category_names.get(category, category.upper()),
            fill='grey'
        )
        
        x = x_items_start + category_label_width
        
        # Show 1 best result per query
        for item in query_items:
            query = item['query']
            results = item['results']
            
            if not results:
                continue
            
            # Take best result for this query
            best = results[0]
            
            try:
                img = Image.open(best['path']).convert('RGB')
                img = img.resize(image_size, Image.LANCZOS)
                collage.paste(img, (int(x), int(category_y)))
                
                # Labels
                is_bold = query.get('is_bold', False)
                color = query.get('color', '')
                garment = query.get('garment_type', '')
                
                text_y = int(category_y + image_size[1] + 3)
                
                # Line 1: Color + Garment
                label = f"{color} {garment}".strip().title()[:22]
                draw.text((int(x), text_y), label, fill='#333333')
                
                # Line 2: Bold tag or descriptor
                if is_bold:
                    draw.text((int(x), text_y + 15), "⚡ BOLD - Color theory", fill='#D4AF37')
                else:
                    draw.text((int(x), text_y + 15), "Safe neutral choice", fill='#888888')
                
            except Exception as e:
                print(f"Image error: {e}")
            
            x += image_size[0] + padding
        
        category_y += image_size[1] + label_height + padding
    
    return collage


def create_ensemble_collage_v2(
    outfit_results: Dict,
    anchor_image_path: str = None,
    image_size: Tuple[int, int] = (160, 220),
) -> Image.Image:
    """
    Production collage with accessories row support.
    Shows 1 best result per query, grouped by category.
    """
    parsed = outfit_results.get('parsed', {})
    ensemble_items = outfit_results.get('ensemble_items', [])
    
    if anchor_image_path is None:
        anchor_image_path = outfit_results.get('anchor_image')
    
    # Group by category
    queries_by_category = {}
    for item in ensemble_items:
        cat = item['query']['garment_category']
        if cat not in queries_by_category:
            queries_by_category[cat] = []
        queries_by_category[cat].append(item)
    
    # Layout
    padding = 12
    label_height = 45
    category_label_width = 90
    anchor_size = (180, 260)
    
    n_categories = len(queries_by_category)
    max_items = max(len(v) for v in queries_by_category.values()) if queries_by_category else 4
    
    has_anchor = anchor_image_path is not None
    
    canvas_width = (
        (anchor_size[0] + padding * 2 if has_anchor else 0) +
        category_label_width +
        max_items * (image_size[0] + padding) + padding
    )
    
    canvas_height = (
        55 +
        n_categories * (image_size[1] + label_height + padding) + padding
    )
    
    if has_anchor:
        canvas_height = max(canvas_height, anchor_size[1] + 100)
    
    collage = Image.new('RGB', (int(canvas_width), int(canvas_height)), color='white')
    draw = ImageDraw.Draw(collage)
    
    # Title
    title = f"OUTFIT: {parsed.get('anchor_color', '')} {parsed.get('anchor_garment', '')} | {parsed.get('occasion', '')}".upper().strip()
    draw.text((padding, 12), title, fill='black')
    
    y_offset = 50
    x_items_start = padding
    
    # Anchor
    if has_anchor:
        try:
            anchor_img = Image.open(anchor_image_path).convert('RGB')
            anchor_img = anchor_img.resize(anchor_size, Image.LANCZOS)
            collage.paste(anchor_img, (padding, y_offset))
            draw.text((padding, y_offset + anchor_size[1] + 3), "YOUR ITEM", fill='darkgreen')
        except:
            pass
        x_items_start = padding + anchor_size[0] + padding
    
    category_names = {'top': 'TOPS', 'bottom': 'BOTTOMS', 'outerwear': 'LAYERS', 'accessory': 'ACCESSORIES'}
    category_y = y_offset
    
    # Draw by category
    for category, query_items in queries_by_category.items():
        draw.text(
            (x_items_start, category_y + image_size[1] // 2 - 10),
            category_names.get(category, category.upper()),
            fill='grey'
        )
        
        x = x_items_start + category_label_width
        
        for item in query_items:
            query = item['query']
            results = item['results']
            
            if not results:
                continue
            
            best = results[0]
            
            try:
                img = Image.open(best['path']).convert('RGB')
                img = img.resize(image_size, Image.LANCZOS)
                collage.paste(img, (int(x), int(category_y)))
                
                is_bold = query.get('is_bold', False)
                color = query.get('color', '')
                garment = query.get('garment_type', '')
                
                text_y = int(category_y + image_size[1] + 2)
                label = f"{color} {garment}".strip().title()[:20]
                draw.text((int(x), text_y), label, fill='#333333')
                
                if is_bold:
                    draw.text((int(x), text_y + 13), "⚡ BOLD", fill='#D4AF37')
                else:
                    draw.text((int(x), text_y + 13), "Safe choice", fill='#888888')
                
            except Exception as e:
                print(f"Collage error: {e}")
            
            x += image_size[0] + padding
        
        category_y += image_size[1] + label_height + padding
    
    return collage


# =============================================================================
# CONTEXT-AWARE PIPELINE V2
# =============================================================================

def build_outfit_context_aware_v2(
    user_text: str,
    anchor_image_path: str = None,
    k: int = 5,
    include_accessories: bool = False,
    # Injected dependencies
    index=None,
    valid_paths=None,
    model=None,
    tokenizer=None,
    device=None,
) -> Dict:
    """
    Production pipeline with 'full view' modifier for complete garment images.
    
    Args:
        user_text: User's fashion request
        anchor_image_path: Path to anchor garment image
        k: Number of results per query
        include_accessories: Whether to include accessory recommendations
        index: FAISS index
        valid_paths: List of image paths
        model: CLIP model
        tokenizer: CLIP tokenizer
        device: torch device
    """
    from ensemble_rules import build_context_aware_queries_v2
    import torch
    
    print("="*60)
    print(f"User: '{user_text}'")
    print("="*60)
    
    # LLM Parse
    print("\n[1] LLM Parsing...")
    parsed = parse_user_query_robust(user_text)
    data = parsed['data']
    print(f"    Anchor: {data.get('anchor_color')} {data['anchor_garment']}")
    print(f"    Occasion: {data.get('occasion')}")
    print(f"    Style: {data.get('style')}")
    
    # Build v2 queries
    print("\n[2] Building Queries (with 'full view')...")
    queries = build_context_aware_queries_v2(
        anchor_garment=data['anchor_garment'],
        anchor_color=data.get('anchor_color', 'white'),
        occasion=data.get('occasion'),
        style=data.get('style'),
        include_accessories=include_accessories
    )
    
    for q in queries:
        bold_tag = "⚡" if q['is_bold'] else "  "
        print(f"    {bold_tag} '{q['search_text'][:45]}'")
    
    # FAISS Search
    print("\n[3] Searching FAISS...")
    results = {
        'parsed': data,
        'anchor_image': anchor_image_path,
        'ensemble_items': []
    }
    
    for q in queries:
        if index is not None and model is not None:
            text_tokens = tokenizer([q['search_text']]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            import numpy as np
            distances, indices = index.search(
                text_features.cpu().numpy().astype(np.float32), k=k
            )
            
            results['ensemble_items'].append({
                'query': q,
                'results': [
                    {'path': valid_paths[i], 'score': float(d)} 
                    for i, d in zip(indices[0], distances[0])
                ]
            })
        else:
            results['ensemble_items'].append({
                'query': q,
                'results': []
            })
    
    print("\n[4] Done!")
    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'FASHION_SYSTEM_PROMPT',
    'STYLE_DESCRIPTOR_PROMPT',
    'parse_user_query',
    'parse_user_query_robust',
    'extract_color_from_text',
    'generate_style_descriptors',
    'generate_fallback_descriptors',
    'build_outfit_pipeline',
    'build_outfit_context_aware',
    'build_outfit_context_aware_v2',
    'create_ensemble_collage',
    'create_ensemble_collage_fixed',
    'create_ensemble_collage_v2',
    'create_simple_collage',
    'generate_shopping_prompt',
    'generate_image_gen_prompt',
]