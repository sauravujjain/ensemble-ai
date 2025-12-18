"""
Ensemble AI - Fashion Outfit Builder
Streamlit Application v3

Changes from v2:
- LLM-only parsing (no fallback)
- Debug expander showing LLM understanding + generated queries
- Clear error if LLM unavailable
"""

import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import faiss
import random
import json

# Garment normalization for robust parsing
from garment_normalizer import normalize_garment, get_garment_category, get_needed_categories, GarmentCategory

# Multimodal search (CLIP image + text)
from multimodal_search import (
    get_image_embedding,
    search_by_image,
    search_multimodal,
    extract_visual_features,
    get_image_description
)

# Page config
st.set_page_config(
    page_title="Ensemble AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Elegant Fashion UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Lato:wght@300;400;500;600;700&display=swap');
    
    /* Clean Cream Background */
    .stApp {
        background: linear-gradient(180deg, #FAF8F5 0%, #F5F0EB 100%);
        min-height: 100vh;
    }
    
    .main > div {
        padding-top: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header */
    .header-container {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
    }
    
    .app-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .app-subtitle {
        font-family: 'Lato', sans-serif;
        font-size: 1.3rem;
        color: #666;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .status-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.6rem 1.5rem;
        border-radius: 25px;
        color: white;
        font-family: 'Lato', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Glass Card */
    .main-card {
        background: white;
        border-radius: 24px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Upload Zone - matches text area height */
    .upload-zone {
        background: #FAFAFA;
        border: 2px dashed #DDD;
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #667eea;
        background: #F5F5FF;
    }
    
    .upload-icon { font-size: 2.5rem; color: #999; margin-bottom: 0.5rem; }
    .upload-title { font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 600; color: #333; margin-bottom: 0.3rem; }
    .upload-subtitle { font-family: 'Lato', sans-serif; font-size: 0.9rem; color: #888; }
    .upload-hint { font-family: 'Lato', sans-serif; font-size: 0.9rem; color: #999; font-style: italic; margin-top: 1rem; }
    
    /* Image Preview Container */
    .image-preview-container {
        width: 100%;
        max-width: 200px;
        height: 180px;
        margin: 0.5rem auto;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .image-preview-container img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        background: #f5f5f5;
    }
    
    /* Plus Connector */
    .plus-connector { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 160px; }
    .plus-icon { width: 50px; height: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; color: white; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); }
    .plus-text { font-family: 'Lato', sans-serif; font-size: 0.8rem; color: #999; text-transform: uppercase; letter-spacing: 3px; margin-top: 0.5rem; font-weight: 600; }
    
    /* Text Area */
    .stTextArea textarea {
        background: #FAFAFA !important;
        border: 2px solid #E0E0E0 !important;
        border-radius: 12px !important;
        color: #333 !important;
        font-size: 1.1rem !important;
        padding: 1.2rem !important;
        min-height: 120px !important;
        font-family: 'Lato', sans-serif !important;
    }
    .stTextArea textarea::placeholder { color: #999 !important; }
    .stTextArea textarea:focus { border-color: #667eea !important; box-shadow: 0 0 0 3px rgba(102,126,234,0.15) !important; }
    
    /* Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
        color: white !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.9rem 2rem !important;
        border-radius: 30px !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.35) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px !important;
    }
    
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.45) !important;
    }
    
    div.stButton > button[kind="secondary"] {
        background: white !important;
        color: #FF8E53 !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.9rem 2rem !important;
        border-radius: 30px !important;
        border: 2px solid #FF8E53 !important;
        transition: all 0.3s ease !important;
    }
    
    div.stButton > button[kind="secondary"]:hover {
        background: #FFF5F0 !important;
    }
    
    /* Query Box */
    .query-box {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border-left: 4px solid #667eea;
        padding: 1.2rem 1.8rem;
        margin: 1rem 0;
        border-radius: 0 12px 12px 0;
        color: #333;
        font-family: 'Lato', sans-serif;
        font-size: 1.1rem;
    }
    
    /* Selection Counter */
    .selection-counter {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-family: 'Lato', sans-serif;
        color: #2E7D32;
        text-align: center;
        margin: 1.5rem 0;
        font-weight: 500;
    }
    
    /* Badges */
    .bold-badge {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #333;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.25rem;
        font-family: 'Lato', sans-serif;
    }
    
    .safe-badge {
        background: #E8F5E9;
        color: #2E7D32;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
        margin-top: 0.25rem;
        font-family: 'Lato', sans-serif;
    }
    
    /* Error Box */
    .error-box {
        background: #FFEBEE;
        border-left: 4px solid #f44336;
        padding: 1.2rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        color: #C62828;
        font-family: 'Lato', sans-serif;
        font-size: 1rem;
    }
    
    /* Page Header for other pages */
    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-family: 'Lato', sans-serif;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    
    /* Result Cards - Uniform size, slightly larger */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stImage img {
        max-height: 250px !important;
        width: 100% !important;
        object-fit: cover !important;
    }
    
    /* Category headers */
    h3 {
        font-family: 'Playfair Display', serif !important;
        color: #2C3E50 !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }
    
    /* General text */
    p, span, div {
        font-family: 'Lato', sans-serif;
    }
    
    /* Captions */
    .stCaption {
        font-family: 'Lato', sans-serif !important;
        font-size: 0.95rem !important;
        color: #666 !important;
    }
    
    /* File Uploader */
    .stFileUploader > div > div { background: transparent !important; border: none !important; }
    .stFileUploader label { color: #666 !important; font-family: 'Lato', sans-serif !important; }
    
    /* Expander */
    .streamlit-expanderHeader { 
        background: #F5F5F5 !important; 
        border-radius: 8px !important; 
        color: #333 !important;
        font-family: 'Lato', sans-serif !important;
    }
    .streamlit-expanderContent { background: #FAFAFA !important; color: #333 !important; }
    
    /* Checkbox */
    .stCheckbox label { color: #333 !important; font-family: 'Lato', sans-serif !important; font-size: 1rem !important; }
    
    /* Info/Success/Warning boxes */
    .stAlert { font-family: 'Lato', sans-serif !important; font-size: 1rem !important; }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# EXAMPLE PROMPTS - With gender and age specificity
# =============================================================================

EXAMPLE_PROMPTS = [
    "I have a navy blazer and need a complete women's office look",
    "Looking for men's casual weekend brunch outfit with white sneakers",
    "Help me style a women's maroon kurta for a family pooja ceremony",
    "I want to wear my black leather jacket for a men's date night look",
    "Need a women's festive saree ensemble for an Indian wedding",
    "Suggest men's smart casual outfits with chinos and loafers",
    "I have beige palazzos - what women's tops would go well?",
    "Looking for a women's summer dress outfit for a garden party",
    "Need boys' formal outfit for a school function",
    "Looking for girls' festive lehenga for Diwali celebration",
    "Men's formal suit options for a business conference",
    "Women's athleisure outfit for gym and brunch",
]

SAMPLE_PROMPTS_FOR_LLM = [
    "women's casual friday office look with jeans",
    "men's elegant evening party outfit in black",
    "women's colorful festive Indian wear for Diwali",
    "men's smart casual outfit for business meeting",
    "women's boho chic summer look with florals",
    "men's minimalist monochrome outfit",
    "women's sporty athleisure weekend wear",
    "men's romantic date night ensemble in navy",
    "girls' cute party dress in pink",
    "boys' casual weekend outfit with shorts",
]


# =============================================================================
# LLM SYSTEM PROMPT
# =============================================================================

LLM_PARSE_PROMPT = """You are a fashion assistant. Parse the user's fashion request and extract structured information.

User request: "{user_text}"

Extract the following and return ONLY valid JSON (no explanation, no markdown):
{{
    "intent": "have" or "looking",
    "anchor_garment": "string",
    "anchor_color": "string or null",
    "occasion": "string",
    "style": "string",
    "gender": "string or null",
    "age_group": "string"
}}

Field definitions:
- intent: "have" = user owns/has the item, "looking" = searching for item
- anchor_garment: main piece (blazer, kurta, dress, jeans, shirt, t-shirt, saree, top, skirt, etc.)
- anchor_color: color if mentioned or null
- occasion: office, party, casual, wedding, festive, date, brunch, formal, gym
- style: western, ethnic, casual, formal, fusion, athleisure
- gender: "mens", "womens", or null if not specified
- age_group: "adult" or "kids"

Gender detection:
- "men", "mens", "men's", "male", "guy", "his", "husband", "boyfriend" → "mens"
- "women", "womens", "women's", "female", "lady", "her", "wife", "girlfriend" → "womens"
- "boy", "boys", "girl", "girls" with kids context → use null for gender, set age_group to "kids"
- If NO gender indicators found → null

Age detection:
- "boy", "girl", "kid", "child", "son", "daughter", "school" → "kids"
- Otherwise → "adult"

Return ONLY the JSON object."""


LLM_DESCRIPTION_PROMPT = """Generate a short, stylish description (max 10 words) for this fashion item:
- Item: {color} {garment}
- Occasion: {occasion}
- Style note: {style_note}

Return ONLY the description text, nothing else. Be creative and fashion-forward.
Examples: "Effortlessly chic for modern professionals", "Bold statement piece that turns heads", "Timeless elegance meets comfort"
"""


# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state():
    defaults = {
        'page': 'landing',
        'model_loaded': False,
        'user_text': '',
        'uploaded_image': None,
        'search_results': [],
        'search_results_shown': 4,
        'selected_anchor_idx': None,
        'ensemble_results': None,
        'shown_items': {'top': 4, 'bottom': 4, 'accessory': 4, 'coordinating': 4},
        'selected_items': {'top': [], 'bottom': [], 'accessory': [], 'coordinating': []},
        'parsed_query': None,
        'generated_queries': [],  # Store all generated CLIP queries
        'llm_available': None,
        'example_prompts_display': None,  # Persisted examples
        'model': None,
        'tokenizer': None,
        'preprocess': None,  # CLIP preprocess function
        'index': None,
        'valid_paths': None,
        'sources': None,
        'device': None,
        # Final Look page state
        'final_look_collage': None,
        'final_look_prompt': None,
        'final_look_prompt_compact': None,
        # Multimodal image processing
        'uploaded_image_embedding': None,  # CLIP embedding of uploaded image
        'uploaded_image_features': None,   # Extracted visual features (color, type, etc.)
        'anchor_from_upload': False,       # Flag: anchor is from user upload (not FAISS selection)
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_models():
    import open_clip
    
    base_path = Path.home() / "projects/fashion-ensemble-builder/data/embeddings"
    
    index_path = base_path / "combined_faiss.index"
    paths_path = base_path / "combined_paths.txt"
    sources_path = base_path / "combined_sources.txt"
    
    if not index_path.exists():
        return None, None, None, None, None, None, None
    
    index = faiss.read_index(str(index_path))
    
    with open(paths_path, 'r') as f:
        valid_paths = [line.strip() for line in f.readlines()]
    
    with open(sources_path, 'r') as f:
        sources = [line.strip() for line in f.readlines()]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:Marqo/marqo-fashionCLIP'
    )
    tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionCLIP')
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, preprocess, index, valid_paths, sources, device


def check_llm_available():
    """Check if Ollama/LLM is available."""
    try:
        import ollama
        # Try a simple test
        response = ollama.generate(model='phi3', prompt='Say "ok"', options={'num_predict': 5})
        return True
    except Exception as e:
        return False


# =============================================================================
# LLM FUNCTIONS (NO FALLBACK)
# =============================================================================

def parse_user_query_llm(user_text):
    """
    Parse user query using LLM ONLY.
    Returns parsed dict or raises exception if LLM fails.
    Applies defaults for gender (womens) and age_group (adult) if not specified.
    """
    import ollama
    import re
    
    prompt = LLM_PARSE_PROMPT.format(user_text=user_text)
    
    response = ollama.generate(
        model='phi3',
        prompt=prompt,
        options={'temperature': 0.3, 'num_predict': 300}
    )
    
    response_text = response['response'].strip()
    
    # Extract JSON from response
    json_match = re.search(r'\{[^{}]+\}', response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            
            # Validate required fields
            required = ['intent', 'anchor_garment', 'occasion']
            if not all(k in parsed for k in required):
                raise ValueError(f"Missing required fields in: {parsed}")
            
            # Track defaults applied
            defaults_applied = []
            
            # Apply gender default if null/missing/unisex
            if not parsed.get('gender') or parsed.get('gender') in ['null', 'unisex', None]:
                parsed['gender'] = 'womens'
                defaults_applied.append('gender → women\'s')
            
            # Apply age_group default if missing
            if not parsed.get('age_group') or parsed.get('age_group') in ['null', None]:
                parsed['age_group'] = 'adult'
                defaults_applied.append('age → adult')
            
            # Store defaults info for UI display
            parsed['_defaults_applied'] = defaults_applied
            
            return parsed
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from LLM: {response_text}")
    
    raise ValueError(f"No JSON found in LLM response: {response_text}")


def generate_item_description_llm(color, garment, occasion, is_bold=False):
    """Generate description using LLM."""
    try:
        import ollama
        
        style_note = "bold statement piece" if is_bold else "versatile classic"
        prompt = LLM_DESCRIPTION_PROMPT.format(
            color=color,
            garment=garment,
            occasion=occasion,
            style_note=style_note
        )
        
        response = ollama.generate(
            model='phi3',
            prompt=prompt,
            options={'temperature': 0.7, 'num_predict': 30}
        )
        
        desc = response['response'].strip()
        # Clean up any quotes or extra text
        desc = desc.strip('"\'')
        if len(desc) > 50:
            desc = desc[:50] + "..."
        return desc
    except:
        # Minimal fallback for descriptions only (not parsing)
        if is_bold:
            return f"Bold {color} {garment} - make a statement"
        return f"Classic {color} {garment} - effortlessly stylish"


# =============================================================================
# SEARCH & ENSEMBLE
# =============================================================================

def search_by_text(query, k=20):
    """Search FAISS index by text query."""
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    index = st.session_state.index
    valid_paths = st.session_state.valid_paths
    sources = st.session_state.sources
    device = st.session_state.device
    
    text_tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    distances, indices = index.search(
        text_features.cpu().numpy().astype(np.float32), k=k
    )
    
    results = []
    for d, idx in zip(distances[0], indices[0]):
        source = sources[idx]
        results.append({
            'path': valid_paths[idx],
            'score': float(d),
            'source': source
        })
        if len(results) >= k:
            break
    
    return results


def generate_ensemble(parsed_query):
    """Generate ensemble based on parsed query."""
    
    garment = parsed_query.get('anchor_garment', 'blazer')
    color = parsed_query.get('anchor_color') or 'navy'
    occasion = parsed_query.get('occasion', 'casual')
    gender = parsed_query.get('gender', 'womens')
    style = parsed_query.get('style', 'western')
    age_group = parsed_query.get('age_group', 'adult')
    
    # Clear previous queries
    st.session_state.generated_queries = []
    
    # Build query prefix based on gender and age
    query_prefix = ""
    if age_group == 'kids':
        if gender == 'mens':
            query_prefix = "boys' "
        elif gender == 'womens':
            query_prefix = "girls' "
        else:
            query_prefix = "kids' "
    else:
        if gender == 'mens':
            query_prefix = "men's "
        elif gender == 'womens':
            query_prefix = "women's "
    
    ensemble = {}
    
    # Normalize garment and determine needed categories using fuzzy matching
    normalized_garment = normalize_garment(garment) if garment else 'top'
    garment_category = get_garment_category(garment) if garment else GarmentCategory.TOP
    needed = get_needed_categories(garment) if garment else {'need_tops': False, 'need_bottoms': True, 'need_outerwear': False}
    
    need_tops = needed['need_tops']
    need_bottoms = needed['need_bottoms']
    
    # Debug: Log normalization results
    print(f"[Normalizer] Input: '{garment}' -> Normalized: '{normalized_garment}' | Category: {garment_category.value}")
    print(f"[Normalizer] Need tops: {need_tops}, Need bottoms: {need_bottoms}")
    
    # Color palettes based on anchor color
    color_palettes = {
        'navy': {'safe': ['white', 'cream', 'light blue', 'blush'], 'bold': ['coral', 'mustard', 'rust']},
        'black': {'safe': ['white', 'grey', 'cream', 'camel'], 'bold': ['red', 'emerald', 'cobalt']},
        'maroon': {'safe': ['beige', 'cream', 'gold', 'white'], 'bold': ['teal', 'mint', 'turquoise']},
        'white': {'safe': ['navy', 'black', 'beige', 'grey'], 'bold': ['cobalt', 'coral', 'emerald']},
        'beige': {'safe': ['white', 'brown', 'cream', 'olive'], 'bold': ['burgundy', 'teal', 'rust']},
        'grey': {'safe': ['white', 'navy', 'black', 'burgundy'], 'bold': ['mustard', 'teal', 'coral']},
        'brown': {'safe': ['cream', 'white', 'beige', 'olive'], 'bold': ['teal', 'rust', 'burgundy']},
        'olive': {'safe': ['white', 'cream', 'tan', 'grey'], 'bold': ['burgundy', 'rust', 'mustard']},
        'red': {'safe': ['white', 'black', 'navy', 'grey'], 'bold': ['gold', 'emerald', 'cobalt']},
        'blue': {'safe': ['white', 'cream', 'grey', 'tan'], 'bold': ['coral', 'mustard', 'burgundy']},
        'green': {'safe': ['white', 'cream', 'beige', 'navy'], 'bold': ['coral', 'burgundy', 'gold']},
        'pink': {'safe': ['white', 'grey', 'navy', 'cream'], 'bold': ['emerald', 'cobalt', 'gold']},
    }
    
    palette = color_palettes.get(color.lower() if color else 'navy', color_palettes['navy'])
    safe_colors = palette['safe']
    bold_colors = palette['bold']
    
    # ==========================================================================
    # GENDER-SPECIFIC GARMENT MAPPINGS
    # ==========================================================================
    
    # TOPS by gender and occasion
    top_garments_by_gender = {
        'mens': {
            'office': ['formal shirt', 'dress shirt', 'oxford shirt', 'button down shirt'],
            'casual': ['t-shirt', 'polo shirt', 'henley', 'casual shirt'],
            'party': ['silk shirt', 'printed shirt', 'velvet shirt', 'statement shirt'],
            'festive': ['embroidered kurta', 'silk kurta', 'ethnic shirt', 'brocade kurta'],
            'date': ['fitted shirt', 'smart casual shirt', 'linen shirt', 'stylish polo'],
            'brunch': ['linen shirt', 'casual polo', 'relaxed shirt', 'cotton henley'],
            'wedding': ['designer kurta', 'silk shirt', 'embroidered shirt', 'formal kurta'],
        },
        'womens': {
            'office': ['formal blouse', 'silk shirt', 'turtleneck', 'tailored top'],
            'casual': ['t-shirt', 'casual blouse', 'tank top', 'knit top'],
            'party': ['silk blouse', 'sequin top', 'statement top', 'dressy camisole'],
            'festive': ['embroidered top', 'silk kurti', 'ethnic blouse', 'brocade top'],
            'date': ['elegant blouse', 'fitted top', 'romantic top', 'chic shirt'],
            'brunch': ['casual blouse', 'flowy top', 'linen shirt', 'relaxed tee'],
            'wedding': ['designer blouse', 'embroidered top', 'silk kurti', 'statement blouse'],
        },
        'boys': {
            'office': ['formal shirt', 'polo shirt', 'smart shirt', 'button down'],
            'casual': ['t-shirt', 'polo', 'graphic tee', 'casual shirt'],
            'party': ['smart shirt', 'printed shirt', 'formal polo', 'dressy shirt'],
            'festive': ['kurta', 'ethnic shirt', 'embroidered kurta', 'silk kurta'],
            'date': ['casual shirt', 'polo', 'smart tee', 'henley'],
            'brunch': ['casual tee', 'polo shirt', 'comfortable shirt', 'cotton top'],
            'wedding': ['formal kurta', 'silk kurta', 'embroidered shirt', 'designer kurta'],
        },
        'girls': {
            'office': ['formal top', 'smart blouse', 'collared shirt', 'neat top'],
            'casual': ['t-shirt', 'casual top', 'tank top', 'cute tee'],
            'party': ['party top', 'sparkly top', 'pretty blouse', 'dressy top'],
            'festive': ['ethnic top', 'kurti', 'embroidered top', 'festive blouse'],
            'date': ['pretty top', 'casual blouse', 'cute shirt', 'stylish tee'],
            'brunch': ['casual top', 'comfortable tee', 'relaxed blouse', 'cotton top'],
            'wedding': ['designer kurti', 'embroidered top', 'festive top', 'silk blouse'],
        },
    }
    
    # BOTTOMS by gender and occasion
    bottom_garments_by_gender = {
        'mens': {
            'office': ['formal trousers', 'dress pants', 'tailored pants', 'chinos'],
            'casual': ['jeans', 'chinos', 'casual pants', 'cargo pants'],
            'party': ['slim fit pants', 'dark jeans', 'dressy trousers', 'fitted chinos'],
            'festive': ['churidar', 'ethnic pants', 'silk pants', 'dhoti pants'],
            'date': ['smart chinos', 'fitted jeans', 'casual trousers', 'stylish pants'],
            'brunch': ['linen pants', 'casual chinos', 'relaxed jeans', 'cotton pants'],
            'wedding': ['silk churidar', 'formal pants', 'ethnic dhoti', 'designer trousers'],
        },
        'womens': {
            'office': ['tailored trousers', 'pencil skirt', 'dress pants', 'midi skirt'],
            'casual': ['jeans', 'casual pants', 'shorts', 'leggings'],
            'party': ['statement skirt', 'fitted pants', 'sequin skirt', 'leather pants'],
            'festive': ['palazzos', 'ethnic skirt', 'churidar', 'silk pants'],
            'date': ['elegant pants', 'midi skirt', 'fitted trousers', 'A-line skirt'],
            'brunch': ['linen pants', 'flowy skirt', 'casual jeans', 'culottes'],
            'wedding': ['silk palazzos', 'lehenga skirt', 'embroidered pants', 'ethnic dhoti'],
        },
        'boys': {
            'office': ['formal pants', 'chinos', 'dress pants', 'smart trousers'],
            'casual': ['jeans', 'cargo pants', 'shorts', 'joggers'],
            'party': ['smart pants', 'dark jeans', 'dressy chinos', 'formal pants'],
            'festive': ['churidar', 'ethnic pants', 'dhoti pants', 'silk pants'],
            'date': ['jeans', 'smart chinos', 'casual pants', 'neat trousers'],
            'brunch': ['casual pants', 'comfortable jeans', 'shorts', 'cotton pants'],
            'wedding': ['formal churidar', 'silk pants', 'ethnic dhoti', 'designer pants'],
        },
        'girls': {
            'office': ['formal pants', 'neat skirt', 'dress pants', 'smart trousers'],
            'casual': ['jeans', 'leggings', 'shorts', 'casual skirt'],
            'party': ['pretty skirt', 'party pants', 'sparkly leggings', 'tutu skirt'],
            'festive': ['lehenga skirt', 'ethnic skirt', 'palazzos', 'churidar'],
            'date': ['cute skirt', 'casual jeans', 'neat pants', 'pretty leggings'],
            'brunch': ['casual skirt', 'comfortable pants', 'shorts', 'cotton leggings'],
            'wedding': ['lehenga skirt', 'festive skirt', 'silk pants', 'embroidered palazzos'],
        },
    }
    
    # ACCESSORIES by gender and occasion (with color theory integration)
    accessory_types_by_gender = {
        'mens': {
            'office': ['leather belt', 'formal watch', 'cufflinks', 'leather briefcase'],
            'casual': ['canvas belt', 'casual watch', 'sunglasses', 'backpack'],
            'party': ['statement watch', 'pocket square', 'tie clip', 'leather shoes'],
            'festive': ['ethnic mojari', 'brooch', 'traditional watch', 'ethnic stole'],
            'date': ['stylish watch', 'leather belt', 'sunglasses', 'smart shoes'],
            'brunch': ['casual watch', 'canvas belt', 'sunglasses', 'messenger bag'],
            'wedding': ['designer mojari', 'safa', 'brooch', 'statement cufflinks'],
        },
        'womens': {
            'office': ['leather handbag', 'stud earrings', 'formal watch', 'leather belt'],
            'casual': ['tote bag', 'hoop earrings', 'sneakers', 'casual watch'],
            'party': ['clutch', 'statement earrings', 'heels', 'bracelet'],
            'festive': ['potli bag', 'jhumka earrings', 'bangles', 'ethnic sandals'],
            'date': ['crossbody bag', 'delicate necklace', 'heels', 'elegant watch'],
            'brunch': ['tote bag', 'sunglasses', 'sandals', 'dainty earrings'],
            'wedding': ['designer clutch', 'statement jewelry', 'embellished heels', 'maang tikka'],
        },
        'boys': {
            'office': ['belt', 'watch', 'formal shoes', 'school bag'],
            'casual': ['cap', 'sneakers', 'backpack', 'casual watch'],
            'party': ['bow tie', 'smart shoes', 'suspenders', 'watch'],
            'festive': ['ethnic mojari', 'brooch', 'turban', 'ethnic belt'],
            'date': ['casual watch', 'sneakers', 'cap', 'sunglasses'],
            'brunch': ['cap', 'sneakers', 'backpack', 'sunglasses'],
            'wedding': ['ethnic mojari', 'safa', 'brooch', 'pagdi'],
        },
        'girls': {
            'office': ['hair band', 'small bag', 'neat shoes', 'simple watch'],
            'casual': ['hair clips', 'sneakers', 'small backpack', 'friendship bracelet'],
            'party': ['tiara', 'sparkly shoes', 'pretty bag', 'charm bracelet'],
            'festive': ['ethnic jewelry', 'juttis', 'potli bag', 'hair accessory'],
            'date': ['cute bag', 'hair band', 'pretty shoes', 'bracelet'],
            'brunch': ['sun hat', 'sandals', 'small bag', 'sunglasses'],
            'wedding': ['ethnic jewelry', 'embellished juttis', 'designer potli', 'maang tikka'],
        },
    }
    
    # Map gender from parsed query to our keys (considering age_group)
    gender_key = 'womens'  # default
    if age_group == 'kids':
        # Kids: map to boys/girls
        if gender == 'mens':
            gender_key = 'boys'
        elif gender == 'womens':
            gender_key = 'girls'
        else:
            gender_key = 'boys'  # default kids to boys
    else:
        # Adults
        if gender == 'mens':
            gender_key = 'mens'
        elif gender == 'womens':
            gender_key = 'womens'
        else:
            gender_key = 'womens'  # default adults to womens
    
    # Get gender-specific garment lists
    top_garments = top_garments_by_gender.get(gender_key, top_garments_by_gender['womens'])
    bottom_garments = bottom_garments_by_gender.get(gender_key, bottom_garments_by_gender['womens'])
    accessory_types = accessory_types_by_gender.get(gender_key, accessory_types_by_gender['womens'])
    
    print(f"[Ensemble] Gender: {gender} -> Key: {gender_key}")
    print(f"[Ensemble] Occasion: {occasion}")
    print(f"[Ensemble] Top types: {top_garments.get(occasion, top_garments['casual'])}")
    print(f"[Ensemble] Bottom types: {bottom_garments.get(occasion, bottom_garments['casual'])}")
    
    # Generate TOPS
    if need_tops:
        top_types = top_garments.get(occasion, top_garments['casual'])
        ensemble['top'] = []
        
        # Safe options (3)
        for i in range(min(3, len(safe_colors))):
            col = safe_colors[i]
            garm = top_types[i % len(top_types)]
            query = f"{query_prefix}{col} {garm} {occasion} full view"
            
            st.session_state.generated_queries.append({
                'category': 'top',
                'type': 'safe',
                'query': query
            })
            
            results = search_by_text(query, k=1)
            if results:
                results[0]['query'] = {'color': col, 'garment_type': garm, 'is_bold': False, 'search_text': query}
                results[0]['description'] = generate_item_description_llm(col, garm, occasion, False)
                ensemble['top'].append(results[0])
        
        # Bold option (1)
        bold_col = random.choice(bold_colors)
        bold_garm = top_types[-1]
        query = f"{query_prefix}{bold_col} {bold_garm} statement {occasion} full view"
        
        st.session_state.generated_queries.append({
            'category': 'top',
            'type': 'bold',
            'query': query
        })
        
        results = search_by_text(query, k=1)
        if results:
            results[0]['query'] = {'color': bold_col, 'garment_type': bold_garm, 'is_bold': True, 'search_text': query}
            results[0]['description'] = generate_item_description_llm(bold_col, bold_garm, occasion, True)
            ensemble['top'].append(results[0])
        
        # Extra for "show more"
        extra_colors = ['grey', 'sage', 'lavender', 'mint']
        for col in extra_colors:
            garm = random.choice(top_types)
            query = f"{query_prefix}{col} {garm} {occasion} full view"
            st.session_state.generated_queries.append({'category': 'top', 'type': 'extra', 'query': query})
            results = search_by_text(query, k=1)
            if results:
                results[0]['query'] = {'color': col, 'garment_type': garm, 'is_bold': False, 'search_text': query}
                results[0]['description'] = generate_item_description_llm(col, garm, occasion, False)
                ensemble['top'].append(results[0])
    
    # Generate BOTTOMS
    if need_bottoms:
        bottom_types = bottom_garments.get(occasion, bottom_garments['casual'])
        ensemble['bottom'] = []
        
        bottom_safe = ['black', 'navy', 'camel', 'grey']
        
        for i in range(min(3, len(bottom_safe))):
            col = bottom_safe[i]
            garm = bottom_types[i % len(bottom_types)]
            query = f"{query_prefix}{col} {garm} {occasion} full view"
            
            st.session_state.generated_queries.append({
                'category': 'bottom',
                'type': 'safe',
                'query': query
            })
            
            results = search_by_text(query, k=1)
            if results:
                results[0]['query'] = {'color': col, 'garment_type': garm, 'is_bold': False, 'search_text': query}
                results[0]['description'] = generate_item_description_llm(col, garm, occasion, False)
                ensemble['bottom'].append(results[0])
        
        # Bold bottom
        bold_col = random.choice(bold_colors)
        bold_garm = bottom_types[-1]
        query = f"{query_prefix}{bold_col} {bold_garm} statement {occasion} full view"
        
        st.session_state.generated_queries.append({
            'category': 'bottom',
            'type': 'bold',
            'query': query
        })
        
        results = search_by_text(query, k=1)
        if results:
            results[0]['query'] = {'color': bold_col, 'garment_type': bold_garm, 'is_bold': True, 'search_text': query}
            results[0]['description'] = generate_item_description_llm(bold_col, bold_garm, occasion, True)
            ensemble['bottom'].append(results[0])
        
        # Extra
        extra_bottom = ['olive', 'burgundy', 'tan', 'charcoal']
        for col in extra_bottom:
            garm = random.choice(bottom_types)
            query = f"{query_prefix}{col} {garm} {occasion} full view"
            st.session_state.generated_queries.append({'category': 'bottom', 'type': 'extra', 'query': query})
            results = search_by_text(query, k=1)
            if results:
                results[0]['query'] = {'color': col, 'garment_type': garm, 'is_bold': False, 'search_text': query}
                results[0]['description'] = generate_item_description_llm(col, garm, occasion, False)
                ensemble['bottom'].append(results[0])
    
    # ==========================================================================
    # SPECIAL HANDLING FOR ETHNIC FULL-BODY GARMENTS
    # ==========================================================================
    # For saree, lehenga, etc. - suggest coordinating pieces (blouse, dupatta)
    
    ethnic_full_body = ['saree', 'sari', 'lehenga', 'ghagra', 'chaniya']
    normalized_lower = normalized_garment.lower() if normalized_garment else ''
    
    if normalized_lower in ethnic_full_body or (garment and any(e in garment.lower() for e in ethnic_full_body)):
        ensemble['coordinating'] = []
        
        # Coordinating pieces for ethnic wear
        if 'saree' in normalized_lower or 'sari' in normalized_lower or (garment and ('saree' in garment.lower() or 'sari' in garment.lower())):
            coord_items = [
                ('matching', 'saree blouse'),
                ('contrast', 'designer blouse'),
                ('embroidered', 'silk blouse'),
            ]
        else:  # lehenga
            coord_items = [
                ('matching', 'choli'),
                ('embroidered', 'lehenga choli'),
                ('designer', 'crop top blouse'),
            ]
        
        for col_type, item in coord_items:
            # Use anchor color for matching, or complementary for contrast
            if col_type == 'matching':
                col = color if color else 'gold'
            elif col_type == 'contrast':
                col = bold_colors[0] if bold_colors else 'gold'
            else:
                col = 'gold'
            
            query = f"{query_prefix}{col} {item} {occasion} ethnic full view"
            
            st.session_state.generated_queries.append({
                'category': 'coordinating',
                'type': 'ethnic',
                'query': query
            })
            
            results = search_by_text(query, k=1)
            if results:
                results[0]['query'] = {'color': col, 'garment_type': item, 'is_bold': False, 'search_text': query}
                results[0]['description'] = f"{col.title()} {item} - perfect pairing"
                ensemble['coordinating'].append(results[0])
        
        # Also add dupatta/stole for ethnic looks
        dupatta_query = f"{query_prefix}{color if color else 'gold'} dupatta stole {occasion} ethnic full view"
        st.session_state.generated_queries.append({
            'category': 'coordinating',
            'type': 'ethnic',
            'query': dupatta_query
        })
        results = search_by_text(dupatta_query, k=1)
        if results:
            results[0]['query'] = {'color': color, 'garment_type': 'dupatta', 'is_bold': False, 'search_text': dupatta_query}
            results[0]['description'] = f"Elegant dupatta - adds grace to your look"
            ensemble['coordinating'].append(results[0])
    
    # Generate ACCESSORIES with color theory
    # Always included - accessories complete the look
    acc_types = accessory_types.get(occasion, accessory_types['casual'])
    ensemble['accessory'] = []
    
    # Accessory colors: mix of safe neutrals and bold accents
    accessory_colors_safe = ['black', 'brown', 'tan', 'silver', 'gold']
    accessory_colors_bold = bold_colors[:2] + ['gold', 'rose gold']
    
    # Safe accessories (3) - use neutral colors
    for i in range(min(3, len(acc_types))):
        col = accessory_colors_safe[i % len(accessory_colors_safe)]
        item = acc_types[i]
        # Include gender prefix for accessories too
        query = f"{query_prefix}{col} {item} {occasion} accessory full view"
        
        st.session_state.generated_queries.append({
            'category': 'accessory',
            'type': 'safe',
            'query': query
        })
        
        results = search_by_text(query, k=1)
        if results:
            results[0]['query'] = {'color': col, 'garment_type': item, 'is_bold': False, 'search_text': query}
            results[0]['description'] = f"Classic {col} {item} - versatile and timeless"
            ensemble['accessory'].append(results[0])
    
    # Bold accessory (1) - use complementary color from anchor
    if len(acc_types) > 0:
        bold_col = random.choice(accessory_colors_bold)
        bold_item = acc_types[-1]  # Last item as statement piece
        query = f"{query_prefix}{bold_col} {bold_item} statement {occasion} accessory full view"
        
        st.session_state.generated_queries.append({
            'category': 'accessory',
            'type': 'bold',
            'query': query
        })
        
        results = search_by_text(query, k=1)
        if results:
            results[0]['query'] = {'color': bold_col, 'garment_type': bold_item, 'is_bold': True, 'search_text': query}
            results[0]['description'] = f"Statement {bold_col} {bold_item} - eye-catching accent"
            ensemble['accessory'].append(results[0])
    
    # Extra accessories for "show more"
    extra_acc_colors = ['navy', 'burgundy', 'olive', 'grey']
    remaining_items = acc_types[3:] if len(acc_types) > 3 else acc_types
    for i, col in enumerate(extra_acc_colors[:len(remaining_items)]):
        item = remaining_items[i % len(remaining_items)] if remaining_items else acc_types[0]
        query = f"{query_prefix}{col} {item} {occasion} accessory full view"
        st.session_state.generated_queries.append({'category': 'accessory', 'type': 'extra', 'query': query})
        results = search_by_text(query, k=1)
        if results:
            results[0]['query'] = {'color': col, 'garment_type': item, 'is_bold': False, 'search_text': query}
            results[0]['description'] = f"Complementary {col} {item}"
            ensemble['accessory'].append(results[0])
    
    return ensemble


# =============================================================================
# PAGE RENDERERS
# =============================================================================

def render_landing_page():
    """Render landing page with modern UI."""
    
    # Header Section
    st.markdown("""
    <div class="header-container">
        <div class="app-title">🎨 Ensemble AI</div>
        <div class="app-subtitle">Your AI-powered fashion stylist. Upload an item + describe your style.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check LLM availability
    if st.session_state.llm_available is None:
        with st.spinner("Checking AI connection..."):
            st.session_state.llm_available = check_llm_available()
    
    if not st.session_state.llm_available:
        st.markdown("""
        <div class="error-box">
            <strong>⚠️ AI Not Available</strong><br>
            Ollama with phi3 model is required. Please ensure Ollama is running:<br>
            <code>ollama serve</code> and <code>ollama pull phi3</code>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Retry Connection"):
            st.session_state.llm_available = check_llm_available()
            st.rerun()
        return
    
    # Status Badge
    st.markdown('<div style="text-align:center;"><span class="status-badge">✅ Fashion AI loaded!</span></div>', unsafe_allow_html=True)
    
    # Main Content Card
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # Two Column Layout: Upload | Plus | Text Input
    col_upload, col_plus, col_input = st.columns([5, 1, 6])
    
    with col_upload:
        # Check if image is already uploaded
        has_image = st.session_state.uploaded_image is not None
        
        if not has_image:
            # Show upload zone placeholder
            st.markdown("""
            <div class="upload-zone">
                <div class="upload-icon">📷</div>
                <div class="upload-title">Upload your garment</div>
                <div class="upload-subtitle">Drag & drop or click to select</div>
                <div class="upload-subtitle">JPG, PNG, WEBP up to 10MB</div>
            </div>
            """, unsafe_allow_html=True)
        
        # File uploader (always present but styled)
        uploaded_file = st.file_uploader(
            "Upload image",
            type=['jpg', 'jpeg', 'png', 'webp'],
            label_visibility="collapsed",
            key="main_uploader"
        )
        
        # Handle new upload
        if uploaded_file:
            st.session_state.uploaded_image = uploaded_file
            # Show image preview with fixed size container
            st.image(Image.open(uploaded_file), caption="📷 Your garment", width=180)
            st.markdown('<p class="upload-hint">✓ Image uploaded! Click "Style This Item" or describe your needs →</p>', unsafe_allow_html=True)
        elif not has_image:
            st.markdown('<p class="upload-hint">or simply describe what you want in the text field →</p>', unsafe_allow_html=True)
    
    with col_plus:
        st.markdown("""
        <div class="plus-connector">
            <div class="plus-icon">+</div>
            <div class="plus-text">PLUS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_input:
        # Label for text input
        st.markdown('<p style="color: #333; font-family: Playfair Display, serif; font-weight: 600; font-size: 1.3rem; margin-bottom: 0.75rem;">Describe your styling needs:</p>', unsafe_allow_html=True)
        
        user_input = st.text_area(
            "Describe your outfit",
            value=st.session_state.user_text,
            placeholder="Example: 'Style this for a business meeting with a professional but approachable look'",
            height=120,
            label_visibility="collapsed"
        )
        st.session_state.user_text = user_input
        
        # Button Row
        btn_col1, btn_col2 = st.columns(2)
        
        # "Style This Item" - only enabled when image is uploaded
        has_uploaded_image = st.session_state.uploaded_image is not None
        
        with btn_col1:
            style_clicked = st.button(
                "🎨 Style This Item", 
                type="primary", 
                use_container_width=True,
                disabled=not has_uploaded_image,
                help="Upload an image first to use this feature" if not has_uploaded_image else "Auto-analyze your garment and create styling suggestions"
            )
        with btn_col2:
            surprise_clicked = st.button("🎲 Surprise me!", type="secondary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-card
    
    # Main CTA Button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        main_cta_clicked = st.button("✨ Style My Look", type="primary", use_container_width=True, key="main_cta")
    
    # Handle button actions
    
    # Surprise me - random prompt
    if surprise_clicked:
        st.session_state.user_text = random.choice(SAMPLE_PROMPTS_FOR_LLM)
        st.rerun()
    
    # "Style This Item" - Auto-analyze image and generate prompt
    if style_clicked and st.session_state.uploaded_image is not None:
        with st.spinner("👁️ Analyzing your garment..."):
            try:
                # Load and analyze image with CLIP
                img = Image.open(st.session_state.uploaded_image).convert('RGB')
                
                visual_features = extract_visual_features(
                    img,
                    st.session_state.model,
                    st.session_state.tokenizer,
                    st.session_state.preprocess,
                    st.session_state.device
                )
                st.session_state.uploaded_image_features = visual_features
                
                # Get image embedding
                img_embedding = get_image_embedding(
                    img,
                    st.session_state.model,
                    st.session_state.preprocess,
                    st.session_state.device
                )
                st.session_state.uploaded_image_embedding = img_embedding
                
                # Generate auto-prompt based on visual features
                color = visual_features.get('color', 'stylish')
                garment = visual_features.get('garment_type', 'garment')
                style = visual_features.get('style', 'casual')
                
                # Create a natural prompt
                auto_prompt = f"I have a {color} {garment}. Help me style it for a {style} look with matching pieces."
                
                st.session_state.user_text = auto_prompt
                
                print(f"[Style This Item] Visual features: {visual_features}")
                print(f"[Style This Item] Generated prompt: {auto_prompt}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Could not analyze image: {e}")
                print(f"[Style This Item] Error: {e}")
    
    # Main CTA - process and go to results/search
    if main_cta_clicked:
        if not st.session_state.user_text.strip():
            st.error("Please describe what you'd like to wear, or upload an image and click 'Style This Item'")
        else:
            try:
                # Step 1: Parse text with LLM
                with st.spinner("🧠 Understanding your request..."):
                    parsed = parse_user_query_llm(st.session_state.user_text)
                    st.session_state.parsed_query = parsed
                
                # Step 2: If image uploaded, extract visual features and augment parsing
                if st.session_state.uploaded_image is not None:
                    with st.spinner("👁️ Analyzing your image..."):
                        try:
                            # Load image
                            img = Image.open(st.session_state.uploaded_image).convert('RGB')
                            
                            # Extract visual features using CLIP (if not already done)
                            if st.session_state.uploaded_image_features is None:
                                visual_features = extract_visual_features(
                                    img,
                                    st.session_state.model,
                                    st.session_state.tokenizer,
                                    st.session_state.preprocess,
                                    st.session_state.device
                                )
                                st.session_state.uploaded_image_features = visual_features
                            else:
                                visual_features = st.session_state.uploaded_image_features
                            
                            # Get image embedding (if not already done)
                            if st.session_state.uploaded_image_embedding is None:
                                img_embedding = get_image_embedding(
                                    img,
                                    st.session_state.model,
                                    st.session_state.preprocess,
                                    st.session_state.device
                                )
                                st.session_state.uploaded_image_embedding = img_embedding
                            
                            # Augment parsed query with visual features
                            llm_color = parsed.get('anchor_color', '')
                            visual_color = visual_features.get('color', '')
                            if not llm_color or llm_color in ['neutral', 'colorful', '']:
                                parsed['anchor_color'] = visual_color
                                parsed['_visual_color_used'] = True
                            
                            llm_garment = parsed.get('anchor_garment', '')
                            visual_garment = visual_features.get('garment_type', '')
                            if not llm_garment or llm_garment in ['item', 'piece', 'garment', '']:
                                parsed['anchor_garment'] = visual_garment
                                parsed['_visual_garment_used'] = True
                            
                            st.session_state.parsed_query = parsed
                            
                            print(f"[Multimodal] Visual features: {visual_features}")
                            print(f"[Multimodal] Augmented parsed: color={parsed.get('anchor_color')}, garment={parsed.get('anchor_garment')}")
                            
                        except Exception as e:
                            print(f"[Multimodal] Image processing warning: {e}")
                    
                    # User provided image = mark anchor from upload, go to results
                    st.session_state.anchor_from_upload = True
                    st.session_state.page = 'results'
                else:
                    # No image = need to select anchor first
                    st.session_state.anchor_from_upload = False
                    st.session_state.page = 'search'
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")
                st.info("Please try rephrasing your request or check AI connection.")


def render_search_page():
    """Render anchor selection page."""
    
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown('<h1 class="main-header">🎨 Ensemble AI</h1>', unsafe_allow_html=True)
    with col2:
        if st.button("← Back"):
            st.session_state.page = 'landing'
            st.session_state.search_results = []
            st.session_state.selected_anchor_idx = None
            st.rerun()
    
    parsed = st.session_state.parsed_query or {}
    
    # Query box
    st.markdown(f"""
    <div class="query-box">
        <strong>🔍 Finding:</strong> {st.session_state.user_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Show defaults applied warning
    defaults_applied = parsed.get('_defaults_applied', [])
    if defaults_applied:
        st.info(f"ℹ️ **Assumed:** {', '.join(defaults_applied)} (not specified in your request)")
    
    # LLM Understanding expander
    with st.expander("🧠 LLM Understanding & Generated Queries", expanded=False):
        st.markdown("**Parsed from your request:**")
        # Remove internal field before displaying
        display_parsed = {k: v for k, v in parsed.items() if not k.startswith('_')}
        st.json(display_parsed)
        
        # Build search query
        gender_mod = "men's " if parsed.get('gender') == 'mens' else "women's " if parsed.get('gender') == 'womens' else ""
        age_mod = "kids' " if parsed.get('age_group') == 'kids' else ""
        search_query = f"{age_mod}{gender_mod}{parsed.get('anchor_color', '')} {parsed.get('anchor_garment', '')} full view".strip()
        
        st.markdown("**Search query for CLIP:**")
        st.code(search_query)
    
    st.markdown("### Select your anchor piece")
    st.caption("Choose the main item you want to build your outfit around")
    
    # Run search
    if not st.session_state.search_results:
        with st.spinner("Searching..."):
            gender_mod = "men's " if parsed.get('gender') == 'mens' else "women's " if parsed.get('gender') == 'womens' else ""
            age_mod = "kids' " if parsed.get('age_group') == 'kids' else ""
            query = f"{age_mod}{gender_mod}{parsed.get('anchor_color', '')} {parsed.get('anchor_garment', '')} full view".strip()
            results = search_by_text(query, k=20)
            
            for r in results:
                r['description'] = generate_item_description_llm(
                    parsed.get('anchor_color', 'stylish'),
                    parsed.get('anchor_garment', 'piece'),
                    parsed.get('occasion', 'casual'),
                    False
                )
            
            st.session_state.search_results = results
    
    results = st.session_state.search_results
    shown = st.session_state.search_results_shown
    
    cols = st.columns(4)
    for i, item in enumerate(results[:shown]):
        with cols[i % 4]:
            try:
                st.image(Image.open(item['path']), use_container_width=True)
            except:
                st.image("https://via.placeholder.com/150x200", use_container_width=True)
            
            st.markdown(f"**{item.get('description', 'Stylish piece')[:35]}**")
            
            is_selected = st.session_state.selected_anchor_idx == i
            if st.checkbox("Select this", key=f"anchor_{i}", value=is_selected):
                st.session_state.selected_anchor_idx = i
            elif is_selected:
                st.session_state.selected_anchor_idx = None
    
    if shown < len(results) and shown < 20:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Show 4 More ↓"):
                st.session_state.search_results_shown = min(shown + 4, 20)
                st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Build My Ensemble →", type="primary", use_container_width=True):
            if st.session_state.selected_anchor_idx is None:
                st.error("Please select an anchor piece first")
            else:
                st.session_state.page = 'results'
                st.rerun()


def render_results_page():
    """Render ensemble results."""
    
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        st.markdown('<h1 class="main-header">🎨 Your Ensemble</h1>', unsafe_allow_html=True)
    with col2:
        if st.button("← New"):
            st.session_state.page = 'landing'
            st.session_state.ensemble_results = None
            st.session_state.selected_items = {'top': [], 'bottom': [], 'accessory': [], 'coordinating': []}
            st.session_state.shown_items = {'top': 4, 'bottom': 4, 'accessory': 4, 'coordinating': 4}
            st.session_state.search_results = []
            st.session_state.selected_anchor_idx = None
            st.session_state.generated_queries = []
            # Clear multimodal state
            st.session_state.uploaded_image = None
            st.session_state.uploaded_image_embedding = None
            st.session_state.uploaded_image_features = None
            st.session_state.anchor_from_upload = False
            st.rerun()
    with col3:
        if st.button("💾 Save"):
            st.toast("Save functionality coming soon!")
    
    parsed = st.session_state.parsed_query or {}
    
    # Query box
    st.markdown(f"""
    <div class="query-box">
        <strong>🎯 Styling:</strong> {st.session_state.user_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Show defaults applied warning
    defaults_applied = parsed.get('_defaults_applied', [])
    if defaults_applied:
        st.info(f"ℹ️ **Assumed:** {', '.join(defaults_applied)} (not specified in your request)")
    
    # LLM Debug expander
    with st.expander("🧠 LLM Understanding & Generated Queries", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Parsed from your request:**")
            # Remove internal field before displaying
            display_parsed = {k: v for k, v in parsed.items() if not k.startswith('_')}
            st.json(display_parsed)
            
            # Show visual features if available
            if st.session_state.uploaded_image_features:
                st.markdown("**Visual features (from CLIP):**")
                st.json(st.session_state.uploaded_image_features)
                
                # Note any augmentation
                if parsed.get('_visual_color_used'):
                    st.caption("ℹ️ Color detected from image")
                if parsed.get('_visual_garment_used'):
                    st.caption("ℹ️ Garment type detected from image")
        
        with col2:
            st.markdown("**Generated CLIP queries:**")
            if st.session_state.generated_queries:
                for q in st.session_state.generated_queries:
                    badge = "🔥" if q['type'] == 'bold' else "✓"
                    st.text(f"{badge} [{q['category']}] {q['query'][:50]}...")
            else:
                st.caption("Queries will appear after generation")
    
    # Generate ensemble
    if st.session_state.ensemble_results is None:
        with st.spinner("Building your ensemble..."):
            ensemble = generate_ensemble(parsed)
            st.session_state.ensemble_results = ensemble
    
    ensemble = st.session_state.ensemble_results
    
    # Anchor display - handle both uploaded image and FAISS selection
    st.markdown("### Your Anchor Piece")
    
    if st.session_state.anchor_from_upload and st.session_state.uploaded_image is not None:
        # Anchor is the uploaded image
        col1, col2 = st.columns([1, 3])
        with col1:
            try:
                img = Image.open(st.session_state.uploaded_image)
                st.image(img, width=180)
            except Exception as e:
                st.warning("Could not display uploaded image")
        with col2:
            # Show visual features if extracted
            visual_features = st.session_state.uploaded_image_features or {}
            color = parsed.get('anchor_color', visual_features.get('color', ''))
            garment = parsed.get('anchor_garment', visual_features.get('garment_type', ''))
            
            st.markdown(f"**Your {color.title()} {garment.title()}**")
            st.caption("📷 Your uploaded image - building outfit around this item")
            
            # Show visual analysis if available
            if visual_features:
                with st.expander("👁️ Visual Analysis (CLIP)", expanded=False):
                    st.json(visual_features)
                    
    elif st.session_state.selected_anchor_idx is not None and st.session_state.search_results:
        # Anchor is from FAISS selection
        anchor = st.session_state.search_results[st.session_state.selected_anchor_idx]
        col1, col2 = st.columns([1, 3])
        with col1:
            try:
                st.image(Image.open(anchor['path']), width=180)
            except:
                pass
        with col2:
            st.markdown(f"**{anchor.get('description', 'Your selected piece')}**")
            st.caption("🎯 Selected from search - building outfit around this item")
    else:
        # No anchor selected - this shouldn't happen normally
        st.info("No anchor piece selected. The ensemble is based on your text description.")
    
    # Categories - including coordinating pieces for ethnic wear
    for category, title in [('coordinating', '✨ COORDINATING PIECES'), ('top', 'TOPS'), ('bottom', 'BOTTOMS'), ('accessory', 'ACCESSORIES')]:
        if category not in ensemble or not ensemble[category]:
            continue
        
        items = ensemble[category]
        shown = st.session_state.shown_items.get(category, 4)
        
        st.markdown(f"### {title}")
        st.caption(f"Select up to 2 items")
        
        cols = st.columns(4)
        for i, item in enumerate(items[:shown]):
            with cols[i % 4]:
                query = item.get('query', {})
                is_bold = query.get('is_bold', False)
                description = item.get('description', '')
                
                try:
                    st.image(Image.open(item['path']), use_container_width=True)
                except:
                    st.image("https://via.placeholder.com/150x200", use_container_width=True)
                
                color = query.get('color', '')
                garment = query.get('garment_type', '')
                st.markdown(f"**{color.title()} {garment.title()}"[:25] + "**")
                st.caption(description[:45] if description else '')
                
                if is_bold:
                    st.markdown('<span class="bold-badge">⚡ BOLD PICK</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="safe-badge">Safe choice</span>', unsafe_allow_html=True)
                
                is_selected = i in st.session_state.selected_items.get(category, [])
                current_count = len(st.session_state.selected_items.get(category, []))
                disabled = current_count >= 2 and not is_selected
                
                if st.checkbox("Select", key=f"sel_{category}_{i}", value=is_selected, disabled=disabled):
                    if i not in st.session_state.selected_items[category]:
                        if len(st.session_state.selected_items[category]) < 2:
                            st.session_state.selected_items[category].append(i)
                else:
                    if i in st.session_state.selected_items[category]:
                        st.session_state.selected_items[category].remove(i)
        
        if shown < len(items):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button(f"Show 4 More {title} ↓", key=f"more_{category}"):
                    st.session_state.shown_items[category] = min(shown + 4, 20)
                    st.rerun()
    
    # Styling tip
    color = parsed.get('anchor_color', 'your chosen color') or 'your item'
    st.markdown(f"""
    <div class="styling-tip">
        <strong>💡 Styling tip:</strong> {color.title()} pairs beautifully with both 
        neutral tones for everyday elegance and bold accents for standout moments.
    </div>
    """, unsafe_allow_html=True)
    
    # Counter
    tops = len(st.session_state.selected_items['top'])
    bottoms = len(st.session_state.selected_items['bottom'])
    accessories = len(st.session_state.selected_items['accessory'])
    coordinating = len(st.session_state.selected_items.get('coordinating', []))
    total = tops + bottoms + accessories + coordinating
    
    coord_text = f", {coordinating} coordinating" if coordinating > 0 else ""
    
    st.markdown(f"""
    <div class="selection-counter">
        <strong>Selected:</strong> {tops} tops, {bottoms} bottoms, {accessories} accessories{coord_text}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📥 Create Final Look", type="primary", use_container_width=True, disabled=total == 0):
            # Generate the final look collage and prompt
            with st.spinner("Creating your final look..."):
                try:
                    from final_look import create_final_look_collage, generate_shopping_prompt, generate_shopping_prompt_compact
                    
                    # Gather selected items with full info
                    selected_items_full = {'top': [], 'bottom': [], 'accessory': [], 'coordinating': []}
                    
                    for category in ['top', 'bottom', 'accessory', 'coordinating']:
                        selected_indices = st.session_state.selected_items.get(category, [])
                        ensemble_items = ensemble.get(category, [])
                        
                        for idx in selected_indices:
                            if idx < len(ensemble_items):
                                selected_items_full[category].append(ensemble_items[idx])
                    
                    # Get anchor item info - handle both uploaded image and FAISS selection
                    anchor_item = None
                    if st.session_state.anchor_from_upload and st.session_state.uploaded_image is not None:
                        # Anchor is the uploaded image
                        # Save uploaded image temporarily to get a path for the collage
                        import tempfile
                        import os
                        
                        try:
                            img = Image.open(st.session_state.uploaded_image)
                            temp_dir = tempfile.gettempdir()
                            temp_path = os.path.join(temp_dir, "ensemble_ai_anchor.png")
                            img.save(temp_path)
                            
                            anchor_item = {
                                'path': temp_path,
                                'description': f"{parsed.get('anchor_color', '')} {parsed.get('anchor_garment', '')}".strip().title()
                            }
                        except Exception as e:
                            print(f"[Final Look] Error saving anchor image: {e}")
                            
                    elif st.session_state.selected_anchor_idx is not None and st.session_state.search_results:
                        # Anchor is from FAISS selection
                        anchor_item = st.session_state.search_results[st.session_state.selected_anchor_idx]
                    
                    # Create collage
                    collage_image = create_final_look_collage(
                        selected_items=selected_items_full,
                        anchor_item=anchor_item,
                        parsed_query=parsed
                    )
                    st.session_state.final_look_collage = collage_image
                    
                    # Generate prompts
                    st.session_state.final_look_prompt = generate_shopping_prompt(
                        selected_items=selected_items_full,
                        anchor_item=anchor_item,
                        parsed_query=parsed
                    )
                    
                    st.session_state.final_look_prompt_compact = generate_shopping_prompt_compact(
                        selected_items=selected_items_full,
                        parsed_query=parsed
                    )
                    
                    # Navigate to final look page
                    st.session_state.page = 'final_look'
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating final look: {e}")
                    import traceback
                    st.code(traceback.format_exc())


def render_final_look_page():
    """Render the final look page with collage and shopping prompt."""
    import io
    
    # Force scroll to top on page load
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    
    # Header with navigation
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        st.markdown('<h1 class="main-header">📸 Your Final Look</h1>', unsafe_allow_html=True)
    with col2:
        if st.button("← Back"):
            st.session_state.page = 'results'
            st.rerun()
    with col3:
        if st.button("🔄 New"):
            # Reset all state
            st.session_state.page = 'landing'
            st.session_state.ensemble_results = None
            st.session_state.selected_items = {'top': [], 'bottom': [], 'accessory': [], 'coordinating': []}
            st.session_state.shown_items = {'top': 4, 'bottom': 4, 'accessory': 4, 'coordinating': 4}
            st.session_state.search_results = []
            st.session_state.selected_anchor_idx = None
            st.session_state.generated_queries = []
            st.session_state.final_look_collage = None
            st.session_state.final_look_prompt = None
            st.session_state.final_look_prompt_compact = None
            # Clear multimodal state
            st.session_state.uploaded_image = None
            st.session_state.uploaded_image_embedding = None
            st.session_state.uploaded_image_features = None
            st.session_state.anchor_from_upload = False
            st.rerun()
    
    st.markdown("""
    <p class="sub-header">
        Your complete outfit - download the collage and use the shopping prompt with any AI assistant.
    </p>
    """, unsafe_allow_html=True)
    
    # Check if we have the collage
    if st.session_state.final_look_collage is None:
        st.warning("No final look generated yet. Please go back and select items.")
        if st.button("← Go to Results"):
            st.session_state.page = 'results'
            st.rerun()
        return
    
    # Display the collage - FULL WIDTH
    st.markdown("### 👗 Your Outfit Collage")
    
    # Full width collage display
    st.image(st.session_state.final_look_collage, use_container_width=True)
    
    # Download button centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        buf = io.BytesIO()
        st.session_state.final_look_collage.save(buf, format='PNG')
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Collage (PNG)",
            data=buf,
            file_name="ensemble_ai_outfit.png",
            mime="image/png",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Shopping prompt section
    st.markdown("### 🛍️ Shopping Prompt")
    st.caption("Copy this prompt and paste it into ChatGPT, Gemini, or Perplexity along with the downloaded image.")
    
    # Prompt style selector
    prompt_style = st.radio(
        "Prompt style:",
        ["Detailed", "Compact"],
        horizontal=True,
        help="Detailed: Full structured request with all item details. Compact: Shorter, conversational style."
    )
    
    # Display the appropriate prompt
    if prompt_style == "Detailed":
        prompt_text = st.session_state.final_look_prompt
    else:
        prompt_text = st.session_state.final_look_prompt_compact
    
    # Display prompt in a text area (read-only style)
    st.text_area(
        "Shopping prompt",
        value=prompt_text,
        height=400,
        label_visibility="collapsed",
        disabled=False  # Allow selection for manual copy
    )
    
    # Copy button using clipboard.js approach (works via st.code alternative)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Streamlit's st.code has a built-in copy button
        if st.button("📋 Copy Prompt to Clipboard", use_container_width=True, type="primary"):
            st.code(prompt_text, language=None)
            st.success("✅ Prompt displayed above with copy button - click the copy icon!")
    
    st.markdown("---")
    
    # Instructions
    st.markdown("### 📝 How to Use")
    
    st.markdown("""
    <div class="styling-tip">
        <strong>Step 1:</strong> Download the outfit collage using the button above<br><br>
        <strong>Step 2:</strong> Copy the shopping prompt<br><br>
        <strong>Step 3:</strong> Go to your preferred AI assistant:
        <ul>
            <li>🤖 <a href="https://chat.openai.com" target="_blank">ChatGPT</a> (GPT-4 with vision)</li>
            <li>🌟 <a href="https://gemini.google.com" target="_blank">Google Gemini</a></li>
            <li>🔍 <a href="https://perplexity.ai" target="_blank">Perplexity AI</a></li>
        </ul>
        <strong>Step 4:</strong> Upload the collage image and paste the prompt<br><br>
        <strong>Step 5:</strong> Get personalized shopping recommendations with links!
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    init_session_state()
    
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI models..."):
            try:
                model, tokenizer, preprocess, index, valid_paths, sources, device = load_models()
                
                if model is not None:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.preprocess = preprocess
                    st.session_state.index = index
                    st.session_state.valid_paths = valid_paths
                    st.session_state.sources = sources
                    st.session_state.device = device
                    st.session_state.model_loaded = True
                else:
                    st.error("Failed to load CLIP models. Check data paths.")
                    return
            except Exception as e:
                st.error(f"Error: {e}")
                return
    
    if st.session_state.page == 'landing':
        render_landing_page()
    elif st.session_state.page == 'search':
        render_search_page()
    elif st.session_state.page == 'results':
        render_results_page()
    elif st.session_state.page == 'final_look':
        render_final_look_page()


if __name__ == "__main__":
    main()