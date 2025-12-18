"""
Multimodal Search Module for Fashion Ensemble AI

Provides:
1. Image embedding extraction via CLIP
2. Text embedding extraction via CLIP  
3. Combined multimodal search (image + text)
4. Visual feature extraction from images

Author: Fashion Ensemble Builder
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


# =============================================================================
# IMAGE EMBEDDING
# =============================================================================

def get_image_embedding(
    image: Union[Image.Image, str, Path],
    model,
    preprocess,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract CLIP embedding from an image.
    
    Args:
        image: PIL Image, or path to image file
        model: CLIP model (already loaded)
        preprocess: CLIP preprocessing function
        device: torch device
        
    Returns:
        Normalized embedding as numpy array (1, embedding_dim)
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Expected PIL Image or path, got {type(image)}")
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess and encode
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy().astype(np.float32)


def get_text_embedding(
    text: str,
    model,
    tokenizer,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract CLIP embedding from text.
    
    Args:
        text: Text query
        model: CLIP model
        tokenizer: CLIP tokenizer
        device: torch device
        
    Returns:
        Normalized embedding as numpy array (1, embedding_dim)
    """
    text_tokens = tokenizer([text]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy().astype(np.float32)


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def search_by_image(
    image: Union[Image.Image, str, Path],
    model,
    preprocess,
    index,
    valid_paths: List[str],
    sources: List[str],
    device: str = 'cpu',
    k: int = 20,
    ethnic_mode: bool = False
) -> List[Dict]:
    """
    Search FAISS index using image embedding.
    
    Args:
        image: Query image
        model: CLIP model
        preprocess: CLIP preprocess function
        index: FAISS index
        valid_paths: List of image paths in index
        sources: List of sources (myntra/hm) for each index entry
        device: torch device
        k: Number of results
        ethnic_mode: If True, filter to myntra only
        
    Returns:
        List of result dicts with path, score, source
    """
    # Get image embedding
    image_embedding = get_image_embedding(image, model, preprocess, device)
    
    # Search
    search_k = k * 3 if ethnic_mode else k
    distances, indices = index.search(image_embedding, k=search_k)
    
    results = []
    for d, idx in zip(distances[0], indices[0]):
        source = sources[idx]
        if ethnic_mode and source != 'myntra':
            continue
        results.append({
            'path': valid_paths[idx],
            'score': float(d),
            'source': source
        })
        if len(results) >= k:
            break
    
    return results


def search_multimodal(
    text_query: Optional[str],
    image: Optional[Union[Image.Image, str, Path]],
    model,
    tokenizer,
    preprocess,
    index,
    valid_paths: List[str],
    sources: List[str],
    device: str = 'cpu',
    k: int = 20,
    text_weight: float = 0.5,
    image_weight: float = 0.5,
    ethnic_mode: bool = False
) -> List[Dict]:
    """
    Combined text + image search using weighted embedding fusion.
    
    Args:
        text_query: Text description (optional)
        image: Query image (optional)
        model: CLIP model
        tokenizer: CLIP tokenizer
        preprocess: CLIP preprocess function
        index: FAISS index
        valid_paths: Image paths
        sources: Source labels
        device: torch device
        k: Number of results
        text_weight: Weight for text embedding (0-1)
        image_weight: Weight for image embedding (0-1)
        ethnic_mode: Filter to myntra only
        
    Returns:
        List of result dicts
    """
    # Get embeddings
    text_embedding = None
    image_embedding = None
    
    if text_query:
        text_embedding = get_text_embedding(text_query, model, tokenizer, device)
    
    if image is not None:
        image_embedding = get_image_embedding(image, model, preprocess, device)
    
    # Combine embeddings
    if text_embedding is not None and image_embedding is not None:
        # Weighted combination
        combined = text_weight * text_embedding + image_weight * image_embedding
        # Re-normalize
        combined = combined / np.linalg.norm(combined, axis=-1, keepdims=True)
    elif text_embedding is not None:
        combined = text_embedding
    elif image_embedding is not None:
        combined = image_embedding
    else:
        raise ValueError("Must provide either text_query or image")
    
    # Search
    search_k = k * 3 if ethnic_mode else k
    distances, indices = index.search(combined.astype(np.float32), k=search_k)
    
    results = []
    for d, idx in zip(distances[0], indices[0]):
        source = sources[idx]
        if ethnic_mode and source != 'myntra':
            continue
        results.append({
            'path': valid_paths[idx],
            'score': float(d),
            'source': source
        })
        if len(results) >= k:
            break
    
    return results


# =============================================================================
# VISUAL FEATURE EXTRACTION
# =============================================================================

# Predefined fashion attributes for zero-shot classification
COLOR_LABELS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", 
    "black", "white", "grey", "gray", "brown", "beige", "cream",
    "navy", "maroon", "burgundy", "teal", "coral", "olive",
    "mustard", "rust", "lavender", "mint", "peach", "gold", "silver"
]

GARMENT_LABELS = [
    # Tops
    "t-shirt", "shirt", "blouse", "polo", "tank top", "crop top",
    "sweater", "hoodie", "cardigan", "turtleneck", "kurta", "kurti",
    # Bottoms
    "jeans", "trousers", "pants", "shorts", "skirt", "leggings",
    "palazzos", "churidar", "salwar",
    # Outerwear
    "blazer", "jacket", "coat", "vest",
    # Full body
    "dress", "jumpsuit", "saree", "lehenga",
]

STYLE_LABELS = [
    "casual", "formal", "elegant", "sporty", "bohemian", "vintage",
    "modern", "classic", "trendy", "minimalist", "ethnic", "western"
]

PATTERN_LABELS = [
    "solid", "striped", "checked", "plaid", "floral", "printed",
    "polka dot", "geometric", "abstract", "embroidered", "plain"
]


def extract_visual_features(
    image: Union[Image.Image, str, Path],
    model,
    tokenizer,
    preprocess,
    device: str = 'cpu'
) -> Dict[str, str]:
    """
    Extract fashion attributes from image using CLIP zero-shot classification.
    
    Args:
        image: Input image
        model: CLIP model
        tokenizer: CLIP tokenizer
        preprocess: CLIP preprocess
        device: torch device
        
    Returns:
        Dict with detected color, garment_type, style, pattern
    """
    # Get image embedding
    image_embedding = get_image_embedding(image, model, preprocess, device)
    image_embedding = torch.from_numpy(image_embedding).to(device)
    
    features = {}
    
    # Classify each attribute
    for attr_name, labels in [
        ('color', COLOR_LABELS),
        ('garment_type', GARMENT_LABELS),
        ('style', STYLE_LABELS),
        ('pattern', PATTERN_LABELS)
    ]:
        # Create text prompts
        if attr_name == 'color':
            prompts = [f"a {label} colored garment" for label in labels]
        elif attr_name == 'garment_type':
            prompts = [f"a photo of a {label}" for label in labels]
        elif attr_name == 'style':
            prompts = [f"a {label} style outfit" for label in labels]
        else:
            prompts = [f"a {label} pattern garment" for label in labels]
        
        # Get text embeddings
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        similarity = (image_embedding @ text_features.T).squeeze(0)
        best_idx = similarity.argmax().item()
        features[attr_name] = labels[best_idx]
    
    return features


def get_image_description(
    image: Union[Image.Image, str, Path],
    model,
    tokenizer,
    preprocess,
    device: str = 'cpu'
) -> str:
    """
    Generate a text description of the garment in the image.
    
    Args:
        image: Input image
        model: CLIP model
        tokenizer: CLIP tokenizer  
        preprocess: CLIP preprocess
        device: torch device
        
    Returns:
        Description string like "navy blue formal shirt"
    """
    features = extract_visual_features(image, model, tokenizer, preprocess, device)
    
    color = features.get('color', '')
    garment = features.get('garment_type', '')
    style = features.get('style', '')
    pattern = features.get('pattern', '')
    
    # Build description
    parts = []
    if color:
        parts.append(color)
    if pattern and pattern != 'solid' and pattern != 'plain':
        parts.append(pattern)
    if style and style not in ['casual', 'western']:  # Skip generic styles
        parts.append(style)
    if garment:
        parts.append(garment)
    
    return ' '.join(parts) if parts else 'garment'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_similar_items(
    anchor_image: Union[Image.Image, str, Path],
    model,
    preprocess,
    index,
    valid_paths: List[str],
    sources: List[str],
    device: str = 'cpu',
    k: int = 10,
    exclude_self: bool = True
) -> List[Dict]:
    """
    Find items visually similar to anchor image.
    
    Useful for finding database matches for an uploaded image.
    """
    results = search_by_image(
        image=anchor_image,
        model=model,
        preprocess=preprocess,
        index=index,
        valid_paths=valid_paths,
        sources=sources,
        device=device,
        k=k + (1 if exclude_self else 0)
    )
    
    # Optionally exclude exact match (score ~1.0)
    if exclude_self and results and results[0]['score'] > 0.99:
        results = results[1:]
    
    return results[:k]


def compute_embedding_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> float:
    """
    Compute cosine similarity between two embeddings.
    """
    # Normalize
    e1 = embedding1 / np.linalg.norm(embedding1)
    e2 = embedding2 / np.linalg.norm(embedding2)
    
    return float(np.dot(e1.flatten(), e2.flatten()))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core functions
    'get_image_embedding',
    'get_text_embedding',
    'search_by_image',
    'search_multimodal',
    # Feature extraction
    'extract_visual_features',
    'get_image_description',
    # Utilities
    'find_similar_items',
    'compute_embedding_similarity',
    # Constants
    'COLOR_LABELS',
    'GARMENT_LABELS',
    'STYLE_LABELS',
    'PATTERN_LABELS',
]