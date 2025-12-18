# 🎨 Ensemble AI - Fashion Outfit Builder

Build complete outfits from any piece in your wardrobe using AI.

## Features

- **Text + Image Input**: Describe your item or upload a photo
- **"I have" vs "Looking for" modes**: Build ensembles from owned items or find new pieces
- **Context-aware recommendations**: Office, party, wedding, ethnic wear support
- **119K+ fashion items**: Combined Myntra + H&M catalog
- **Bold + Safe picks**: Color-theory based recommendations
- **Ethnic wear mode**: Curated Indian traditional styles

## Setup

### Prerequisites

- Python 3.10+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone/copy the app
cd ~/projects/fashion-ensemble-builder

# Install dependencies
pip install -r app/requirements.txt

# Ensure data files exist
ls data/embeddings/
# Should show: combined_faiss.index, combined_paths.txt, combined_sources.txt
```

### Run

```bash
streamlit run app/app.py
```

App will be available at `http://localhost:8501`

## File Structure

```
fashion-ensemble-builder/
├── app/
│   ├── app.py              # Main Streamlit app
│   └── requirements.txt    # Dependencies
├── src/
│   ├── color_theory.py     # Color matching logic
│   ├── ensemble_rules.py   # Outfit building rules
│   └── pipeline.py         # ML pipeline
├── data/
│   ├── raw/
│   │   ├── myntra/         # Myntra images
│   │   └── hm/             # H&M images
│   └── embeddings/
│       ├── combined_faiss.index
│       ├── combined_paths.txt
│       └── combined_sources.txt
```

## Usage

1. **Quick Start**: Click chips to select garment type, color, occasion
2. **Text Input**: Type natural language like "I have a navy blazer for office"
3. **Upload**: Add image to help match your style
4. **Ethnic Mode**: Toggle for Indian traditional wear (Myntra-only catalog)
5. **Results**: Select up to 2 items per category, use "Show More" for alternatives

## Coming Soon

- Final look collage generation
- Shopping links
- Save/share outfits
- Image-based anchor detection