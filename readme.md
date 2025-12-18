# 🎨 Ensemble AI

### Multimodal Fashion Styling Assistant

<p align="center">
  <strong>Build complete, coordinated outfits using AI-powered visual understanding and natural language</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#installation">Installation</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#acknowledgments">Acknowledgments</a>
</p>

---

## 🌟 Overview

Ensemble AI is a **multimodal fashion recommendation system** that combines computer vision (CLIP) with natural language processing (Phi-3) to help users build complete, coordinated outfits.

**Input**: Upload a garment image + describe your style needs in plain English  
**Output**: A complete outfit with matching tops, bottoms, and accessories

### Key Differentiators

- **True Multimodal**: Fuses image understanding with text intent
- **Color Theory**: Applies complementary/analogous color matching
- **100K+ Items**: Searches across massive fashion catalog instantly
- **Ethnic Wear Support**: Special handling for sarees, lehengas, kurtas

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📷 **Image Upload** | Upload any garment as your anchor piece |
| 💬 **Natural Language** | Describe style needs in plain English |
| 🎨 **Color Matching** | AI applies color theory for harmony |
| 👗 **Complete Ensembles** | Tops, bottoms, accessories - coordinated |
| 🪔 **Ethnic Wear** | Indian fashion with coordinating pieces |
| 📸 **Outfit Collage** | Downloadable outfit card |
| 🛍️ **Shopping Prompts** | AI-ready prompts for online shopping |

---

## 🏗️ How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INPUT                              │
│            (Image Upload + Text Description)                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Marqo CLIP     │     │   Phi-3 LLM     │
│  (Vision)       │     │   (Language)    │
│                 │     │                 │
│ • Color detect  │     │ • Parse intent  │
│ • Garment type  │     │ • Occasion      │
│ • Style extract │     │ • Gender/Style  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
        ┌────────────────────────┐
        │    FEATURE FUSION      │
        │  Visual augments text  │
        └───────────┬────────────┘
                    ▼
        ┌────────────────────────┐
        │   FAISS Vector Search  │
        │   (100K+ embeddings)   │
        └───────────┬────────────┘
                    ▼
        ┌────────────────────────┐
        │  ENSEMBLE GENERATOR    │
        │  + Color Theory Engine │
        └───────────┬────────────┘
                    ▼
        ┌────────────────────────┐
        │       OUTPUT           │
        │  Outfit Collage +      │
        │  Shopping Prompt       │
        └────────────────────────┘
```

---

## 🧠 Tech Stack

### Vision: Marqo FashionCLIP

We use [**Marqo's FashionCLIP**](https://huggingface.co/Marqo/marqo-fashionCLIP), a CLIP model fine-tuned on **800K+ fashion image-text pairs** from fashion datasets.

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:Marqo/marqo-fashionCLIP'
)
```

**Capabilities:**
- Zero-shot color/garment/style classification
- 512-dimensional fashion-aware embeddings
- Cross-modal image-text matching

### Language: Microsoft Phi-3

[**Phi-3 Mini**](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) via Ollama parses natural language requests into structured queries.

```python
# "I have a navy blazer for a business meeting"
# → {"anchor_garment": "blazer", "anchor_color": "navy", "occasion": "office"}
```

### Search: FAISS

Facebook's [**FAISS**](https://github.com/facebookresearch/faiss) enables sub-millisecond similarity search across 100K+ fashion embeddings.

### Frontend: Streamlit

[**Streamlit**](https://streamlit.io/) powers the interactive web interface.

---

## 📊 Datasets

This project uses fashion image datasets from Kaggle for research and demonstration purposes.

### Myntra Fashion Dataset
- **Source**: [Kaggle - Myntra Fashion Product Dataset](https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset)
- **Size**: ~45,000 images
- **Content**: Indian fashion, ethnic wear, western clothing
- **Author**: Hitesh Suthar

### H&M Fashion Dataset  
- **Source**: [Kaggle - H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
- **Size**: ~60,000 images (subset used)
- **Content**: Western fashion, basics, accessories
- **Provider**: H&M Group

### ⚠️ Dataset Setup

The image datasets are **not included** in this repository due to size (~10GB). To use this project:

1. Download datasets from the Kaggle links above
2. Extract images to:
   ```
   data/raw/myntra/images/
   data/raw/hm/images/
   ```
3. Run the embedding generation script (see Installation)

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- 8GB+ RAM
- [Ollama](https://ollama.ai/) for LLM

### Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ensemble-ai.git
cd ensemble-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Phi-3 model
ollama pull phi3

# Download datasets from Kaggle (see Dataset section)
# Place images in data/raw/myntra/images/ and data/raw/hm/images/

# Generate embeddings (first time only)
python scripts/generate_embeddings.py

# Run the app
streamlit run app/app.py
```

### Directory Structure

```
ensemble-ai/
├── app/
│   ├── app.py                 # Main Streamlit application
│   ├── final_look.py          # Collage generation
│   ├── garment_normalizer.py  # Garment name normalization (400+ mappings)
│   ├── multimodal_search.py   # CLIP image processing
│   ├── color_theory.py        # Color matching rules
│   └── ensemble_rules.py      # Outfit generation logic
├── data/
│   ├── embeddings/            # FAISS index + metadata (generated)
│   └── raw/                   # Image datasets (download from Kaggle)
├── scripts/
│   └── generate_embeddings.py # Creates FAISS index
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t ensemble-ai .

# Run container
docker run -p 8501:8501 ensemble-ai
```

---

## ☁️ Cloud Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed GCP Cloud Run deployment instructions.

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Image encoding | ~50ms (GPU) / ~500ms (CPU) |
| FAISS search (100K items) | <5ms |
| LLM parsing | ~1-2s |
| **Total response time** | **~5-8s** |

---

## 🙏 Acknowledgments

This project builds on the incredible work of:

### Models

| Model | Authors | License | Link |
|-------|---------|---------|------|
| **Marqo FashionCLIP** | Marqo AI | Apache 2.0 | [HuggingFace](https://huggingface.co/Marqo/marqo-fashionCLIP) |
| **Phi-3 Mini** | Microsoft | MIT | [HuggingFace](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| **OpenCLIP** | LAION | MIT | [GitHub](https://github.com/mlfoundations/open_clip) |
| **FAISS** | Meta AI | MIT | [GitHub](https://github.com/facebookresearch/faiss) |

### Datasets

| Dataset | Provider | License | Link |
|---------|----------|---------|------|
| **Myntra Fashion** | Hitesh Suthar | CC0 | [Kaggle](https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset) |
| **H&M Fashion** | H&M Group | Competition | [Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) |

### Frameworks

- [Streamlit](https://streamlit.io/) - Web framework
- [Ollama](https://ollama.ai/) - Local LLM inference
- [PyTorch](https://pytorch.org/) - Deep learning

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The datasets used have their own licenses. Please review the Kaggle dataset pages for usage terms.

---

## 🛣️ Roadmap

- [ ] Virtual try-on integration
- [ ] User preference learning
- [ ] Price-aware recommendations
- [ ] Multi-language support
- [ ] API endpoint

---

## 👤 Author

**Saurav Sharma**
- LinkedIn: [Your LinkedIn]
- Email: [Your Email]

---

<p align="center">
  Made with ❤️ for fashion enthusiasts
</p>
