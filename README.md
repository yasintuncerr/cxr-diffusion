# CXR-Diffusion: Synthetic Chest X-Ray Generation Using Multiple Conditioning Approaches

## Project Overview

CXR-Diffusion is a research project focused on generating high-quality synthetic chest X-ray images using various diffusion model approaches. The project explores and compares different conditioning methods to improve the quality and clinical relevance of generated images.

### Key Objectives
- Generate realistic synthetic chest X-ray images using diffusion models
- Compare different conditioning approaches for image generation
- Evaluate the quality and clinical accuracy of generated images
- Provide insights into the most effective methods for medical image synthesis

### Conditioning Approaches
1. **CLIP Text Encoder Conditioning**
   - Utilizing CLIP's text encoder for text-guided image generation
   - Enabling natural language descriptions to control image synthesis

2. **CLIP Vision Encoder Conditioning**
   - Leveraging CLIP's vision encoder for image-guided generation
   - Allowing reference-based image synthesis

3. **ELIXR Feature Conditioning**
   - Using ELIXR-B text embeddings for specialized medical text understanding
   - Employing ELIXR-C image embeddings for CXR-specific visual features
   - Combining both for enhanced multimodal conditioning

### Methodology
- Implementation of multiple diffusion model architectures
- Comparative analysis of different conditioning methods
- Quantitative and qualitative evaluation of generated images
- Assessment of clinical relevance and accuracy

## Dataset Structure

The project utilizes a comprehensive dataset organized as follows:

```
nih-cxr/
├── original/                 # Original NIH Chest X-rays with renamed files
├── {img_size}x{img_size}/   # Resized versions of original images[Optional]
├── elixrb/                  # Text embedding vectors from ELIXR-B
├── elixrc/                  # Image embedding vectors from ELIXR-C
└── google-labels/           # Expert annotations from Google Healthcare
```

### Components Description

#### Original Images
- Contains the original NIH Chest X-ray images
- Source: [NIH Chest X-rays Dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- Images have been renamed from their original format for consistency

#### Resized Images
- Located in directory named according to target resolution (e.g., "512x512")
- Derived from the original images
- Maintains aspect ratio while resizing to target dimensions
- Useful for standardized input sizes in deep learning models

#### ELIXR-B Text Embeddings
- Text embedding vectors generated using the CXR Foundation ELIXR-B model
- Available for download: [NIH-CXR14 ELIXR-B Text Embeddings](https://huggingface.co/datasets/8bits-ai/nih-cxr14-elixr-b-text-embeddings)
- Useful for text-based analysis and multimodal learning

#### ELIXR-C Image Embeddings
- Image embedding vectors generated using the CXR Foundation ELIXR-C model
- Available for download: [NIH-CXR14 ELIXR-C V2 Embeddings](https://huggingface.co/datasets/8bits-ai/nih-cxr14-elixr-c-v2-embeddings)
- Suitable for transfer learning and image similarity tasks

#### Google Expert Labels
- Additional expert annotations provided by Google
- Source: [Google Cloud Healthcare API Public Datasets](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest)
- Provides high-quality ground truth labels for model validation

## Usage Notes

1. When using the dataset, ensure you cite both the original NIH dataset and any additional resources used (ELIXR models, Google labels).
2. The resized images are provided for convenience but always validate against the original images for clinical applications.
3. The embedding vectors can be used directly for transfer learning or as features for downstream tasks.

## Acknowledgments

- National Institutes of Health (NIH) for the original chest X-ray dataset
- Google Healthcare for expert annotations
- CXR Foundation for the ELIXR-B and ELIXR-C models
- OpenAI for the CLIP model

## License

Please refer to the original data sources for specific licensing information:
- NIH Chest X-rays: [Kaggle Dataset License](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- Google Labels: [Google Cloud Terms of Service](https://cloud.google.com/terms)
- ELIXR Embeddings: Check respective HuggingFace repository licenses