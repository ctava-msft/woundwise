# WoundWise: Visual Question Answering for Wound Care

WoundWise is a visual question answering (VQA) system designed for wound care applications, extending the MEDIQA-M3G 2024 shared task. The system generates free-text responses to patient queries accompanied by wound images, helping healthcare providers deliver faster and higher-quality remote care.

## Overview

This project addresses the growing burden on healthcare providers in remote patient care by providing AI-assisted draft responses to patient wound care queries. The system processes both visual (wound images) and textual (patient questions) inputs to generate medically relevant responses.

## Dataset

The WoundWise dataset contains 500 questions and answers in English and Chinese, with the following structure:

### Data Statistics
| Split | Instances | Responses | Images |
|-------|-----------|-----------|--------|
| Train | 279       | 279       | 449    |
| Valid | 105       | 210       | 147    |
| Test  | 93        | 279       | 152    |

### Data Format
Each query instance includes:
- `encounter_id`: Unique identifier (e.g., 'ENC0001')
- `query_title_{LANG}`: Query title in specified language
- `query_content_{LANG}`: Query content in specified language
- `image_ids`: List of associated image IDs
- `responses`: List of response objects with author_id and content
- **Medical metadata**:
  - `anatomic_locations`: List of anatomic locations
  - `wound_type`: Wound type category
  - `wound_thickness`: Wound thickness category  
  - `tissue_color`: Tissue color category
  - `drainage_amount`: Drainage amount category
  - `drainage_type`: Drainage type category
  - `infection`: Infection category

### Text Statistics

**English (en) - Average Words:**
| Split | Query | Response |
|-------|-------|----------|
| Train | 46    | 29       |
| Valid | 44    | 41       |
| Test  | 52    | 47       |

**Chinese (zh) - Average Characters:**
| Split | Query | Response |
|-------|-------|----------|
| Train | 52    | 43       |
| Valid | 50    | 60       |
| Test  | 60    | 68       |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ctava-msft/woundwise.git
```

2. Install dependencies:
# setup venv environment

```
python -m venv .venv
./.venv/Scripts/activate
pip install -r requirements.txt
```

```bash
pip install -r requirements.txt
```

```

## Usage

### Data Loading

```python
from data_loader import create_data_loaders

# Create data loaders
train_loader, valid_loader, test_loader = create_data_loaders(
    data_path="path/to/dataset",
    images_path="path/to/images",
    language="en",  # or "zh" for Chinese
    batch_size=8
)
```

### Training

Train the VQA model using the provided training script:

```bash
python train.py \
    --data_path ./data \
    --images_path ./data \
    --model_name Salesforce/blip-image-captioning-base \
    --language en \
    --batch_size 8 \
    --epochs 10 \
    --learning_rate 5e-5 \
    --output_dir ./outputs \
    --use_wandb
```

### Evaluation

Evaluate a trained model:

```python
from evaluation import evaluate_model, print_evaluation_results
from vqa_model import WoundWiseVQAModel

# Load model
model = WoundWiseVQAModel(language="en")
model.load_state_dict(torch.load("path/to/model.pt"))

# Evaluate
results = evaluate_model(model, test_loader, device, language="en")
print_evaluation_results(results)
```

### Inference

Generate responses for new wound images and queries:

```python
from PIL import Image
from vqa_model import WoundWiseVQAModel

# Load model
model = WoundWiseVQAModel(language="en")
model.load_state_dict(torch.load("path/to/model.pt"))

# Load image and prepare query
image = Image.open("wound_image.jpg")
query = "What type of wound is this and how should it be treated?"

# Generate response
response = model.generate_response(image, query)
print(f"AI Response: {response}")
```

## Model Architecture

The system supports multiple model architectures:

1. **BLIP-based Models**: Pre-trained vision-language models from Salesforce
2. **Custom Multimodal Encoder**: Combines ResNet vision encoder with BERT text encoder
3. **Cross-attention Mechanisms**: For better visual-textual understanding

### Key Features:
- Multi-language support (English and Chinese)
- Medical metadata integration
- Flexible model architectures
- Comprehensive evaluation metrics

## Evaluation Metrics

The system uses multiple metrics for comprehensive evaluation:

- **BLEU**: Measures n-gram overlap with reference responses
- **ROUGE**: Evaluates recall-oriented understanding
- **METEOR**: Considers synonyms and paraphrases
- **chrF**: Character-level F-score
- **Length Statistics**: Analyzes response length patterns

### Per-Category Analysis
Metrics are computed overall and broken down by:
- Wound type
- Anatomic location
- Infection status
- Other medical metadata

## File Structure

```
woundwise/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data_loader.py           # Dataset loading and processing
├── vqa_model.py            # VQA model implementations
├── train.py                # Training script
├── evaluation.py           # Evaluation metrics and utilities
└── outputs/                # Model outputs and results
```

## Supported Languages

- **English (en)**: Uses `tokenizer_13a` for BLEU evaluation
- **Chinese (zh)**: Uses `tokenizer_zh` for BLEU evaluation

## Medical Applications

This system is designed to assist healthcare providers with:

1. **Remote Wound Assessment**: Initial evaluation of wound images
2. **Treatment Recommendations**: Evidence-based treatment suggestions
3. **Patient Education**: Clear explanations of wound conditions
4. **Triage Support**: Identifying cases requiring urgent attention

## Contributing

Contributions are welcome! Please ensure all medical recommendations are reviewed by qualified healthcare professionals.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{woundwise2024,
  title={WoundWise: Visual Question Answering for Wound Care},
  author={WoundWise Team},
  year={2024},
  note={Extension of MEDIQA-M3G 2024 shared task}
}
```

## Disclaimer

This system is designed to assist healthcare providers and should not replace professional medical judgment. All AI-generated responses should be reviewed by qualified medical professionals before being provided to patients.