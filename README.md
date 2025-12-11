# DeepSeek OCR Fine-Tuning and Evaluation Project

## Project Overview
This project focuses on fine-tuning a DeepSeek optical character recognition (OCR) model on Vietnamese text data and evaluating its performance. The work involves preprocessing raw OCR output, fine-tuning models with different dataset sizes, and comparing performance metrics across multiple model variants.

You could file my code here: [Notebook - Kaggle](https://www.kaggle.com/code/khngxuninh/deepseek-ocr-finetuning?scriptVersionId=285152875)

## Project Structure

```
Lab02 NLP/
├── README.md                           # This file
├── 23127398_report.pdf                 # Research report
├── raw_output/                         # Raw OCR outputs and logs
│   ├── base_output_log_100k.txt       # Base model output (100k training samples)
│   ├── base_output_log_80k_21.txt     # Base model output (80k training samples)
│   ├── final_output_log_100k.txt      # Fine-tuned model output (100k training samples)
│   ├── final_output_log_80k.txt       # Fine-tuned model output (80k training samples)
│   └── finetuned_output_log_21.txt    # Additional fine-tuned model output
├── cleaned_output/                     # Cleaned and processed prediction data
│   ├── base_model_prediction_100k.csv # Base model predictions (100k)
│   ├── base_model_prediction_80k_21.csv # Base model predictions (80k)
│   ├── final_model_prediction_100k.csv # Fine-tuned model predictions (100k)
│   ├── final_model_prediction_80k.csv # Fine-tuned model predictions (80k)
│   └── finetuned_model_prediction_80k_21.csv # Additional codes for processing
├── preprocess_output/                  # Preprocessing notebooks and logs
│   ├── clean_output.ipynb             # Main preprocessing script
│   └── clean_final_output_log.ipynb   # Final preprocessing log
├── Finetune_code/                      # Fine-tuning implementation
│   ├── deepseek-ocr-finetuning_80k.ipynb  # Fine-tuning with 80k samples
│   └── deepseek-ocr-finetuning_100k.ipynb # Fine-tuning with 100k samples
└── evaluate_code/                      # Evaluation scripts
    └── evaluate.ipynb                  # Model evaluation and comparison
```

## Dataset Information

Data source: [UIT_HWDB Dataset](https://github.com/nghiangh/UIT-HWDB-dataset)

### Data Distribution
The project evaluates three model variants:
1. **Base Model**: Pre-trained DeepSeek OCR model without fine-tuning
2. **Fine-tuned Model (80k)**: Model trained on 80,000 Vietnamese text samples
3. **Fine-tuned Model (100k)**: Model trained on 100,000 Vietnamese text samples

## Workflow

### 1. Raw Output Generation
Raw OCR outputs are generated from the model inference on test images. These outputs are stored in `raw_output/` with separate logs for:
- Base model predictions
- Fine-tuned model predictions (80k and 100k variants)

**File formats**: Text logs containing predicted OCR text and ground truth labels

### 2. Data Preprocessing
The `preprocess_output/` notebooks clean and process raw model outputs:
- Extract predicted text (y_pred) and ground truth text (y_true)
- Handle formatting inconsistencies
- Convert to CSV format for evaluation
- Remove invalid or malformed entries

**Key scripts**:
- `clean_output.ipynb`: Main preprocessing pipeline
- `clean_final_output_log.ipynb`: Final output cleaning and validation

**Output**: CSV files with two columns (y_pred, y_true) for each model variant

### 3. Model Fine-Tuning
The `Finetune_code/` notebooks implement the fine-tuning process using different dataset sizes:

**deepseek-ocr-finetuning_80k.ipynb**:
- Loads DeepSeek model
- Fine-tunes on 80,000 image samples
- Implements custom training loop
- Saves intermediate checkpoints
- Generates predictions on test set

**deepseek-ocr-finetuning_100k.ipynb**:
- Similar workflow as 80k variant
- Uses 100,000 image samples for training
- Larger dataset for improved model performance

**Training outputs**: Raw prediction logs and model checkpoints

### 4. Evaluation
The `evaluate_code/evaluate.ipynb` notebook computes comprehensive performance metrics:

**Metrics Calculated**:
- **CER (Character Error Rate)**: Percentage of character-level errors
- **WER (Word Error Rate)**: Percentage of word-level errors
- **Exact Match (EM)**: Percentage of perfectly predicted samples

**Evaluation Approach**:
- Global metrics across all test samples
- Per-sample metrics for detailed analysis
- Comparison tables across model variants
- Visualization of performance differences

**Comparison Structure**:
- Base model vs. 80k fine-tuned model
- Base model vs. 100k fine-tuned model
- 80k model vs. 100k model

## Key Files and Their Purpose

### Input Data
- `raw_output/*.txt`: Raw model predictions and ground truth from inference
- Training datasets: 80k and 100k Vietnamese OCR images (referenced in finetune notebooks)

### Intermediate Data
- `cleaned_output/*.csv`: Processed predictions ready for evaluation
  - Format: `y_pred, y_true` columns
  - ~3,100 test samples for 100k models
  - ~23 test samples for 80k model variant

### Processing Scripts
- **Preprocessing**: Notebooks in `preprocess_output/` clean raw outputs
- **Fine-tuning**: Notebooks in `Finetune_code/` implement model training
- **Evaluation**: `evaluate_code/evaluate.ipynb` computes all metrics

### Output
- Performance metrics (CER, WER, EM)
- Comparison visualizations (charts and tables)
- Detailed error analysis per sample

## Model Details

### Architecture
- **Base Model**: DeepSeek pre-trained OCR model
- **Fine-tuning Method**: Supervised learning on Vietnamese OCR dataset
- **Input Format**: Images of Vietnamese text documents
- **Output Format**: Predicted text strings

### Performance Expectations
- Base model shows lower accuracy (baseline)
- Fine-tuning with 80k samples improves performance
- Fine-tuning with 100k samples provides the best performance
- CER and WER are typically lower for fine-tuned models
- Exact match rate increases with fine-tuning

## Technology Stack
- **Deep Learning**: PyTorch (used in fine-tuning notebooks)
- **Data Processing**: Pandas, NumPy
- **Evaluation Metrics**: jiwer (for CER/WER calculation)
- **Visualization**: Matplotlib
- **Development Environment**: Jupyter Notebooks

## How to Use

### To Preprocess Data
1. Generate raw outputs from model inference
2. Open `preprocess_output/clean_output.ipynb`
3. Run all cells to clean and extract predictions
4. Output saved to `cleaned_output/*.csv`

### To Fine-Tune a Model
1. Prepare training dataset (80k or 100k images)
2. Open corresponding notebook in `Finetune_code/`
3. Update data paths as needed
4. Run training cells
5. Save model checkpoint
6. Run inference on test set to generate predictions

### To Evaluate Models
1. Ensure cleaned CSV files are in `cleaned_output/`
2. Open `evaluate_code/evaluate.ipynb`
3. Run evaluation cells to compute metrics
4. Review comparison tables and visualizations
5. Analyze per-sample errors if needed

## Key Findings

The evaluation framework enables direct comparison of:
- **Pre-trained vs. Fine-tuned**: Impact of domain-specific fine-tuning on Vietnamese OCR
- **Dataset Size Impact**: Effect of training on 80k vs. 100k samples
- **Error Analysis**: Character-level vs. word-level error patterns

Fine-tuning consistently improves OCR accuracy, with larger training sets (100k) generally outperforming smaller ones (80k) in terms of CER, WER, and exact match rates.

## Notes
- Ground truth data comes from manually annotated Vietnamese text documents
- Base model output serves as the performance baseline
- All evaluation metrics are calculated using the jiwer library
- Results are specific to Vietnamese text OCR task
- Dataset preprocessing is critical for fair model comparison

## Author
Dinh Xuan Khuong - FITUS

---

Last Updated: December 2025
