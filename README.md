# Fine-Tuning Data Pipeline

A simple, working data pipeline for preparing datasets for fine-tuning language models.

## Features

- Load datasets from Hugging Face (supports Alpaca format)
- Clean data (remove duplicates, missing values, length filtering)
- Text preprocessing (lowercasing, URL removal, contraction expansion)
- Format data for training (instruction-input-output format)
- Split into train/validation sets
- Tokenize with any Hugging Face tokenizer
- Save processed data in multiple formats (JSON, CSV, PyTorch tensors)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd fine_tuning_pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package (optional)(Don't try not done yet):
```bash
pip install -e .
```

## Usage

### Quick Start

Run the full pipeline with default settings:
```bash
python main.py
```

### Custom Configuration

1. Edit `config.yaml` to customize the pipeline settings
2. Run with custom config:
```bash
python main.py --config your_config.yaml
```

### Step-by-Step Execution

Run individual pipeline steps:
```bash
python main.py --step load
python main.py --step clean
python main.py --step preprocess
# ... etc
```

### Programmatic Usage

```python
from data_pipeline import DataPipeline

# Simple usage
pipeline = DataPipeline()
results = pipeline.run()

# With custom config
config = {
    'dataset': {'name': 'tatsu-lab/alpaca'},
    'model': {'name': 'gpt2'},
    # ... other settings
}
pipeline = DataPipeline(config)
results = pipeline.run()

# Step by step
pipeline = DataPipeline()
raw_data = pipeline.run_step('load')
cleaned_data = pipeline.run_step('clean', raw_data)
# ... continue
```

## Configuration

The pipeline is configured via `config.yaml`. Key settings:

- `dataset.name`: Hugging Face dataset name
- `model.name`: Model name for tokenizer
- `cleaning`: Data cleaning parameters
- `preprocessing`: Text preprocessing options
- `splitting`: Train/validation split ratios
- `tokenization`: Tokenizer settings
- `output`: Output file settings

## Output Files

The pipeline generates several output files in the `output/` directory:

- `train_cleaned.json/csv`: Cleaned training data
- `validation_cleaned.json/csv`: Cleaned validation data
- `train_tokenized.pt`: Tokenized training data (PyTorch tensors)
- `validation_tokenized.pt`: Tokenized validation data
- `pipeline_stats.json`: Processing statistics
- `pipeline_report.txt`: Human-readable summary

## Pipeline Steps

1. **Data Loading**: Load dataset from Hugging Face
2. **Data Cleaning**: Remove missing values, duplicates, filter by length
3. **Text Preprocessing**: Clean and normalize text
4. **Data Formatting**: Create training format (instruction + input + output)
5. **Dataset Splitting**: Split into train/validation sets
6. **Tokenization**: Convert text to tokens for model training
7. **Data Saving**: Save processed data in multiple formats

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- Pandas
- NumPy
- PyYAML

## Project Structure

```
fine_tuning_pipeline/
├── README.md
├── requirements.txt
├── setup.py
├── config.yaml
├── main.py
├── data_pipeline/
│   ├── __init__.py
│   ├── pipeline.py          # Main orchestrator
│   ├── loader.py            # Data loading
│   ├── cleaner.py           # Data cleaning
│   ├── preprocessor.py      # Text preprocessing
│   ├── formatter.py         # Data formatting
│   ├── splitter.py          # Dataset splitting
│   ├── tokenizer_handler.py # Tokenization
│   └── saver.py            # Data saving
└── output/                  # Generated files
```

## Example Output

```
Loading dataset: tatsu-lab/alpaca
Loaded 52002 examples
Dataset validation passed
Starting data cleaning...
After removing missing values: 52002 rows
After removing duplicates: 51760 rows
After length filtering: 51658 rows
Cleaning complete. Removed 344 rows
Starting text preprocessing...
Processing column: instruction
Processing column: input
Processing column: output
Text preprocessing complete
Starting data formatting...
Formatted 51658 examples
Starting dataset splitting...
Train set: 46492 examples
Validation set: 5166 examples
Starting tokenization...
Loading tokenizer: gpt2
Tokenizing train set...
Tokenized 46492 examples for train
Tokenizing validation set...
Tokenized 5166 examples for validation
Starting data saving...
Saved train dataset: 46492 examples
Saved validation dataset: 5166 examples
Saved train tokenized data: 46492 examples
Saved validation tokenized data: 5166 examples
Saved pipeline statistics
Created summary report
Saved 9 files to output
Pipeline completed successfully!
```

## License
