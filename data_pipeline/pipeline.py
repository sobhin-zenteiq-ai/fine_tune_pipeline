import yaml
import pandas as pd
from typing import Dict, Any, Optional
from .loader import DataLoader
from .cleaner import DataCleaner
from .preprocessor import TextPreprocessor
from .formatter import DataFormatter
from .splitter import DataSplitter
from .tokenizer_handler import TokenizerHandler
from .validator import DatasetValidator
from .saver import DataSaver

class DataPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None, task: Optional[str] = "qa", dataset_name: Optional[str] = None):
        """
        Initialize the data pipeline
        
        Args:
            config: Configuration dictionary
            config_path: Path to configuration YAML file
            task: Task to run ('qa' or 'summarization')
            dataset_name: Name of the dataset to use
        """
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            # Default configuration
            self.config = self._get_default_config()

        self.task = task  # Store the task (either 'qa' or 'summarization')
        self.dataset_name = dataset_name  # Store the dataset name

        # Initialize all components with the task information
        self.loader = DataLoader(self.config, task=self.task)
        self.validator = DatasetValidator(self.config, task=self.task)
        self.cleaner = DataCleaner(self.config)
        self.preprocessor = TextPreprocessor(self.config)
        self.formatter = DataFormatter(self.config)
        self.splitter = DataSplitter(self.config)
        self.tokenizer_handler = TokenizerHandler(self.config)
        self.saver = DataSaver(self.config)

        # Store processing statistics
        self.stats = {}
        
    def run(self, dataset_name: Optional[str] = None,df: Optional[pd.DataFrame]=None) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Args:
            dataset_name: Name of the dataset to use (overrides the one set in __init__)
        
        Returns:
            Dictionary containing results and file paths
        """
        print("Starting Data Pipeline...")
        print("=" * 50)
        
        # Use provided dataset_name or fall back to the one set in __init__
        dataset_to_use = dataset_name or self.dataset_name
        
        if not dataset_to_use:
            raise ValueError("Dataset name must be provided either in __init__ or run() method")
        
        try:
            # Step 1: Load data
            print(f"Step 1: Loading dataset '{dataset_to_use}' for task '{self.task}'...")
            if df is not None:
                # If a DataFrame is provided, use it directly
                raw_data = df
                loader_stats = self.loader.get_info(raw_data)
                self.stats.update(loader_stats)
            else: 
                raw_data = self.loader.load(dataset_name=dataset_to_use)
                loader_stats = self.loader.get_info(raw_data)
                self.stats.update(loader_stats)

            # Step 1.5: Validate dataset
            print("Step 1.5: Validating dataset...")
            validation_result = self.validator.validate_dataset(dataset_features=set(loader_stats['columns']))
            self.stats['validation_result'] = validation_result['is_valid']
            self.stats['validation_warnings'] = validation_result.get('warnings', [])

            print(f"Dataset validation result: {validation_result['is_valid']}")
            if not validation_result['is_valid']:
                raise ValueError(f"Dataset schema validation failed: {validation_result['error_message']}")

            # Step 2: Clean data
            print("Step 2: Cleaning data...")
            original_data = raw_data.copy()
            cleaned_data = self.cleaner.clean(raw_data)
            cleaning_stats = self.cleaner.get_cleaning_stats(original_data, cleaned_data)
            self.stats.update(cleaning_stats)
            
            # Step 3: Preprocess text
            print("Step 3: Preprocessing text...")
            preprocessed_data = self.preprocessor.process(cleaned_data)
            preprocessing_stats = self.preprocessor.get_preprocessing_stats(cleaned_data, preprocessed_data)
            self.stats.update(preprocessing_stats)
            
            # Step 4: Format data
            print("Step 4: Formatting data...")
            formatted_data = self.formatter.format(preprocessed_data)
            format_stats = self.formatter.get_format_stats(formatted_data)
            self.stats.update(format_stats)
            
            # Step 5: Split data
            print("Step 5: Splitting data...")
            splits = self.splitter.split(formatted_data)
            split_stats = self.splitter.get_split_stats(splits)
            self.stats.update(split_stats)
            
            # Step 6: Tokenize data
            print("Step 6: Tokenizing data...")
            tokenized_splits = self.tokenizer_handler.tokenize(splits)
            tokenization_stats = self.tokenizer_handler.get_tokenization_stats(tokenized_splits)
            self.stats.update(tokenization_stats)
            
            # Step 7: Save results
            print("Step 7: Saving results...")
            saved_files = self.saver.save(splits, tokenized_splits, self.stats)
            summary_report = self.saver.create_summary_report(self.stats)
            saved_files['summary_report'] = summary_report
            
            print("=" * 50)
            print("Pipeline completed successfully!")
            print(f"Processed {self.stats.get('total_size', 'N/A')} examples")
            print(f"Files saved to: {self.saver.output_dir}")
            
            return {
                'success': True,
                'stats': self.stats,
                'files': saved_files,
                'splits': splits,
                'tokenized_splits': tokenized_splits
            }
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }
    
    def run_step(self, step_name: str, input_data=None):
        """
        Run a specific pipeline step
        
        Args:
            step_name: Name of the step to run
            input_data: Input data for the step
        """
        steps = {
            'load': lambda: self.loader.load(dataset_name=self.dataset_name),
            'validate': lambda: self.validator.validate_dataset(
                dataset_features=set(input_data.columns) if input_data is not None else set(),
                task_type=self.task
            ),
            'clean': lambda data: self.cleaner.clean(data),
            'preprocess': lambda data: self.preprocessor.process(data),
            'format': lambda data: self.formatter.format(data),
            'split': lambda data: self.splitter.split(data),
            'tokenize': lambda data: self.tokenizer_handler.tokenize(data),
        }
        
        if step_name not in steps:
            raise ValueError(f"Unknown step: {step_name}")
        
        if input_data is not None:
            return steps[step_name](input_data)
        else:
            return steps[step_name]()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from: {config_path}")
            return config
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            print("Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'dataset': {'name': 'tatsu-lab/alpaca'},
            'model': {'name': 'gpt2'},
            'tasks': {
                'qa': {
                    'available_datasets': ['squad', 'natural_questions', 'custom_qa_dataset']
                },
                'summarization': {
                    'available_datasets': ['xsum', 'cnn_dailymail', 'custom_summarization_dataset']
                }
            },
            'cleaning': {
                'min_length': 10,
                'max_length': 1000,
                'remove_duplicates': True
            },
            'preprocessing': {
                'lowercase': True,
                'remove_urls': True,
                'remove_extra_whitespace': True,
                'expand_contractions': True
            },
            'splitting': {
                'train_ratio': 0.9,
                'validation_ratio': 0.1,
                'random_seed': 42
            },
            'tokenization': {
                'max_length': 512,
                'padding': True,
                'truncation': True
            },
            'output': {
                'save_cleaned': True,
                'save_tokenized': True,
                'save_stats': True,
                'output_dir': 'output'
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics"""
        return self.stats.copy()
    
    def print_summary(self):
        """Print a summary of the pipeline results"""
        if not self.stats:
            print("No statistics available. Run the pipeline first.")
            return
        
        print("\nPipeline Summary:")
        print("-" * 30)
        print(f"Total examples: {self.stats.get('total_examples', 'N/A')}")
        print(f"Training examples: {self.stats.get('train_size', 'N/A')}")
        print(f"Validation examples: {self.stats.get('validation_size', 'N/A')}")
        print(f"Examples removed: {self.stats.get('removed_count', 'N/A')}")
        print(f"Average tokens (train): {self.stats.get('train_avg_tokens', 'N/A')}")
        print(f"Average tokens (validation): {self.stats.get('validation_avg_tokens', 'N/A')}")
        print(f"Output directory: {self.saver.output_dir}")
    
    def get_available_datasets(self) -> list:
        """Get list of available datasets for the current task"""
        return self.config.get('tasks', {}).get(self.task, {}).get('available_datasets', [])
    
    def set_dataset(self, dataset_name: str):
        """Set the dataset name for the pipeline"""
        if dataset_name not in self.get_available_datasets():
            raise ValueError(f"Dataset '{dataset_name}' is not available for task '{self.task}'. Available datasets: {self.get_available_datasets()}")
        self.dataset_name = dataset_name

    def load_custom_dataset(self, file_path: str) -> bool:
        """
        Load and validate a custom dataset
        
        Args:
            file_path: Path to the custom dataset file
            
        Returns:
            bool: True if dataset is valid and loaded successfully
        """
        try:
            import pandas as pd
            
            # Try to load the file (supports CSV for now)
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or JSON.")
            
            # Validate the dataset
            is_valid = self.loader.validate_custom_dataset(df, self.task)
            
            if is_valid:
                print(f"Custom dataset loaded successfully: {len(df)} examples")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error loading custom dataset: {str(e)}")
            return False