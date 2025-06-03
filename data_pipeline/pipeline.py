import yaml
from typing import Dict, Any, Optional
from .loader import DataLoader
from .cleaner import DataCleaner
from .preprocessor import TextPreprocessor
from .formatter import DataFormatter
from .splitter import DataSplitter
from .tokenizer_handler import TokenizerHandler
from .saver import DataSaver

class DataPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        Initialize the data pipeline
        
        Args:
            config: Configuration dictionary
            config_path: Path to configuration YAML file
        """
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            # Default configuration
            self.config = self._get_default_config()
        
        # Initialize all components
        self.loader = DataLoader(self.config)
        self.cleaner = DataCleaner(self.config)
        self.preprocessor = TextPreprocessor(self.config)
        self.formatter = DataFormatter(self.config)
        self.splitter = DataSplitter(self.config)
        self.tokenizer_handler = TokenizerHandler(self.config)
        self.saver = DataSaver(self.config)
        
        # Store processing statistics
        self.stats = {}
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Returns:
            Dictionary containing results and file paths
        """
        print("Starting Data Pipeline...")
        print("=" * 50)
        
        try:
            # Step 1: Load data
            raw_data = self.loader.load()
            loader_stats = self.loader.get_info(raw_data)
            self.stats.update(loader_stats)
            
            # Step 2: Clean data
            original_data = raw_data.copy()
            cleaned_data = self.cleaner.clean(raw_data)
            cleaning_stats = self.cleaner.get_cleaning_stats(original_data, cleaned_data)
            self.stats.update(cleaning_stats)
            
            # Step 3: Preprocess text
            preprocessed_data = self.preprocessor.process(cleaned_data)
            preprocessing_stats = self.preprocessor.get_preprocessing_stats(cleaned_data, preprocessed_data)
            self.stats.update(preprocessing_stats)
            
            # Step 4: Format data
            formatted_data = self.formatter.format(preprocessed_data)
            format_stats = self.formatter.get_format_stats(formatted_data)
            self.stats.update(format_stats)
            
            # Step 5: Split data
            splits = self.splitter.split(formatted_data)
            split_stats = self.splitter.get_split_stats(splits)
            self.stats.update(split_stats)
            
            # Step 6: Tokenize data
            tokenized_splits = self.tokenizer_handler.tokenize(splits)
            tokenization_stats = self.tokenizer_handler.get_tokenization_stats(tokenized_splits)
            self.stats.update(tokenization_stats)
            
            # Step 7: Save results
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
            'load': lambda: self.loader.load(),
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
        print(f"Total examples: {self.stats.get('total_size', 'N/A')}")
        print(f"Training examples: {self.stats.get('train_size', 'N/A')}")
        print(f"Validation examples: {self.stats.get('validation_size', 'N/A')}")
        print(f"Examples removed: {self.stats.get('removed_count', 'N/A')}")
        print(f"Average tokens (train): {self.stats.get('train_avg_tokens', 'N/A')}")
        print(f"Average tokens (validation): {self.stats.get('validation_avg_tokens', 'N/A')}")
        print(f"Output directory: {self.saver.output_dir}")