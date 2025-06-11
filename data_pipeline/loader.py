import pandas as pd
from typing import Dict, Any
from datasets import load_dataset

class DataLoader:
    """Handles loading datasets from various sources"""

    def __init__(self, config: Dict[str, Any], task: str = "qa"):
        """
        Initializes the DataLoader with the configuration and task type.
        :param config: The configuration dictionary.
        :param task: The task to be handled ('qa' or 'summarization').
        """
        self.config = config
        self.task = task  # 'qa' or 'summarization'
        self.available_datasets = self.config.get('tasks', {}).get(self.task, {}).get('available_datasets', [])
        
    def load(self, dataset_name: str = None) -> pd.DataFrame:
        """
        Load dataset from Hugging Face or from custom upload based on task.
        
        Args:
            dataset_name: The name of the dataset to load (for Hugging Face or custom upload).
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print(f"Loading dataset for task: {self.task}")
        
        if dataset_name is None:
            raise ValueError("Dataset name must be provided.")
        
        if dataset_name in self.available_datasets:
            # If the dataset is available in the config for this task
            if dataset_name == 'squad':
                return self._load_squad_dataset(dataset_name)
            elif dataset_name in ['natural_questions']:
                return self._load_from_huggingface(dataset_name)
            elif dataset_name in ['xsum', 'cnn_dailymail']:
                return self._load_summarization_dataset(dataset_name)
            else:
                # Load dataset from Hugging Face for other tasks
                return self._load_from_huggingface(dataset_name)
        else:
            raise ValueError(f"Dataset {dataset_name} is not available for the {self.task} task.")
    
    def _load_squad_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load the SQuAD dataset from Hugging Face and map columns"""
        print(f"Loading dataset: {dataset_name} from Hugging Face")
        try:
            dataset = load_dataset(dataset_name)

            # Convert to pandas DataFrame
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                first_split = list(dataset.keys())[0]
                df = dataset[first_split].to_pandas()

            print(f"Loaded {len(df)} examples")

            # Map SQuAD columns to QA task columns
            df['instruction'] = df['question']  # Mapping 'question' to 'instruction'
            df['input'] = df['context']  # Mapping 'context' to 'input'
            
            # Extract answers and join them if multiple answers exist
            df['output'] = df['answers'].apply(lambda x: ' '.join(x['text']) if isinstance(x, dict) and 'text' in x else str(x))

            # Keep only the required columns and drop others to avoid serialization issues
            df = df[['instruction', 'input', 'output']].copy()

            self._validate_dataframe(df)

            return df
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def _load_summarization_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load summarization datasets and map columns appropriately"""
        print(f"Loading dataset: {dataset_name} from Hugging Face")
        try:
            dataset = load_dataset(dataset_name)

            # Convert to pandas DataFrame
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                first_split = list(dataset.keys())[0]
                df = dataset[first_split].to_pandas()

            print(f"Loaded {len(df)} examples")

            # Map columns based on dataset
            if dataset_name == 'xsum':
                df['instruction'] = "Summarize the following document:"
                df['input'] = df['document']
                df['output'] = df['summary']
            elif dataset_name == 'cnn_dailymail':
                df['instruction'] = "Summarize the following article:"
                df['input'] = df['article']
                df['output'] = df['highlights']

            # Keep only the required columns
            df = df[['instruction', 'input', 'output']].copy()

            self._validate_dataframe(df)
            return df

        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def _load_from_huggingface(self, dataset_name: str) -> pd.DataFrame:
        """Load dataset from Hugging Face"""
        print(f"Loading dataset: {dataset_name} from Hugging Face")
        try:
            dataset = load_dataset(dataset_name)

            # Convert to pandas DataFrame
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                first_split = list(dataset.keys())[0]
                df = dataset[first_split].to_pandas()

            print(f"Loaded {len(df)} examples")
            
            # Basic validation to ensure required columns exist
            self._validate_dataframe(df)

            return df

        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate if DataFrame has required columns"""
        required_columns = ['instruction', 'input', 'output']  # Basic check for QA
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print("Dataset validation passed")
    
    def get_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get information about the loaded dataset"""
        info = {
            'total_examples': len(df),
            'columns': list(df.columns),
            'memory_usage': int(df.memory_usage(deep=True).sum()),
            'null_counts': {}
        }
        
        # Convert dtypes and null counts to JSON-serializable format
        for col in df.columns:
            info['null_counts'][col] = int(df[col].isnull().sum())
        
        return info
    
    def get_available_datasets(self) -> list:
        """Get list of available datasets for the current task"""
        return self.available_datasets
    
    def validate_custom_dataset(self, df: pd.DataFrame, task: str) -> bool:
        """Validate if a custom dataset is suitable for the given task"""
        if task == 'qa':
            required_columns = ['instruction', 'input', 'output']
        elif task == 'summarization':
            required_columns = ['instruction', 'input', 'output']
        else:
            raise ValueError(f"Unknown task: {task}")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Custom dataset validation failed. Missing columns: {missing_columns}")
            return False
        
        print("Custom dataset validation passed")
        return True