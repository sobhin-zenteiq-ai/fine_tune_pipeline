import pandas as pd
from typing import Dict, Any
from datasets import load_dataset,get_dataset_config_names

class DataLoader:
    """Handles loading datasets from various sources"""

    def __init__(self, config: Dict[str, Any], task: str = "qa"):
        """
        Initializes the DataLoader with the configuration and task type.
        :param config: The configuration dictionary.
        :param task: The task to be handled ('qa' or 'summarization').
        """
        self.config = config
        self.task = self.config.get('model', {}).get('task','qa')  # 'qa' or 'summarization'
        self.available_datasets = self.config.get('tasks', {}).get(self.task, {}).get('available_datasets', [])
        self.instruction_column = self.config.get('dataset', {}).get('instruction', 'None')
        self.input_column = self.config.get('dataset', {}).get('input', 'None')
        self.output_column = self.config.get('dataset', {}).get('output', 'None')

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
        
        return self._load_from_huggingface(dataset_name)
    
    
    def _load_from_huggingface(self, dataset_name: str) -> pd.DataFrame:
        """Load dataset from Hugging Face"""
        print(f"Loading dataset: {dataset_name} from Hugging Face")
        try:
            dataset = load_dataset(dataset_name)
            print(f"Dataset {dataset_name} loaded successfully")
            print(dataset)
        except ValueError as e:
            # If config name is missing, list all configs and pick one
            if "Config name is missing" in str(e):
                configs = get_dataset_config_names(dataset_name)
                # Strategy: pick the latest or default one (e.g., highest version)
                selected_config = sorted(configs)[-1]  # or logic to choose default
                dataset =  load_dataset(dataset_name, selected_config)
            else:
                raise e

            # Convert to pandas DataFrame
        if 'train' in dataset:
            df = dataset['train'].to_pandas()
            print("hey")
            #print(df[0]['instruction'])
            print(f"Loaded {len(df)} training examples")
        else:
            first_split = list(dataset.keys())[0]
            df = dataset[first_split].to_pandas()

        print(f"Loaded {len(df)} examples")
        
        # Basic validation to ensure required columns exist
        self.validate_dataframe(df)

        return self.format_dataframe(df)

    def format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format DataFrame to match expected structure for the task"""
        
        config = self.config.get('dataset', {})
        instruction_column = config.get('instruction', '')
        input_column = config.get('input', 'input')
        output_column = config.get('output', 'output')
        df['input'] = df[input_column]
        df['output'] = df[output_column]
        if instruction_column != '':
            df['instruction'] = df[instruction_column]
            df = df[['instruction','input', 'output']].copy()

        else:
            df = df[['input', 'output']].copy()

        if self.task == 'classification':
            label2id = {label: idx for idx, label in enumerate(df['output'].unique())}
            df['label_id'] = df['output'].map(label2id)
        return df
    
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate if DataFrame has required columns"""
        config = self.config.get('dataset', {})
        instruction_column = config.get('instruction', '')
        input_column = config.get('input', 'input')
        output_column = config.get('output', 'output')

        if instruction_column == '' and self.task != 'qa': 
            required_columns = [input_column, output_column]  # Basic check for summarization
        else:
            required_columns = [instruction_column,input_column, output_column]  # Basic check for QA
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