import pandas as pd
from datasets import load_dataset
from typing import Dict, Any

class DataLoader:
    """Handles loading datasets from various sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = config.get('dataset', {}).get('name', 'tatsu-lab/alpaca')
        
    def load(self) -> pd.DataFrame:
        """
        Load dataset from Hugging Face and convert to DataFrame
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print(f"Loading dataset: {self.dataset_name}")
        
        try:
            # Load from Hugging Face
            dataset = load_dataset(self.dataset_name)
            
            # Convert to pandas DataFrame
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                # If no train split, use the first available split
                first_split = list(dataset.keys())[0]
                df = dataset[first_split].to_pandas()
            
            print(f"Loaded {len(df)} examples")
            
            # Validate required columns
            self._validate_dataframe(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has required columns"""
        required_columns = ['instruction', 'output']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print("Dataset validation passed")
    
    def get_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the loaded dataset"""
        return {
            'total_rows': len(df),
            'columns': list(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'null_counts': df.isnull().sum().to_dict()
        }