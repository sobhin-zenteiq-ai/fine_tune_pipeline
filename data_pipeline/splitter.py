import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

class DataSplitter:
    """Handles splitting dataset into train/validation sets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        splitting_config = config.get('splitting', {})
        self.train_ratio = splitting_config.get('train_ratio', 0.9)
        self.validation_ratio = splitting_config.get('validation_ratio', 0.1)
        self.random_seed = splitting_config.get('random_seed', 42)
        
    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split dataset into train and validation sets
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict containing 'train' and 'validation' DataFrames
        """
        print("Starting dataset splitting...")
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Shuffle the dataset
        df_shuffled = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Calculate split indices
        total_size = len(df_shuffled)
        train_size = int(total_size * self.train_ratio)
        
        # Split the data
        train_df = df_shuffled[:train_size].copy()
        validation_df = df_shuffled[train_size:].copy()
        
        print(f"Train set: {len(train_df)} examples")
        print(f"Validation set: {len(validation_df)} examples")
        
        return {
            'train': train_df,
            'validation': validation_df
        }
    
    def get_split_stats(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get statistics about the dataset splits"""
        stats = {}
        
        for split_name, split_df in splits.items():
            if 'text' in split_df.columns:
                text_lengths = split_df['text'].str.len()
                stats[f'{split_name}_size'] = len(split_df)
                stats[f'{split_name}_avg_length'] = text_lengths.mean()
                stats[f'{split_name}_min_length'] = text_lengths.min()
                stats[f'{split_name}_max_length'] = text_lengths.max()
        
        # Overall stats
        total_size = sum(len(split_df) for split_df in splits.values())
        stats['total_size'] = total_size
        stats['actual_train_ratio'] = len(splits['train']) / total_size if 'train' in splits else 0
        stats['actual_validation_ratio'] = len(splits['validation']) / total_size if 'validation' in splits else 0
        
        return stats