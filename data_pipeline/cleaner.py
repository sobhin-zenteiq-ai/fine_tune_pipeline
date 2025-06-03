import pandas as pd
from typing import Dict, Any

class DataCleaner:
    """Handles basic data cleaning operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        cleaning_config = config.get('cleaning', {})
        self.min_length = cleaning_config.get('min_length', 10)
        self.max_length = cleaning_config.get('max_length', 1000)
        self.remove_duplicates = cleaning_config.get('remove_duplicates', True)
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cleaning operations to the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        print("Starting data cleaning...")
        original_size = len(df)
        
        # Remove missing values
        df = self._remove_missing_values(df)
        print(f"After removing missing values: {len(df)} rows")
        
        # Remove duplicates
        if self.remove_duplicates:
            df = self._remove_duplicates(df)
            print(f"After removing duplicates: {len(df)} rows")
        
        # Filter by length
        df = self._filter_by_length(df)
        print(f"After length filtering: {len(df)} rows")
        
        print(f"Cleaning complete. Removed {original_size - len(df)} rows")
        return df.reset_index(drop=True)
    
    def _remove_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing or empty values in critical columns"""
        # Check instruction and output columns
        df = df.dropna(subset=['instruction', 'output'])
        
        # Remove empty strings
        df = df[df['instruction'].str.strip() != '']
        df = df[df['output'].str.strip() != '']
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate entries based on instruction and input"""
        # Create a composite key for duplicate detection
        if 'input' in df.columns:
            df['_composite_key'] = df['instruction'] + '|' + df['input'].fillna('')
        else:
            df['_composite_key'] = df['instruction']
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['_composite_key'])
        df = df.drop(columns=['_composite_key'])
        
        return df
    
    def _filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out entries that are too short or too long"""
        # Calculate total length (instruction + input + output)
        df['_total_length'] = (
            df['instruction'].str.len() + 
            df.get('input', pd.Series([''] * len(df))).fillna('').str.len() + 
            df['output'].str.len()
        )
        
        # Filter by length
        df = df[
            (df['_total_length'] >= self.min_length) & 
            (df['_total_length'] <= self.max_length)
        ]
        
        df = df.drop(columns=['_total_length'])
        return df
    
    def get_cleaning_stats(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the cleaning process"""
        return {
            'original_size': len(original_df),
            'cleaned_size': len(cleaned_df),
            'removed_count': len(original_df) - len(cleaned_df),
            'removal_percentage': (len(original_df) - len(cleaned_df)) / len(original_df) * 100
        }