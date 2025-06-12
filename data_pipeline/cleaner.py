import pandas as pd
from typing import Dict, Any

class DataCleaner:
    """Handles basic data cleaning operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        cleaning_config = config.get('cleaning', {})
        self.task = config.get('model',{}).get('task', 'qa')
        self.min_length = cleaning_config.get('min_length', 10)
        self.max_length = cleaning_config.get('max_length', 1000)
        self.remove_duplicates = cleaning_config.get('remove_duplicates', True)
        self.map = {
            'qa': ["instruction", "output"],
            'lm': ["text"],
            'summarization': ["articles", "summaries"],
            'classification': ["text", "label"],
            'translation': ["source", "target"],
            }
        self.req_columns = set(self.map.get(self.task, []))
        
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
        
        df = self._convert_columns_to_string(df)

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
    
    def _convert_columns_to_string(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all required columns to string type."""
        for col in self.req_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df

    
    def _remove_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing or empty values in critical columns"""
        # Check instruction and output columns
        df = df.dropna(subset=self.req_columns)
        
        # Remove empty strings
        for col in self.req_columns:
            if col in df.columns:
                df = df[df[col].str.strip() != '']
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate entries based on the specified set of columns."""
        # Convert set to sorted list for consistent column ordering
        col_list = sorted(self.req_columns)

        # Fill NaNs with empty string and create a composite key
        df['_composite_key'] = df[col_list].fillna('').astype(str).agg('|'.join, axis=1)

        # Drop duplicates based on the composite key
        df = df.drop_duplicates(subset=['_composite_key'])
        df = df.drop(columns=['_composite_key'])

        return df

    
    def _filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out entries whose combined length of specified columns is too short or too long."""
        # Convert set to sorted list for consistent column ordering
        col_list = sorted(self.req_columns)

        # Compute total length by summing string lengths of each specified column
        total_length = sum(
            df[col].fillna('').astype(str).str.len() for col in col_list
        )
        df['_total_length'] = total_length

        # Filter by length
        df = df[
            (df['_total_length'] >= self.min_length) &
            (df['_total_length'] <= self.max_length)
        ]

        # Drop helper column
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