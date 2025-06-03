import pandas as pd
from typing import Dict, Any

class DataFormatter:
    """Handles formatting data for training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the dataset for training by combining instruction, input, and output
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Formatted DataFrame with 'text' column
        """
        print("Starting data formatting...")
        
        # Create training text format
        df['text'] = df.apply(self._create_training_text, axis=1)
        
        # Keep original columns and add formatted text
        formatted_df = df.copy()
        
        print(f"Formatted {len(formatted_df)} examples")
        return formatted_df
    
    def _create_training_text(self, row: pd.Series) -> str:
        """
        Create training text from instruction, input, and output
        
        Format: ### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}
        """
        instruction = row['instruction']
        input_text = row.get('input', '')
        output = row['output']
        
        # Build the training text
        text_parts = []
        
        # Add instruction
        text_parts.append(f"### Instruction:\n{instruction}")
        
        # Add input if it exists and is not empty
        if input_text and str(input_text).strip():
            text_parts.append(f"### Input:\n{input_text}")
        
        # Add response
        text_parts.append(f"### Response:\n{output}")
        
        # Join with double newlines
        training_text = "\n\n".join(text_parts)
        
        return training_text
    
    def get_format_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the formatted data"""
        if 'text' not in df.columns:
            return {}
        
        text_lengths = df['text'].str.len()
        
        return {
            'total_examples': len(df),
            'avg_text_length': text_lengths.mean(),
            'min_text_length': text_lengths.min(),
            'max_text_length': text_lengths.max(),
            'median_text_length': text_lengths.median(),
            'examples_with_input': (df['input'].fillna('').str.strip() != '').sum()
        }