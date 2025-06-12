import pandas as pd
from typing import Dict, Any

class DataFormatter:
    """Handles formatting data for training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_type = config.get('model', {}).get('task', 'qa')
        
    def format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the dataset for training by combining instruction, input, and output
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Formatted DataFrame with 'text' column
        """
        print("Starting data formatting...")
        
        task = self.task_type.lower()
        
        if task == 'qa': df['tokenizer_input'] = df.apply(self._create_training_text_for_QA, axis=1)
        elif task == 'summarization': df['tokenizer_input'] = df.apply(self._create_training_text_for_summarization, axis=1)
        elif task == 'translation': df['tokenizer_input'] = df.apply(self._create_training_text_for_translation, axis=1)
        elif task == 'classification': df['tokenizer_input'] = df['text'].astype(str) 
        elif task == 'lm': df['tokenizer_input'] = df['text'].astype(str) 
        else:
            raise ValueError(f"Task not supported for formatting: {task}")
        # Keep original columns and add formatted text
        formatted_df = df.copy()
        print(f"Formatted {len(formatted_df)} examples")
        return formatted_df
    
    def _create_training_text_for_QA(self, row: pd.Series) -> str:
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
    
    def _create_training_text_for_summarization(self, row: pd.Series) -> str:
        """
        Create training text from input, and output
        
        Format: ### Summarize:\n{Article}\n\n### Summary:\n{input}
        """

        Article = row['article']
        summary = row.get('summary', '')
        
        # Build the training text
        text_parts = []
        
        # Add instruction
        text_parts.append(f"### Summarize the article:\n{Article}")
        
        # Add input if it exists and is not empty
        if summary and str(summary).strip():
            text_parts.append(f"### summary:\n{summary}")
        
        # Join with double newlines
        training_text = "\n\n".join(text_parts)
        
        return training_text

    def _create_training_text_for_translation(self, row: pd.Series) -> str:
        """
        Create training text from input, and output
        
        Format: ### Summarize:\n{Article}\n\n### Summary:\n{input}
        """

        source = row['source']
        target = row.get('target', '')
        
        # Build the training text
        text_parts = []
        
        # Add instruction
        text_parts.append(f"### Translate this sentence:\n{source}")
        
        # Add input if it exists and is not empty
        text_parts.append(f"### into:\n{target}")
        
        # Join with double newlines
        training_text = "\n\n".join(text_parts)
        
        return training_text
     
    
    def _create_training_text_for_classification(self, row: pd.Series) -> str:
        """
        Create training text from input, and output
        
        Format: ### Summarize:\n{Article}\n\n### Summary:\n{input}
        """

        text = row['text']
        label = row.get('label', '')
        
        # Build the training text
        text_parts = []
        
        # Add instruction
        text_parts.append(f"### Classify this sentence:\n{text}")
        
        # Add input if it exists and is not empty
        text_parts.append(f"### Label:\n{label}")
        
        # Join with double newlines
        training_text = "\n\n".join(text_parts)
        
        return training_text  
  
    
    def get_format_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the formatted data"""
        if 'tokenizer_input' not in df.columns:
            return {}
        
        text_lengths = df['tokenizer_input'].str.len()
        
        return {
            'total_examples': len(df),
            'avg_text_length': text_lengths.mean(),
            'min_text_length': text_lengths.min(),
            'max_text_length': text_lengths.max(),
            'median_text_length': text_lengths.median()
        }