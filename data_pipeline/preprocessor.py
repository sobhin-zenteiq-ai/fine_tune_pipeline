import pandas as pd
import re
from typing import Dict, Any

class TextPreprocessor:
    """Handles text preprocessing operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        preprocessing_config = config.get('preprocessing', {})
        self.lowercase = preprocessing_config.get('lowercase', True)
        self.remove_urls = preprocessing_config.get('remove_urls', True)
        self.remove_extra_whitespace = preprocessing_config.get('remove_extra_whitespace', True)
        self.expand_contractions = preprocessing_config.get('expand_contractions', True)
        
        # Contraction mapping
        self.contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "it's": "it is",
            "that's": "that is", "what's": "what is", "there's": "there is",
            "here's": "here is", "where's": "where is", "how's": "how is",
            "let's": "let us", "who's": "who is", "don't": "do not",
            "doesn't": "does not", "didn't": "did not", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
            "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
            "couldn't": "could not", "mustn't": "must not"
        }
        self.task = config.get('model', {}).get('task', 'qa')
      
        if self.task == 'qa':
            self.req_columns = {"instruction", "output"}
        else:
            self.req_columns = {'input','output'}
        
        self.url_pattern = re.compile(r'https?://\S+')
        self.ws_pattern = re.compile(r'\s+')
        self.contractions_pattern = re.compile(
            "|".join(re.escape(k) for k in self.contractions), flags=re.IGNORECASE
        )
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply text preprocessing to the dataset

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        print("Starting text preprocessing...")

        for column in self.req_columns:
            if column in df.columns:
                print(f"Processing column: {column}")
                df[column] = df[column].astype(str)
                if self.remove_urls:
                    df[column] = df[column].str.replace(self.url_pattern, '', regex=True)
                if self.expand_contractions:
                    df[column] = df[column].str.replace(
                        self.contractions_pattern,
                        lambda m: self.contractions.get(m.group(0).lower(), m.group(0)),
                        regex=True
                    )
                if self.lowercase:
                    df[column] = df[column].str.lower()
                if self.remove_extra_whitespace:
                    df[column] = df[column].str.replace(self.ws_pattern, ' ', regex=True)
                df[column] = df[column].str.strip()

        print("Text preprocessing complete")
        return df
    
    
    def get_preprocessing_stats(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about preprocessing"""
        stats = {}
        
        text_columns = processed_df.columns
        if self.task == 'classification':
            text_columns = {'input', 'output'}
        for column in text_columns:
            if column in original_df.columns and column in processed_df.columns:
                orig_avg_len = original_df[column].fillna('').str.len().mean()
                proc_avg_len = processed_df[column].fillna('').str.len().mean()
                
                stats[f'{column}_avg_length_before'] = orig_avg_len
                stats[f'{column}_avg_length_after'] = proc_avg_len
                stats[f'{column}_length_change'] = proc_avg_len - orig_avg_len
        
        return stats