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
        # Map task to required columns
        self.map = {
            'qa': ["instruction", "input", "output"],
            'lm': ["text"],
            'summarization': ["articles", "summaries"],
            'classification': ["text", "label"],
            'translation': ["source", "target"],
            }
        self.req_columns = set(self.map.get(self.task, []))
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply text preprocessing to the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        print("Starting text preprocessing...")
        
        # Process each text column
        text_columns = self.req_columns
        
        for column in text_columns:
            if column in df.columns:
                print(f"Processing column: {column}")
                df[column] = df[column].apply(self._preprocess_text)
        
        print("Text preprocessing complete")
        return df
    
    def _preprocess_text(self, text: str) -> str:
        """Apply preprocessing steps to a single text"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)
        
        # Expand contractions
        if self.expand_contractions:
            text = self._expand_contractions(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text"""
        for contraction, expansion in self.contractions.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace (multiple spaces/tabs/newlines to single space)"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def get_preprocessing_stats(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about preprocessing"""
        stats = {}
        
        text_columns = processed_df.columns
        for column in text_columns:
            if column in original_df.columns and column in processed_df.columns:
                orig_avg_len = original_df[column].fillna('').str.len().mean()
                proc_avg_len = processed_df[column].fillna('').str.len().mean()
                
                stats[f'{column}_avg_length_before'] = orig_avg_len
                stats[f'{column}_avg_length_after'] = proc_avg_len
                stats[f'{column}_length_change'] = proc_avg_len - orig_avg_len
        
        return stats