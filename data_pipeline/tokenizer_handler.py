import pandas as pd
import torch
from transformers import AutoTokenizer
from typing import Dict, Any, List
from torch.utils.data import Dataset

class TokenizerHandler:
    """Handles tokenization of text data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_config = config.get('model', {})
        tokenization_config = config.get('tokenization', {})
        
        self.model_name = model_config.get('name', 'gpt2')
        self.max_length = tokenization_config.get('max_length', 512)
        self.padding = tokenization_config.get('padding', True)
        self.truncation = tokenization_config.get('truncation', True)
        
        # Initialize tokenizer
        self.tokenizer = self._load_tokenizer()
        
    def _load_tokenizer(self):
        """Load and configure tokenizer"""
        print(f"Loading tokenizer: {self.model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    
    def tokenize(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, 'TokenizedDataset']:
        """
        Tokenize the text data for each split
        
        Args:
            splits: Dictionary containing train/validation DataFrames
            
        Returns:
            Dictionary containing tokenized datasets
        """
        print("Starting tokenization...")
        
        tokenized_splits = {}
        
        for split_name, df in splits.items():
            print(f"Tokenizing {split_name} set...")
            tokenized_dataset = self._tokenize_dataframe(df)
            tokenized_splits[split_name] = tokenized_dataset
            print(f"Tokenized {len(tokenized_dataset)} examples for {split_name}")
        
        return tokenized_splits
    
    def _tokenize_dataframe(self, df: pd.DataFrame) -> 'TokenizedDataset':
        """Tokenize a single DataFrame"""
        texts = df['text'].tolist()
        
        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return TokenizedDataset(encodings, df)
    
    def get_tokenization_stats(self, tokenized_splits: Dict[str, 'TokenizedDataset']) -> Dict[str, Any]:
        """Get statistics about tokenization"""
        stats = {}
        
        for split_name, dataset in tokenized_splits.items():
            input_ids = dataset.encodings['input_ids']
            
            # Calculate lengths (non-padding tokens)
            if self.tokenizer.pad_token_id is not None:
                lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1)
            else:
                lengths = torch.tensor([len(ids) for ids in input_ids])
            
            stats[f'{split_name}_avg_tokens'] = lengths.float().mean().item()
            stats[f'{split_name}_min_tokens'] = lengths.min().item()
            stats[f'{split_name}_max_tokens'] = lengths.max().item()
            stats[f'{split_name}_examples'] = len(dataset)
        
        stats['tokenizer_vocab_size'] = self.tokenizer.vocab_size
        stats['max_length'] = self.max_length
        
        return stats


class TokenizedDataset(Dataset):
    """Custom Dataset class for tokenized data"""
    
    def __init__(self, encodings, dataframe):
        self.encodings = encodings
        self.dataframe = dataframe
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def get_original_data(self, idx):
        """Get original data from the dataframe by index"""
        return self.dataframe.iloc[idx]
    

# Example usage of TokenizerHandler and TokenizedDataset

def main():
    # Example configuration
    config = {
        'model': {
            'name': 'gpt2'  # Replace with the desired model
        },
        'tokenization': {
            'max_length': 512,
            'padding': True,
            'truncation': True
        }
    }

    # Example DataFrames for train and validation (replace with actual data)
    train_df = pd.DataFrame({'text': ["Example sentence 1.", "Another sentence for tokenization."]})
    val_df = pd.DataFrame({'text': ["Validation sentence 1.", "This is another validation example."]})
    
    # Prepare splits dictionary
    splits = {
        'train': train_df,
        'validation': val_df
    }

    # Initialize TokenizerHandler
    tokenizer_handler = TokenizerHandler(config)

    # Tokenize the dataset
    tokenized_splits = tokenizer_handler.tokenize(splits)

    # Get tokenization statistics
    stats = tokenizer_handler.get_tokenization_stats(tokenized_splits)
    print("Tokenization Stats:", stats)

    # Example: Access the tokenized data and original data for an example
    tokenized_train = tokenized_splits['train']
    print("Example of tokenized data:", tokenized_train[0])

    original_data = tokenized_train.get_original_data(0)
    print("Original data (from DataFrame):", original_data)


if __name__ == "__main__":
    main()
