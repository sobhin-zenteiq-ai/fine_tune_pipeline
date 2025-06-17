import pandas as pd
import torch
from transformers import AutoTokenizer
from typing import Dict, Any, List
from torch.utils.data import Dataset
import datasets
import os

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
        self.batch_size = tokenization_config.get('batch_size', 8)
        self.task = model_config.get('task', 'qa')
        
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
        """
        Efficiently tokenize a DataFrame using Hugging Face Datasets for large-scale processing.
        Applies label masking for decoder-only models.
        """
        # Map task to separator
        separator_map = {
            'qa': '### Response:\n',
            'summarization': '### summary:\n',
            'translation': '### into:\n'
        }

        task = self.task.lower()
        separator = separator_map.get(task, None)

        # Convert DataFrame to HF Dataset
        hf_dataset = datasets.Dataset.from_pandas(df[['tokenizer_input']])

        def tokenize_and_mask(batch):
            encodings = self.tokenizer(
                batch['tokenizer_input'],
                truncation=self.truncation,
                padding='max_length',
                max_length=self.max_length,
            )
            
            if task in separator_map:
                labels = []
                for text, input_ids in zip(batch['tokenizer_input'], encodings['input_ids']):
                    # Try masking the prompt part
                    try:
                        prompt = text.split(separator)[0] + separator
                    except IndexError:
                        prompt = text  # fallback
                    
                    prompt_tokens = self.tokenizer(
                        prompt,
                        truncation=self.truncation,
                        max_length=self.max_length,
                    )['input_ids']
                    
                    # Mask prompt part in labels
                    label = input_ids.copy()
                    prompt_len = len(prompt_tokens)
                    label[:prompt_len] = [-100] * prompt_len
                    labels.append(label)
                
                encodings['labels'] = labels
            else:
                # No masking for classification or unknown tasks
                encodings['labels'] = encodings['input_ids'].copy()

            return encodings

        # Use batched mapping for speed and parallelism
        tokenized_dataset = hf_dataset.map(
            tokenize_and_mask,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=["tokenizer_input"],
            num_proc=1,  # Increase if CPU-bound and using multiple cores
            load_from_cache_file=True
        )

        # Convert fields to torch.Tensor
        input_ids = torch.tensor(tokenized_dataset["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized_dataset["attention_mask"], dtype=torch.long)
        if (self.task == 'classification'): labels = torch.tensor(df['label_id'].tolist(), dtype=torch.long)
        else : labels = torch.tensor(tokenized_dataset["labels"], dtype=torch.long)

        return TokenizedDataset(input_ids, attention_mask, labels)

    
    def get_tokenization_stats(self, tokenized_splits: Dict[str, 'TokenizedDataset']) -> Dict[str, Any]:
        """Get statistics about tokenization"""
        stats = {}
        
        for split_name, dataset in tokenized_splits.items():
            input_ids = dataset.input_ids
            
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
    def __init__(self, input_ids, attention_mask,labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels if labels is not None else input_ids.clone()

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
