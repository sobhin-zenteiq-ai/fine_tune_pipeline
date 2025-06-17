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
    

    def tokenize(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, Dataset]:
        """
        Tokenize each DataFrame split and return a dict of Hugging Face Datasets.

        Args:
            splits: {"train": df_train, "validation": df_val, ...}

        Returns:
            {"train": Dataset, "validation": Dataset, ...}
            (Each Dataset already contains `input_ids`, `attention_mask`, `labels`
            and is formatted as PyTorch Tensors.)
        """
        print("Starting tokenization...")
        tokenized_splits: Dict[str, Dataset] = {}

        for split_name, df in splits.items():
            print(f"Tokenizing {split_name} set...")
            tokenized_ds = self._tokenize_dataframe(df)     # <- now returns HF Dataset
            tokenized_splits[split_name] = tokenized_ds
            print(f"Tokenized {len(tokenized_ds)} examples for {split_name}")

        # Optional: wrap in a DatasetDict if that’s more convenient downstream
        # return DatasetDict(tokenized_splits)  # uncomment if desired
        return tokenized_splits



# --------------------------------------------------------------------------- #
# 2.  Internal helper – tokenizes ONE DataFrame and returns an HF Dataset
# --------------------------------------------------------------------------- #
    def _tokenize_dataframe(self, df: pd.DataFrame) -> Dataset:
        """
        Efficiently tokenize a DataFrame and return a Hugging Face Dataset.
        • Supports label‑masking for decoder‑only tasks (qa / summarization / translation)
        • For classification tasks, expects an integer column `label_id`
        """

        # ----------------------------- task‑specific setup ---------------------- #
        separator_map = {
            "qa":            "### Response:\n",
            "summarization": "### summary:\n",
            "translation":   "### into:\n",
        }
        task      = self.task.lower()
        separator = separator_map.get(task)        # None for classification

        # --------------------------- build a raw HF Dataset --------------------- #
        # Store only the text column plus label_id when present.
        keep_cols = ["tokenizer_input"] + (["label_id"] if "label_id" in df.columns else [])
        hf_ds = datasets.Dataset.from_pandas(df[keep_cols].reset_index(drop=True))

        # ----------------------------- tokenize function ------------------------ #
        def tokenize_and_mask(batch):
            enc = self.tokenizer(
                batch["tokenizer_input"],
                truncation=self.truncation,
                padding="max_length",
                max_length=self.max_length,
            )

            # -------- label masking for generative tasks ------------------------ #
            if task in separator_map:
                masked = []
                for text, ids in zip(batch["tokenizer_input"], enc["input_ids"]):
                    prompt = (text.split(separator)[0] + separator) if separator in text else text
                    prompt_ids = self.tokenizer(
                        prompt,
                        truncation=self.truncation,
                        max_length=self.max_length,
                    )["input_ids"]

                    lbl = ids.copy()
                    lbl[:len(prompt_ids)] = [-100] * len(prompt_ids)  # mask prompt tokens
                    masked.append(lbl)
                enc["labels"] = masked
            else:
                # Classification or unknown → no masking here
                # We'll attach the integer label later.
                enc["labels"] = enc["input_ids"].copy()

            return enc

        # ---------------------------- batched mapping -------------------------- #
        tokenized_ds = hf_ds.map(
            tokenize_and_mask,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=["tokenizer_input"],
            num_proc=1,                 # bump if you can parallelise
            load_from_cache_file=True,
        )

        # ------------------------------ attach labels -------------------------- #
        if task == "classification":
            # Use the integer labels already in the DataFrame.
            tokenized_ds = tokenized_ds.add_column("labels", df["label_id"].tolist())

        # --------------------------- make tensors on demand -------------------- #
        tokenized_ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=False,
        )

        return tokenized_ds

    
    def get_tokenization_stats(self, tokenized_splits: Dict[str, 'TokenizedDataset']) -> Dict[str, Any]:
        """Get statistics about tokenization"""
        stats = {}
        
        for split_name, dataset in tokenized_splits.items():
            input_ids = dataset['input_ids']
            
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
