import pandas as pd
from typing import Dict, Any


class DataFormatter:
    """Handles formatting data for training"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_type = config.get('model', {}).get('task', 'qa').lower()

    def format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the dataset for training by combining instruction, input, and output.
        Adds a 'tokenizer_input' column.
        """
        print("Starting data formatting…")

        formatters = {
            'qa': self._format_qa,
            'summarization': self._format_summarization,
            'translation': self._format_translation,
            'classification': self._format_classification,
            'lm': lambda row: str(row.get('input', ''))
        }

        if self.task_type not in formatters:
            raise ValueError(f"Task not supported for formatting: {self.task_type}")

        formatter = formatters[self.task_type]
        df = df.copy()
        df['tokenizer_input'] = df.apply(formatter, axis=1)

        print(f"Formatted {len(df)} examples.")
        return df

    # ─────────────────────────────────────────────────────────────────────────────
    #                          FORMATTER METHODS
    # ─────────────────────────────────────────────────────────────────────────────
    def _format_qa(self, row: pd.Series) -> str:
        instruction = str(row.get('instruction', '')).strip()
        input_text  = str(row.get('input', '')).strip()
        output      = str(row.get('output', '')).strip()

        parts = [f"### Instruction:\n{instruction}"]
        if input_text:
            parts.append(f"### Input:\n{input_text}")
        parts.append(f"### Response:\n{output}")

        return "\n\n".join(parts)

    def _format_summarization(self, row: pd.Series) -> str:
        article = str(row.get('input', '')).strip()
        summary = str(row.get('output', '')).strip()

        parts = [f"### Summarize the article:\n{article}"]
        if summary:
            parts.append(f"### Summary:\n{summary}")

        return "\n\n".join(parts)

    def _format_translation(self, row: pd.Series) -> str:
        source = str(row.get('input', '')).strip()
        target = str(row.get('output', '')).strip()

        return f"### Translate this sentence:\n{source}\n\n### Into:\n{target}"

    def _format_classification(self, row: pd.Series) -> str:
        text  = str(row.get('input', '')).strip()
        label = str(row.get('output', '')).strip()

        return f"### Classify this sentence:\n{text}\n\n### Label:\n{label}"

    # ─────────────────────────────────────────────────────────────────────────────
    #                          STATS METHOD
    # ─────────────────────────────────────────────────────────────────────────────
    def get_format_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return basic statistics about the formatted data."""
        if 'tokenizer_input' not in df.columns:
            return {}

        lengths = df['tokenizer_input'].str.len()
        return {
            'total_examples': len(df),
            'avg_text_length': lengths.mean(),
            'min_text_length': lengths.min(),
            'max_text_length': lengths.max(),
            'median_text_length': lengths.median()
        }
