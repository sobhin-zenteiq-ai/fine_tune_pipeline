import os
import json
from typing import Dict, Any

import pandas as pd
from datasets import Dataset  # Hugging Face
import datasets               # for type hints
# No need to import / define TokenizedDataset anymore

class DataSaver:
    """Handles saving cleaned DataFrames, tokenized HF datasets, and stats."""

    # --------------------------------------------------------------------- init
    def __init__(self, config: Dict[str, Any]):
        output_cfg         = config.get("output", {})
        self.save_cleaned  = output_cfg.get("save_cleaned", True)
        self.save_tokenized = output_cfg.get("save_tokenized", True)
        self.save_stats    = output_cfg.get("save_stats", True)
        self.output_dir    = output_cfg.get("output_dir", "output")

        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------------------------- api
    def save(
        self,
        splits: Dict[str, pd.DataFrame],
        tokenized_splits: Dict[str, Dataset],          # <‑‑ updated
        stats: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Save cleaned CSV/JSON, tokenized Dataset(s), and stats.
        Returns a dict {logical_name: path}.
        """
        print("Starting data saving …")
        saved_files: Dict[str, str] = {}

        if self.save_cleaned:
            saved_files |= self._save_cleaned_data(splits)

        if self.save_tokenized:
            saved_files |= self._save_tokenized_data(tokenized_splits)

        if self.save_stats:
            saved_files["stats"] = self._save_stats(stats)

        print(f"Saved {len(saved_files)} artefacts to {self.output_dir}")
        return saved_files

    # --------------------------------------------------------- save cleaned csv/json
    def _save_cleaned_data(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        saved: Dict[str, str] = {}
        for name, df in splits.items():
            json_path = os.path.join(self.output_dir, f"{name}_cleaned.json")
            csv_path  = os.path.join(self.output_dir, f"{name}_cleaned.csv")

            df.to_json(json_path, orient="records", indent=2)
            df.to_csv(csv_path, index=False)

            saved[f"{name}_json"] = json_path
            saved[f"{name}_csv"]  = csv_path
            print(f"✓ Saved cleaned {name} ({len(df)} rows)")
        return saved

    # ---------------------------------------------------- save tokenized HF dataset
    def _save_tokenized_data(self, tokenized: Dict[str, Dataset]) -> Dict[str, str]:
        """
        Persist each Hugging Face `Dataset` to disk using `save_to_disk`.
        This creates a directory (Arrow format) which is lightweight, fast,
        and reload‑able anywhere with `datasets.load_from_disk`.
        """
        saved: Dict[str, str] = {}
        for name, ds in tokenized.items():
            dir_path = os.path.join(self.output_dir, f"{name}_tokenized")
            ds.save_to_disk(dir_path)
            saved[f"{name}_tokenized"] = dir_path
            print(f"✓ Saved tokenized {name} ({len(ds)} rows) → {dir_path}")
        return saved

    # --------------------------------------------------------------- save stats
    def _save_stats(self, stats: Dict[str, Any]) -> str:
        path = os.path.join(self.output_dir, "pipeline_stats.json")
        with open(path, "w") as f:
            json.dump(self._clean_stats_for_json(stats), f, indent=2)
        print("✓ Saved pipeline statistics")
        return path

    # ------------------------------------------------------ helper: clean numpy
    @staticmethod
    def _clean_stats_for_json(stats: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for k, v in stats.items():
            if hasattr(v, "item"):              # numpy scalar
                clean[k] = v.item()
            elif isinstance(v, (int, float, str, bool, list, dict)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    # ---------------------------------------------------------- pretty report txt
    def create_summary_report(self, stats: Dict[str, Any]) -> str:
        report_path = os.path.join(self.output_dir, "pipeline_report.txt")
        with open(report_path, "w") as f:
            f.write("Data Pipeline Processing Report\n" + "="*40 + "\n\n")
            # Basic counts
            for key in ("total_size", "train_size", "validation_size"):
                if key in stats:
                    f.write(f"{key.replace('_', ' ').title()}: {stats[key]}\n")
            f.write("\n")
            # Removal % etc.
            if "removal_percentage" in stats:
                f.write(f"Removal Percentage: {stats['removal_percentage']:.2f}%\n")
            if "train_avg_tokens" in stats:
                f.write(f"Avg Tokens (Train): {stats['train_avg_tokens']:.1f}\n")
            if "validation_avg_tokens" in stats:
                f.write(f"Avg Tokens (Val): {stats['validation_avg_tokens']:.1f}\n")
        print("✓ Created summary report")
        return report_path
