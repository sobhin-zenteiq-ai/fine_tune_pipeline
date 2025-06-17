import pandas as pd
from typing import Dict, Any, Set


class DataCleaner:
    """Efficient, vectorised data‑cleaning utility."""

    # ────────────────────────────────────────────────
    #                INITIALISATION
    # ────────────────────────────────────────────────
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        c_conf          = config.get("cleaning", {})
        self.min_length = c_conf.get("min_length", 10)
        self.max_length = c_conf.get("max_length", 1000)
        self.remove_duplicates = c_conf.get("remove_duplicates", True)

        self.task = config.get("model", {}).get("task", "qa")
        self.req_columns: Set[str] = {"instruction", "output"} if self.task == "qa" else {"input", "output"}

    # ────────────────────────────────────────────────
    #                     PUBLIC
    # ────────────────────────────────────────────────
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a cleaned copy of *df*:
        • drop rows with NaNs / blank strings in required cols  
        • optionally drop duplicates (case‑insensitive)  
        • filter rows whose combined text length falls outside [min_length, max_length]
        """
        print("Starting data cleaning…")
        original_size = len(df)

        # Work on a copy and ensure required cols are string‑typed
        col_list = sorted(self.req_columns & set(df.columns))  # keep only existing columns
        df = df.copy()
        df[col_list] = df[col_list].astype(str)

        # 1. remove missing / purely‑whitespace rows
        mask_non_na = df[col_list].notna().all(axis=1)
        mask_non_blank = (
            df[col_list]
            .replace(r"^\s*$", pd.NA, regex=True)
            .notna()
            .all(axis=1)
        )
        df = df[mask_non_na & mask_non_blank]
        print(f"After removing missing / blank values: {len(df)} rows")

        # 2. remove duplicates
        if self.remove_duplicates:
            df[col_list] = df[col_list].apply(lambda s: s.str.casefold())  # case‑insensitive
            df = df.drop_duplicates(subset=col_list)
            print(f"After removing duplicates: {len(df)} rows")

        # 3. length filter
        total_len = df[col_list].map(len).sum(axis=1)
        df = df[(total_len >= self.min_length) & (total_len <= self.max_length)]
        print(f"After length filtering: {len(df)} rows")

        print(f"Cleaning complete. Removed {original_size - len(df)} rows.")
        return df.reset_index(drop=True)

    def get_cleaning_stats(
        self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Return basic stats about how many rows were kept / removed."""
        return {
            "original_size": len(original_df),
            "cleaned_size": len(cleaned_df),
            "removed_count": len(original_df) - len(cleaned_df),
            "removal_percentage": 100.0 * (len(original_df) - len(cleaned_df)) / len(original_df),
        }
