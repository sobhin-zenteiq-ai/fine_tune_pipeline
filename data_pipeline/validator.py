import pandas as pd
from typing import Dict, Any, Set, List

class DatasetValidator:
    """Validates datasets for different tasks and models"""
    
    def __init__(self, config: Dict[str, Any], task: str = "qa"):
        self.config = config
        self.task = task
        
        # Define required columns for each task
        self.task_schemas = {
            'qa': {
                'required_columns': {'instruction', 'input', 'output'},
                'optional_columns': {'id', 'title', 'context'},
                'description': 'Question Answering task'
            },
            'summarization': {
                'required_columns': {'instruction', 'input', 'output'},
                'optional_columns': {'id', 'title', 'document', 'summary'},
                'description': 'Text Summarization task'
            }
        }
    
    def validate_dataset(self, dataset_features: Set[str] = None, 
                        model: str = None, 
                        task_type: str = None, 
                        dataset_name: str = None) -> Dict[str, Any]:
        """
        Validate if a dataset is suitable for the given task and model
        
        Args:
            dataset_features: Set of column names in the dataset
            model: Model name (optional)
            task_type: Task type (optional, uses self.task if not provided)
            dataset_name: Name of the dataset (optional)
        
        Returns:
            Dictionary with validation results
        """
        task_to_validate = task_type or self.task
        
        # Get schema for the task
        if task_to_validate not in self.task_schemas:
            return {
                'is_valid': False,
                'error_message': f"Unknown task: {task_to_validate}",
                'warnings': [],
                'task': task_to_validate
            }
        
        schema = self.task_schemas[task_to_validate]
        required_columns = schema['required_columns']
        
        # Check if dataset_features is provided
        if dataset_features is None:
            return {
                'is_valid': False,
                'error_message': "Dataset features not provided for validation",
                'warnings': [],
                'task': task_to_validate
            }
        
        # Check for required columns
        missing_columns = required_columns - dataset_features
        if missing_columns:
            return {
                'is_valid': False,
                'error_message': f"Missing required columns for {task_to_validate}: {missing_columns}",
                'warnings': [],
                'task': task_to_validate,
                'missing_columns': list(missing_columns),
                'required_columns': list(required_columns)
            }
        
        # Generate warnings for unexpected columns
        expected_columns = required_columns | schema['optional_columns']
        unexpected_columns = dataset_features - expected_columns
        warnings = []
        
        if unexpected_columns:
            warnings.append(f"Unexpected columns found: {unexpected_columns}")
        
        return {
            'is_valid': True,
            'error_message': None,
            'warnings': warnings,
            'task': task_to_validate,
            'schema_description': schema['description'],
            'required_columns': list(required_columns),
            'found_columns': list(dataset_features)
        }
    
    def validate_dataframe(self, df: pd.DataFrame, task_type: str = None) -> Dict[str, Any]:
        """
        Validate a pandas DataFrame for the given task
        
        Args:
            df: DataFrame to validate
            task_type: Task type (optional, uses self.task if not provided)
        
        Returns:
            Dictionary with validation results
        """
        if df is None or df.empty:
            return {
                'is_valid': False,
                'error_message': "DataFrame is None or empty",
                'warnings': [],
                'task': task_type or self.task
            }
        
        # Get dataset features (column names)
        dataset_features = set(df.columns)
        
        # Validate using the dataset_features method
        validation_result = self.validate_dataset(
            dataset_features=dataset_features,
            task_type=task_type
        )
        
        # Add DataFrame-specific validation
        if validation_result['is_valid']:
            # Check for empty columns
            empty_columns = []
            for col in df.columns:
                if df[col].isna().all() or (df[col].astype(str).str.strip() == '').all():
                    empty_columns.append(col)
            
            if empty_columns:
                validation_result['warnings'].append(f"Empty columns found: {empty_columns}")
            
            # Add data quality metrics
            validation_result['data_quality'] = {
                'total_rows': len(df),
                'null_counts': df.isnull().sum().to_dict(),
                'empty_string_counts': (df.astype(str).str.strip() == '').sum().to_dict(),
                'duplicate_rows': df.duplicated().sum()
            }
        
        return validation_result
    
    def get_task_schema(self, task_type: str = None) -> Dict[str, Any]:
        """
        Get the schema for a specific task
        
        Args:
            task_type: Task type (optional, uses self.task if not provided)
        
        Returns:
            Dictionary with task schema information
        """
        task_to_get = task_type or self.task
        
        if task_to_get not in self.task_schemas:
            return {
                'error': f"Unknown task: {task_to_get}",
                'available_tasks': list(self.task_schemas.keys())
            }
        
        return self.task_schemas[task_to_get]
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available tasks"""
        return list(self.task_schemas.keys())
    
    def suggest_column_mapping(self, dataset_features: Set[str], task_type: str = None) -> Dict[str, str]:
        """
        Suggest column mapping for a dataset based on column names
        
        Args:
            dataset_features: Set of column names in the dataset
            task_type: Task type (optional, uses self.task if not provided)
        
        Returns:
            Dictionary with suggested column mappings
        """
        task_to_map = task_type or self.task
        
        if task_to_map not in self.task_schemas:
            return {}
        
        # Common column name mappings
        mapping_suggestions = {
            'qa': {
                'question': 'instruction',
                'context': 'input',
                'answer': 'output',
                'text': 'input'
            },
            'summarization': {
                'document': 'input',
                'article': 'input',
                'summary': 'output',
                'highlights': 'output',
                'text': 'input'
            }
        }
        
        suggestions = {}
        task_mappings = mapping_suggestions.get(task_to_map, {})
        
        for dataset_col in dataset_features:
            if dataset_col.lower() in task_mappings:
                suggestions[dataset_col] = task_mappings[dataset_col.lower()]
        
        return suggestions