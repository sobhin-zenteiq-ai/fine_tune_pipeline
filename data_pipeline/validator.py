from abc import ABC, abstractmethod
from typing import Dict, Set, Optional,Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskValidator(ABC):
    """Abstract base class for task-specific validators"""
    
    @abstractmethod
    def get_required_features(self) -> Set[str]:
        """Return required features for the task"""
        pass
    
    
    @abstractmethod
    def get_task_name(self) -> str:
        """Return task name"""
        pass
    
    def validate_features(self, dataset_features: Set[str]) -> bool:
        """Validate if dataset has required features"""
        required = self.get_required_features()
        return required.issubset(dataset_features)
    
    def get_missing_features(self, dataset_features: Set[str]) -> Set[str]:
        """Get missing required features"""
        required = self.get_required_features()
        return required - set(dataset_features)
    
    # def get_feature_compatibility_score(self, dataset_features: Set[str]) -> float:
    #     """Calculate compatibility score between 0 and 1"""
    #     required = self.get_required_features()
    #     optional = self.get_optional_features()
    #     all_possible = required.union(optional)
        
    #     if not all_possible:
    #         return 1.0
        
    #     matching_features = dataset_features.intersection(all_possible)
    #     return len(matching_features) / len(all_possible)


class QAValidator(TaskValidator):
    """Validator for Question Answering tasks"""
    
    def get_required_features(self) -> Set[str]:
        return {"instruction", "input", "output"}
    
    def get_task_name(self) -> str:
        return "Question Answering (QA)"


class LMValidator(TaskValidator):
    """Validator for Language Modeling tasks"""
    
    def get_required_features(self) -> Set[str]:
        return {"text"}
    
    def get_task_name(self) -> str:
        return "Language Modeling (LM)"


class SummarizationValidator(TaskValidator):
    """Validator for Summarization tasks"""
    
    def get_required_features(self) -> Set[str]:
        return {"articles", "summaries"}
    
    def get_task_name(self) -> str:
        return "Summarization"


class ClassificationValidator(TaskValidator):
    """Validator for Text Classification tasks"""
    
    def get_required_features(self) -> Set[str]:
        return {"text", "label"}
    
    def get_task_name(self) -> str:
        return "Text Classification"


class TranslationValidator(TaskValidator):
    """Validator for Translation tasks"""
    
    def get_required_features(self) -> Set[str]:
        return {"source", "target"}
    
    def get_task_name(self) -> str:
        return "Translation"


class ConversationValidator(TaskValidator):
    """Validator for Conversational AI tasks"""
    
    def get_required_features(self) -> Set[str]:
        return {"conversations"}
    
    def get_optional_features(self) -> Set[str]:
        return {"dialogue", "messages", "chat", "prompt", "response", "turn"}
    
    def get_task_name(self) -> str:
        return "Conversation/Chat"


class SentimentValidator(TaskValidator):
    """Validator for Sentiment Analysis tasks"""
    
    def get_required_features(self) -> Set[str]:
        return {"text", "sentiment"}
    
    def get_task_name(self) -> str:
        return "Sentiment Analysis"


class NERValidator(TaskValidator):
    """Validator for Named Entity Recognition tasks"""
    
    def get_required_features(self) -> Set[str]:
        return {"tokens", "ner_tags"}
    
    def get_task_name(self) -> str:
        return "Named Entity Recognition (NER)"


class DatasetValidationResult:
    """Class to hold validation results"""
    
    def __init__(self, is_valid: bool, task_name: str, dataset_name: str, 
                 missing_features: Set[str] = None, available_features: Set[str] = None,
                 error_message: str = None, compatibility_score: float = 0.0):
        self.is_valid = is_valid
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.missing_features = missing_features or set()
        self.available_features = available_features or set()
        self.error_message = error_message
        self.compatibility_score = compatibility_score
    
    def __str__(self) -> str:
        if self.error_message:
            return f"❌ Error: {self.error_message}"
        
        if self.is_valid:
            return (f"✅ Dataset '{self.dataset_name}' is suitable for {self.task_name} task\n"
                   f"   Compatibility Score: {self.compatibility_score:.2%}")
        else:
            return (f"❌ This dataset cannot be used to finetune for this specific task\n"
                   f"   Task: {self.task_name}\n"
                   f"   Dataset: {self.dataset_name}\n"
                   f"   Missing required features: {', '.join(self.missing_features)}\n"
                   f"   Available features: {', '.join(self.available_features)}\n"
                   f"   Compatibility Score: {self.compatibility_score:.2%}")
    
    def to_dict(self) -> Dict:
        """Convert validation result to dictionary"""
        return {
            'is_valid': self.is_valid,
            'task_name': self.task_name,
            'dataset_name': self.dataset_name,
            'missing_features': list(self.missing_features),
            'available_features': list(self.available_features),
            'error_message': self.error_message,
            'compatibility_score': self.compatibility_score
        }


class DatasetValidator:
    """Main validator class that coordinates validation for different tasks"""
    
    def __init__(self,config: Dict[str, Any]):
        self.validators = {
            'qa': QAValidator(),
            'lm': LMValidator(),
            'summarization': SummarizationValidator(),
            'classification': ClassificationValidator(),
            'translation': TranslationValidator(),
            'conversation': ConversationValidator(),
            'sentiment': SentimentValidator(),
            'ner': NERValidator()
        }
        
        self.supported_models = {
            'gpt', 'bert', 'roberta', 'distilbert', 'electra', 'deberta',
            'llama', 'mistral', 'falcon', 'bloom', 'opt', 'gpt-neo',
            'gpt-j', 'pythia', 'dolly', 'vicuna', 'alpaca', 'chatglm',
            'claude', 'gemini', 'palm', 't5', 'bart', 'pegasus'
        }

        self.task_type = config.get('model', {}).get('task', 'lm')
        self.model_name = config.get('model', {}).get('name', 'Unknown')
        self.dataset_name = config.get('dataset', {}).get('name', 'Unknown')
    
    def is_supported_model(self, model_name = None) -> bool:
        """Check if the model is supported"""
        if model_name is None:
            model_name = self.model_name
        model_lower = model_name.lower()
        return any(supported in model_lower for supported in self.supported_models)
    
    def validate_dataset(self, model = None, task_type = None, dataset_features: Set[str] = None, 
                        dataset_name=None) -> Dict[str, Any]:
        """
        Main validation method
        
        Args:
            model: Model name (e.g., 'llama', 'gpt', 'bert')
            task_type: Task type ('qa', 'lm', 'summarization', etc.)
            dataset_features: Set of available features/columns in dataset
            dataset_name: Name of the dataset for reporting
            
        Returns:
            DatasetValidationResult object
        """
        if model is None:
            model = self.model_name
        if task_type is None:
            task_type = self.task_type
            #print(f"Using default task type: {task_type}")
        if dataset_name is None:
            dataset_name = self.dataset_name
        
         # Get validator and validate
        # Validate task type
        task_type_lower = task_type.lower()
        

        # Validate model
        if not self.is_supported_model(self.model_name):
            return {
                'is_valid':False,
                'task_name':task_type,
                'dataset_name':dataset_name,
                'error_message':f"Model '{self.model_name}' is not supported or recognized"
            }
        
        
        if task_type_lower not in self.validators:
            available_tasks = ', '.join(self.validators.keys())
            return {
                'is_valid':False,
                'task_name':task_type,
                'dataset_name':dataset_name,
                'error_message': f"Task type: {task_type} not supported. Available tasks: {available_tasks}"
            }
        
        validator = self.validators[task_type_lower]
        is_valid = validator.validate_features(dataset_features)
        missing_features = validator.get_missing_features(dataset_features)
        #compatibility_score = validator.get_feature_compatibility_score(dataset_features)
        
        if not is_valid:
            return{
                'is_valid': False,
                'error_message': f"Dataset '{dataset_name}' is not suitable for {task_type} task. Available features: {', '.join(dataset_features)} .Missing required features: {', '.join(missing_features)}",
            }
            
       
        return {
            'is_valid': is_valid,
            'task_name':task_type,
            'available_features':dataset_features,
           # 'compatibility_score':compatibility_score
        }
    