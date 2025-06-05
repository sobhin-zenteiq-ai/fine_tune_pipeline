from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModel,
    AutoTokenizer
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model_Loader:
    def __init__(self, config: dict):
        self.model_name = config.get("model", {}).get("name")
        self.task_type = config.get("model", {}).get("task", "").lower()

        if not self.model_name:
            raise ValueError("Model name not specified in the configuration.")

    def load_model(self):
        logger.info(f"Loading model '{self.model_name}' for task '{self.task_type}'")

        task_map = {
            "lm": AutoModelForCausalLM,
            "seq2seq": AutoModelForSeq2SeqLM,
            "qa": AutoModelForQuestionAnswering,
            "token-classification": AutoModelForTokenClassification,
            "sequence-classification": AutoModelForSequenceClassification,
            "auto": AutoModel,  # generic fallback
        }

        model_class = task_map.get(self.task_type)

        if model_class is None:
            raise ValueError(f"Unsupported task type: '{self.task_type}'. Available tasks: {list(task_map.keys())}")

        try:
            model = model_class.from_pretrained(self.model_name)
            logger.info(f"Successfully loaded model: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info(f"Successfully loaded tokenizer for model: {self.model_name}")
        return tokenizer

# Example usage:
# if __name__ == "__main__":
#     config = {
#         "model": {
#             "name": "openai-community/gpt2",
#             "task_type": "qa"
#         }
#     }

#     loader = Model_Loader(config)
#     model = loader.load_model()
