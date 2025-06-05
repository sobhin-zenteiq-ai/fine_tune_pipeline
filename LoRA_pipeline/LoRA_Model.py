import logging
from peft import LoraConfig, get_peft_model
from peft.utils import TaskType

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoRA_Model:
    def __init__(self, model, config: dict):
        self.model = model
        self.lora_config = config.get("lora", {})
        self.model_config = config.get("model", {})
        task_type_map = {
            "lm": TaskType.CAUSAL_LM,
            "causal_lm": TaskType.CAUSAL_LM,
            "seq2seq": TaskType.SEQ_2_SEQ_LM,
            "sequence_to_sequence": TaskType.SEQ_2_SEQ_LM,
            "text2text": TaskType.SEQ_2_SEQ_LM,
            "qa": TaskType.SEQ_2_SEQ_LM,
            "question_answering": TaskType.SEQ_2_SEQ_LM,
            "classification": TaskType.SEQ_CLS,
            "sequence_classification": TaskType.SEQ_CLS,
            "token_classification": TaskType.TOKEN_CLS,
            "tokencls": TaskType.TOKEN_CLS,
            "generation": TaskType.CAUSAL_LM,
        }

        try:
            self.lora_rank = self.lora_config.get("rank", 8)
            self.lora_alpha = self.lora_config.get("alpha", 16)
            self.lora_dropout = self.lora_config.get("dropout", 0.1)

            task_str = self.model_config.get("task_type", "CAUSAL_LM").lower()
            if task_str not in task_type_map:
                raise ValueError(f"Unsupported task_type '{task_str}'. Available options: {list(task_type_map.keys())}")

            self.task_type = task_type_map[task_str]
            self.target_modules = self.lora_config.get("target_modules", None)

            logger.info(f"Initialized LoRA config with rank={self.lora_rank}, alpha={self.lora_alpha}, "
                        f"dropout={self.lora_dropout}, task_type={self.task_type}, "
                        f"target_modules={self.target_modules}")
        except Exception as e:
            logger.error(f"Error initializing LoRA_Model: {e}")
            raise

    def apply_lora(self):
        try:
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=self.task_type,
                target_modules=self.target_modules,
            )

            model_with_lora = get_peft_model(self.model, lora_config)
            logger.info("Successfully applied LoRA to the model.")
            return model_with_lora

        except Exception as e:
            logger.error(f"Failed to apply LoRA: {e}")
            raise

# Example usage:
# if __name__ == "__main__":
#     # Example model and config
#     model = None  # Replace with actual model loading logic
#     config = {
#         "lora": {
#             "rank": 8,
#             "alpha": 16,
#             "dropout": 0.1,
#             "target_modules": ["q_proj", "v_proj"]
#         },
#         "model": {
#             "task_type": "lm"
#         }
#     }

#     lora_model = LoRA_Model(model, config)
#     model_with_lora = lora_model.apply_lora()
#     print(model_with_lora)  # This will print the model with LoRA applied
