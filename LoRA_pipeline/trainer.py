import logging
from transformers import Trainer, TrainingArguments,DataCollatorForLanguageModeling

class LoRA_Trainer:
    def __init__(self, model,tokenizer ,config: dict, train_dataset=None, eval_dataset=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Load training arguments from config
        lora = config.get("LoRA", {})
        training_args_dict = lora.get("training_args", {})
        
        if not training_args_dict:
            raise ValueError("Missing 'training_args' section in config.")

        self.training_args = TrainingArguments(**training_args_dict)

        self.data_collator = DataCollatorForLanguageModeling(
                                    tokenizer=tokenizer,
                                    mlm=False  # Important: GPT-2 uses causal LM, not masked LM
                                )
        # Create internal Hugging Face Trainer
        self.trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator
        )

        self.logger.info("LoRA_Trainer initialized with training arguments from config.")

    def model_trainer(self, resume_from_checkpoint=None, trial=None, **kwargs):
        self.logger.info("Starting training process...")
        result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint, trial=trial, **kwargs)
        self.logger.info(f"Training completed. Metrics: {result.metrics}")
        return self.trainer.model

