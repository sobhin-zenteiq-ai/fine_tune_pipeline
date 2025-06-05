import logging
import torch
from .Model_Loader import Model_Loader
from .LoRA_Model import LoRA_Model
from .saver import Saver
from data_pipeline.saver import TokenizedDataset
from .trainer import LoRA_Trainer
from .evaluator import Evaluator

torch.serialization.add_safe_globals([TokenizedDataset])

class LoRAPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Internal components
        self.model_loader = Model_Loader(config)
        self.model = None
        self.tokenizer = None
        self.model_with_lora = None
        self.trained_model = None
        self.evaluation_results = None
        self.train_dataset = None
        self.eval_dataset = None
        self.metrics = None
        self.saver = None

    def run(self):
        """
        Run the full LoRA fine-tuning pipeline
        """
        self.logger.info("Running full LoRA pipeline...")

        # Step 1: Load model and tokenizer
        self.model = self.model_loader.load_model()
        self.tokenizer = self.model_loader.load_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        # Step 2: Load tokenized datasets
        self.train_dataset = torch.load(self.config['dataset']['tokenized_train_path'],weights_only=False)
        self.eval_dataset = torch.load(self.config['dataset']['tokenized_validation_path'],weights_only=False)

        # Step 3: Apply LoRA
        lora_applier = LoRA_Model(self.model, self.config)
        self.model_with_lora = lora_applier.apply_lora()

        # Step 4: Train the model
        trainer = LoRA_Trainer(
            model=self.model_with_lora,
            tokenizer=self.tokenizer,
            config=self.config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            #compute_metrics=self.model_loader.load_compute_metrics()
        )
        print("Starting training loop...")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Evaluation samples: {len(self.eval_dataset)}")
        #print(f"Training args: {training_args}")
        self.trained_model = trainer.model_trainer()

        # # Step 5: Evaluate the model
        # evaluator = Evaluator(self.trained_model, self.tokenizer, self.config)
        # self.evaluation_results = evaluator.evaluate(self.eval_dataset)

        # Step 6: Save the model and tokenizer
        self.saver = Saver(self.trained_model, self.tokenizer, self.config)
        self.saver.save()

        self.logger.info("LoRA pipeline completed successfully.")
        return {
            "model": self.trained_model,
            "evaluation": self.evaluation_results
        }

    def run_step(self, step_name: str):
        """
        Run a specific step of the LoRA pipeline
        """
        steps = {
            'load': self._load_model_and_data,
            'apply_lora': self._apply_lora,
            'train': self._train_model,
            'evaluate': self._evaluate_model,
            'save': self._save_model
        }

        if step_name not in steps:
            raise ValueError(f"Unknown step: {step_name}")

        return steps[step_name]()

    def _load_model_and_data(self):
        self.logger.info("Loading model, tokenizer, and tokenized data...")
        self.model = self.model_loader.load_model()
        self.tokenizer = self.model_loader.load_tokenizer()
        self.train_dataset = torch.load(self.config['dataset']['tokenized_train_path'])
        self.eval_dataset = torch.load(self.config['dataset']['tokenized_validation_path'])
        self.metrics = self.model_loader.load_compute_metrics()

    def _apply_lora(self):
        if not self.model:
            raise RuntimeError("Model not loaded. Run 'load' step first.")
        lora_applier = LoRA_Model(self.model, self.config)
        self.model_with_lora = lora_applier.apply_lora()

    def _train_model(self):
        if not self.model_with_lora or self.train_dataset is None:
            raise RuntimeError("Model with LoRA or dataset not prepared.")
        trainer = LoRA_Trainer(
            model=self.model_with_lora,
            config=self.config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.metrics
        )
        self.trained_model = trainer.model_trainer()

    def _evaluate_model(self):
        if not self.trained_model or not self.eval_dataset:
            raise RuntimeError("Model or eval dataset missing.")
        evaluator = Evaluator(self.trained_model, self.tokenizer, self.config)
        self.evaluation_results = evaluator.evaluate(self.eval_dataset)

    def _save_model(self):
        if not self.trained_model or not self.tokenizer:
            raise RuntimeError("Nothing to save. Run training first.")
        self.saver = Saver(self.trained_model, self.tokenizer, self.config)
        self.saver.save()
