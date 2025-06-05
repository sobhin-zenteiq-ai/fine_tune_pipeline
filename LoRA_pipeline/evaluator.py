import logging
from transformers import Trainer, TrainingArguments, EvalPrediction
import numpy as np
import evaluate

class Evaluator:
    def __init__(self, model, tokenizer, eval_dataset, config: dict):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.config = config

        # Load training args related to evaluation
        training_args_config = config.get("training_args", {})
        self.training_args = TrainingArguments(
            output_dir=training_args_config.get("output_dir", "./outputs"),
            per_device_eval_batch_size=training_args_config.get("per_device_eval_batch_size", 4),
            do_eval=True,
            report_to="none"
        )

        # Optional metric configuration
        self.metrics = config.get("metrics", ["accuracy"])
        self.metric_functions = self._load_metrics()

        self.logger.info(f"Evaluator initialized with metrics: {self.metrics}")

    def _load_metrics(self):
        metric_funcs = []
        for metric_name in self.metrics:
            try:
                metric = evaluate.load(metric_name)
                metric_funcs.append(metric)
                self.logger.info(f"Loaded metric: {metric_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load metric '{metric_name}': {e}")
        return metric_funcs

    def compute_metrics(self, eval_pred: EvalPrediction):
        predictions = eval_pred.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.argmax(predictions, axis=-1)
        references = eval_pred.label_ids

        results = {}
        for metric in self.metric_functions:
            try:
                result = metric.compute(predictions=predictions, references=references)
                results.update(result)
            except Exception as e:
                self.logger.warning(f"Metric computation failed for {metric}: {e}")
        return results

    def run_evaluation(self):
        self.logger.info("Running evaluation...")
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics if self.metric_functions else None
        )

        metrics = trainer.evaluate()
        self.logger.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics
