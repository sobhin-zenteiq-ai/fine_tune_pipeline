import os
import logging

class Saver:
    def __init__(self, model, tokenizer, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.output_dir = config.get("save", {}).get("output_dir", "./saved_model")
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self):
        try:
            # Save model
            model_path = os.path.join(self.output_dir, "model")
            self.model.save_pretrained(model_path)
            self.logger.info(f"Model saved at: {model_path}")

            # Save tokenizer
            tokenizer_path = os.path.join(self.output_dir, "tokenizer")
            self.tokenizer.save_pretrained(tokenizer_path)
            self.logger.info(f"Tokenizer saved at: {tokenizer_path}")

            # Optional: Save config as JSON
            config_path = os.path.join(self.output_dir, "config.json")
            with open(config_path, "w") as f:
                import json
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Config saved at: {config_path}")

        except Exception as e:
            self.logger.error(f"Failed to save model/tokenizer: {e}")
            raise
