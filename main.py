#!/usr/bin/env python3
"""
Main script to run the data and LoRA fine-tuning pipeline
"""

import sys
import argparse
import torch
from data_pipeline import DataPipeline
from LoRA_pipeline import LoRAPipeline
import yaml


def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description='Run the fine-tuning data pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--step', type=str, choices=['load', 'clean', 'preprocess', 'format', 'split', 'tokenize'],
                        help='Run only a specific step in the data pipeline')
    
    args = parser.parse_args()

    # Load config from file
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config file: {e}")
        sys.exit(1)

    try:
        # Initialize data pipeline
        print("Initializing data pipeline...")
        data_pipeline = DataPipeline(config_path=args.config)

        if args.step:
            # Run specific step
            print(f"Running step: {args.step}")
            result = data_pipeline.run_step(args.step)
            print(f"Step {args.step} completed successfully")
            sys.exit(0)  # Exit after running just a step
        else:
            # Run full data pipeline
            print("Running full data pipeline...")
            result = data_pipeline.run()

            if result['success']:
                data_pipeline.print_summary()
                print("\nGenerated files:")
                for file_type, file_path in result['files'].items():
                    print(f"  {file_type}: {file_path}")
            else:
                print(f"Data pipeline failed: {result['error']}")
                sys.exit(1)

        # Run LoRA fine-tuning after data processing
        print("\nRunning LoRA fine-tuning pipeline...")
        lora_pipeline = LoRAPipeline(config)
        lora_pipeline.run()  # Run full LoRA training + evaluation + saving

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
