#!/usr/bin/env python3
"""
Main script to run the data pipeline
"""

import sys
import argparse
import pandas as pd
from data_pipeline import DataPipeline

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description='Run the fine-tuning data pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--task', type=str, choices=['qa', 'summarization'], required=True, help='Choose the task')
    parser.add_argument('--dataset', type=str, help='Select dataset for the task')
    parser.add_argument('--upload', type=str, help='Upload custom dataset path')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets for the task')

    args = parser.parse_args()

    try:
        # Initialize pipeline
        print("Initializing data pipeline...")
        pipeline = DataPipeline(config_path=args.config, task=args.task, dataset_name=args.dataset)

        # If list-datasets is requested, show available datasets and exit
        if args.list_datasets:
            available_datasets = pipeline.get_available_datasets()
            print(f"\nAvailable datasets for task '{args.task}':")
            for dataset in available_datasets:
                print(f"  - {dataset}")
            return
        
        # Check if dataset is provided
        if not args.dataset and not args.upload:
            available_datasets = pipeline.get_available_datasets()
            print(f"\nError: Please specify a dataset using --dataset or upload a custom dataset using --upload")
            print(f"Available datasets for task '{args.task}':")
            for dataset in available_datasets:
                print(f"  - {dataset}")
            sys.exit(1)

        # Handle custom dataset upload
        if args.upload:
            print(f"Uploading custom dataset: {args.upload}")
            # Handle the upload of custom dataset logic here
            try:
                custom_df = pd.read_csv(args.upload)  # Update as necessary for file types
                print(f"Loaded custom dataset with {len(custom_df)} rows")
                # TODO: Add validation for custom dataset based on task
                # For now, we'll assume the custom dataset is properly formatted
            except Exception as e:
                print(f"Error loading custom dataset: {str(e)}")
                sys.exit(1)

        # Run the pipeline
        if args.dataset:
            print(f"Using dataset: {args.dataset}")
            result = pipeline.run(dataset_name=args.dataset)
        else:
            # For custom upload, we would need to modify the pipeline to accept custom dataframes
            print("Custom dataset upload is not fully implemented yet")
            sys.exit(1)
            
        if result['success']:
            # Print summary
            pipeline.print_summary()
            print("\nGenerated files:")
            for file_type, file_path in result['files'].items():
                print(f"  {file_type}: {file_path}")
        else:
            print(f"Pipeline failed: {result['error']}")
            sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()