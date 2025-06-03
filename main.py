#!/usr/bin/env python3
"""
Main script to run the data pipeline
"""

import sys
import argparse
from data_pipeline import DataPipeline

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description='Run the fine-tuning data pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--step', type=str, choices=['load', 'clean', 'preprocess', 'format', 'split', 'tokenize'],
                        help='Run only a specific step')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        print("Initializing data pipeline...")
        pipeline = DataPipeline(config_path=args.config)
        
        if args.step:
            # Run specific step
            print(f"Running step: {args.step}")
            result = pipeline.run_step(args.step)
            print(f"Step {args.step} completed successfully")
        else:
            # Run full pipeline
            print("Running full pipeline...")
            result = pipeline.run()
            
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