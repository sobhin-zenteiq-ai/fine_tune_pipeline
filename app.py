#!/usr/bin/env python3
"""
Gradio UI for Fine-tuning Data Pipeline
A comprehensive web interface for dataset processing and fine-tuning preparation
"""

import gradio as gr
import pandas as pd
from functools import partial
import json
import os
import yaml
from typing import Dict, Any, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging

# Import your pipeline components
from data_pipeline import DataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = "C:\\Users\\HP\\Desktop\\Pipeline\\fine_tune_pipeline\\config.yaml"
def update_task_in_config(selected_task,name,input_col,instruction_col,output_col, CONFIG_PATH: str = CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["task"] = selected_task  # 👈 Update task field
    config["dataset"]["name"] = name  # 👈 Update dataset name field
    config["dataset"]["input"] = input_col
    config["dataset"]["instruction"] = instruction_col
    config["dataset"]["output"] = output_col

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)

class GradioDataPipelineUI:
    def __init__(self):
        self.pipeline = None
        self.current_stats = {}
        self.processing_history = []
        
    def create_pipeline(self, task: str, dataset_name,config_path: str = "config.yaml") -> DataPipeline:
        """Create a new pipeline instance"""
        try:
            return DataPipeline(config_path=config_path, task=task,dataset_name=dataset_name)
        except Exception as e:
            logger.error(f"Error creating pipeline: {str(e)}")
            raise gr.Error(f"Failed to create pipeline: {str(e)}")
    
    def get_available_datasets(self, task: str) -> List[str]:
        """Get available datasets for the selected task"""
        try:
            temp_pipeline = self.create_pipeline(task)
            return temp_pipeline.get_available_datasets()
        except Exception as e:
            return []
    
    def validate_custom_dataset(self, file_obj) -> Tuple[bool, str, pd.DataFrame]:
        """Validate uploaded custom dataset"""
        if file_obj is None:
            return False, "No file uploaded", pd.DataFrame()
        
        try:
            # Read the file
            if file_obj.name.endswith('.csv'):
                df = pd.read_csv(file_obj.name)
            elif file_obj.name.endswith('.json'):
                df = pd.read_json(file_obj.name)
            else:
                return False, "Unsupported file format. Please upload CSV or JSON.", pd.DataFrame()
            
            
            validation_msg = f"✅ Dataset is valid!\n"
            validation_msg += f"📊 Total examples: {len(df)}\n"
            validation_msg += f"📝 Columns: {list(df.columns)}\n"
            
            
            return True, validation_msg, df
            
        except Exception as e:
            return False, f"Error validating dataset: {str(e)}", pd.DataFrame()
    
    def empty_plot(self)->go.Figure:
        fig = go.Figure()
        fig.update_layout(title="No data", template="plotly_white")
        return fig
    
    def process_dataset(self, task: str, dataset_source: str, dataset_name: str = None, 
                       custom_file=None, progress=gr.Progress()) -> Tuple[str, go.Figure, str, str]:
        """Process the dataset through the pipeline"""
        try:
            progress(0, desc="Initializing pipeline...")
            
            # Create pipeline
            self.pipeline = self.create_pipeline(task,dataset_name)
            df = None
            # Handle custom dataset
            if dataset_source == "Custom Upload" and custom_file is not None:
                progress(0.1, desc="Validating custom dataset...")
                is_valid, validation_msg, df = self.validate_custom_dataset(custom_file)
                
                if not is_valid:
                    return validation_msg, self.empty_plot(), "", ""
                
                # Save custom dataset temporarily
                temp_path = f"temp_custom_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(temp_path, index=False)
                
                # Note: This would require modifying your pipeline to accept file paths
                # For now, we'll use a predefined dataset
                dataset_name = list(self.pipeline.get_available_datasets())[0] if self.pipeline.get_available_datasets() else ""
            
            elif dataset_source == "Hugging Face":
                if not dataset_name:
                    return "❌ Please select a dataset", self.empty_plot(), "", ""
            
            progress(0.2, desc="Starting data processing...")
            
            # Run pipeline
            result = self.pipeline.run(dataset_name=dataset_name,df = df)
            
            if result['success']:
                progress(1.0, desc="Processing complete!")
                
                # Store stats
                self.current_stats = result['stats']
                
                # Create summary
                summary = self.create_processing_summary(result)
                
                # Create visualizations
                stats_plot = self.create_stats_visualization(result['stats'])
                
                # Generate file list
                file_list = self.format_file_list(result['files'])
                
                # Add to processing history
                self.processing_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'task': task,
                    'dataset': dataset_name,
                    'stats': result['stats']
                })
                
                return summary, stats_plot, file_list, "✅ Processing completed successfully!"
            else:
                return f"❌ Processing failed: {result['error']}", self.empty_plot(), "", ""
                
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            return f"❌ Error: {str(e)}", self.empty_plot(), "", ""
    
    def create_processing_summary(self, result: Dict[str, Any]) -> str:
        """Create a formatted summary of processing results"""
        stats = result['stats']
        
        summary = "# 📊 Processing Summary\n\n"
        summary += f"**Total Examples:** {stats.get('total_examples', 'N/A')}\n"
        summary += f"**Training Set:** {stats.get('train_size', 'N/A')} examples\n"
        summary += f"**Validation Set:** {stats.get('validation_size', 'N/A')} examples\n\n"
        
        summary += "## 🔧 Processing Statistics\n"
        summary += f"- **Original Size:** {stats.get('original_size', 'N/A')}\n"
        summary += f"- **Cleaned Size:** {stats.get('cleaned_size', 'N/A')}\n"
        summary += f"- **Removed Examples:** {stats.get('removed_count', 'N/A')}\n"
        summary += f"- **Removal Rate:** {stats.get('removal_percentage', 0):.2f}%\n\n"
        
        summary += "## 🎯 Tokenization Info\n"
        summary += f"- **Avg Tokens (Train):** {stats.get('train_avg_tokens', 'N/A'):.1f}\n"
        summary += f"- **Avg Tokens (Val):** {stats.get('validation_avg_tokens', 'N/A'):.1f}\n"
        summary += f"- **Max Length:** {stats.get('max_length', 'N/A')}\n"
        summary += f"- **Vocab Size:** {stats.get('tokenizer_vocab_size', 'N/A')}\n"
        
        return summary
    
    def create_stats_visualization(self, stats: Dict[str, Any]) -> go.Figure:
        """Create visualization of processing statistics"""
        fig = go.Figure()
        
        # Dataset size comparison
        if 'original_size' in stats and 'cleaned_size' in stats:
            fig.add_trace(go.Bar(
                x=['Original', 'After Cleaning'],
                y=[stats['original_size'], stats['cleaned_size']],
                name='Dataset Size',
                marker_color=['#ff7f0e', '#2ca02c']
            ))
        
        fig.update_layout(
            title='Dataset Processing Statistics',
            xaxis_title='Processing Stage',
            yaxis_title='Number of Examples',
            template='plotly_white'
        )
        
        return fig
    
    def format_file_list(self, files: Dict[str, str]) -> str:
        """Format the list of generated files"""
        if not files:
            return "No files generated"
        
        file_list = "# 📁 Generated Files\n\n"
        for file_type, file_path in files.items():
            file_list += f"- **{file_type.replace('_', ' ').title()}:** `{file_path}`\n"
        
        return file_list
    
    def preview_dataset(self, task: str, dataset_name: str) -> pd.DataFrame:
        """Preview a dataset"""
        if not dataset_name:
            return pd.DataFrame()
        
        try:
            temp_pipeline = self.create_pipeline(task,dataset_name)
            df = temp_pipeline.loader.load(dataset_name)
            return df.head(5)  # Return first 5 rows for preview
        except Exception as e:
            logger.error(f"Error previewing dataset: {str(e)}")
            return pd.DataFrame()
    
    def load_config(self, config_file) -> str:
        """Load and display configuration"""
        if config_file is None:
            return "No configuration file uploaded"
        
        try:
            with open(config_file.name, 'r') as f:
                config = yaml.safe_load(f)
            return yaml.dump(config, default_flow_style=False)
        except Exception as e:
            return f"Error loading config: {str(e)}"
    
    def create_interface(self):
        """Create the main Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .output-markdown {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        """
        
        with gr.Blocks(css=css, title="🚀 Fine-tuning Data Pipeline") as interface:
            gr.Markdown("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; margin-bottom: 20px;">
                <h1>🚀 Fine-tuning Data Pipeline</h1>
                <p>Process and prepare datasets for fine-tuning with ease</p>
            </div>
            """)
            
            with gr.Tabs():
                # Main Processing Tab
                with gr.Tab("📊 Dataset Processing"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### ⚙️ Configuration")
                            
                            task_dropdown = gr.Dropdown(
                                choices=["qa", "summarization","classification"],
                                value="qa",
                                label="Select Task",
                                info="Choose the type of task you want to prepare data for"
                            )
                            
                            dataset_source = gr.Radio(
                                choices=["Hugging Face", "Custom Upload"],
                                value="Hugging Face",
                                label="Dataset Source"
                            )
                            
                            
                            custom_file = gr.File(
                                label="Upload Custom Dataset (CSV/JSON)",
                                file_types=[".csv", ".json"],
                                visible=False
                            )
                            
                            

                            with gr.Group(visible=True) as column_mapping_group:
                                gr.Markdown("### 📋 Column Mapping")
                                gr.Markdown("*Specify which columns from the dataset correspond to each field*")

                                Name_column = gr.Textbox(
                                    label="👤 Name Column",
                                    placeholder="e.g., cnn-dailymail",
                                    info="Column containing the dataset name or identifier",
                                    value="Unknown"
                                )
                                instruction_column = gr.Textbox(
                                    label="📝 Instruction Column",
                                    placeholder="e.g., question, prompt, instruction",
                                    info="Column containing the instruction/question"
                                )
                                
                                input_column = gr.Textbox(
                                    label="📥 Input Column", 
                                    placeholder="e.g., context, passage, text",
                                    info="Column containing the input/context"
                                )
                                
                                output_column = gr.Textbox(
                                    label="📤 Output Column",
                                    placeholder="e.g., answer, summary, label", 
                                    info="Column containing the expected output"
                                )

                            process_btn = gr.Button(
                                "🚀 Start Processing",
                                variant="primary"
                            )

                            process_btn.click(
                                fn=update_task_in_config,
                                inputs=[task_dropdown,Name_column,input_column, instruction_column, output_column]
                                )
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### 📊 Results")
                            
                            status_display = gr.Textbox(
                                label="Status",
                                interactive=False,
                                value="Ready to process..."
                            )
                            
                            with gr.Tabs():
                                with gr.Tab("📈 Summary"):
                                    summary_output = gr.Markdown()
                                
                                with gr.Tab("📊 Visualizations"):
                                    stats_plot = gr.Plot(label="Processing Statistics")
                                
                                with gr.Tab("📁 Files"):
                                    files_output = gr.Markdown()
                
                
                # Custom Dataset Validation Tab
                with gr.Tab("✅ Preview Custom Dataset"):
                    gr.Markdown("### 🔍 Preview Your Custom Dataset")
                    
                    with gr.Row():
                        with gr.Column():

                            validation_file = gr.File(
                                label="Upload Dataset",
                                file_types=[".csv", ".json"]
                            )
                            validate_btn = gr.Button("Validate Dataset")
                        
                        with gr.Column():
                            validation_output = gr.Textbox(
                                label="Validation Results",
                                lines=10,
                                interactive=False
                            )
                    
                    validated_preview = gr.Dataframe(
                        label="Dataset Preview",
                        interactive=False
                    )
                
                # Configuration Tab
                with gr.Tab("⚙️ Configuration"):
                    gr.Markdown("### 🔧 Pipeline Configuration")
                    
                    with gr.Row():
                        with gr.Column():
                            config_file = gr.File(
                                label="Upload Configuration (YAML)",
                                file_types=[".yaml", ".yml"]
                            )
                            load_config_btn = gr.Button("Load Configuration")
                        
                        with gr.Column():
                            config_display = gr.Code(
                                label="Current Configuration",
                                language="yaml",
                                interactive=False
                            )
                
                # Processing History Tab
                with gr.Tab("📚 Processing History"):
                    gr.Markdown("### 📋 Processing History")
                    
                    history_output = gr.Dataframe(
                        headers=["Timestamp", "Task", "Dataset", "Total Examples", "Train Size", "Val Size"],
                        label="Processing History",
                        interactive=False
                    )
                    
                    refresh_history_btn = gr.Button("Refresh History")
            
            # Set up event handlers
            def update_ui_visibility(dataset_source):
                """Update UI visibility based on dataset source selection"""
                if dataset_source == "Hugging Face":
                    return (
                        #gr.update(visible=False),   # dataset_dropdown
                        gr.update(visible=False),  # custom_file
                        gr.update(visible=True)    # column_mapping_group
                    )
                else:  # Custom Upload
                    return (
                        #gr.update(visible=False),  # dataset_dropdown
                        gr.update(visible=True),   # custom_file
                        gr.update(visible=True)   # column_mapping_group
                    )

            # Event handlers
            # task_dropdown.change(
            #     fn=self.update_dataset_dropdown,
            #     inputs=[task_dropdown],
            #     outputs=[dataset_dropdown]
            # )
            
            dataset_source.change(
                fn=update_ui_visibility,
                inputs=[dataset_source],
                outputs=[custom_file, column_mapping_group]
            )
            
            process_btn.click(
                fn=self.process_dataset,
                inputs=[task_dropdown, dataset_source, Name_column, custom_file],
                outputs=[summary_output, stats_plot, files_output, status_display]
            )
            
            # # Preview tab events
            # preview_task.change(
            #     fn=self.update_dataset_dropdown,
            #     inputs=[preview_task],
            #     outputs=[preview_dataset]
            # )
            
            # preview_btn.click(
            #     fn=self.preview_dataset,
            #     inputs=[preview_task, preview_dataset],
            #     outputs=[preview_output]
            # )
            
            # Validation tab events
            validate_btn.click(
                fn=lambda f: self.validate_custom_dataset(f)[1:],
                inputs=[validation_file],
                outputs=[validation_output, validated_preview]
            )
            
            # Configuration tab events
            load_config_btn.click(
                fn=self.load_config,
                inputs=[config_file],
                outputs=[config_display]
            )
            
            # History tab events
            refresh_history_btn.click(
                fn=lambda: pd.DataFrame([{
                    'Timestamp': h['timestamp'],
                    'Task': h['task'],
                    'Dataset': h['dataset'],
                    'Total Examples': h['stats'].get('total_examples', 'N/A'),
                    'Train Size': h['stats'].get('train_size', 'N/A'),
                    'Val Size': h['stats'].get('validation_size', 'N/A')
                } for h in self.processing_history]) if self.processing_history else pd.DataFrame(),
                outputs=[history_output]
            )
            
            # Load initial datasets
            # interface.load(
            #     fn=lambda: self.update_dataset_dropdown("qa"),
            #     outputs=[dataset_dropdown]
            # )
            
            # interface.load(
            #     fn=lambda: self.update_dataset_dropdown("qa"),
            #     outputs=[preview_dataset]
            # )
        
        return interface

def main():
    """Main function to launch the Gradio interface"""
    try:
        # Create UI instance
        ui = GradioDataPipelineUI()
        
        # Create and launch interface
        interface = ui.create_interface()
        
        # Launch with custom settings
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
        
    except Exception as e:
        logger.error(f"Error launching interface: {str(e)}")
        print(f"Failed to launch interface: {str(e)}")

if __name__ == "__main__":
    main()