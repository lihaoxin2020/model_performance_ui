"""
UI component utilities for the model performance dashboard
"""
import streamlit as st
from data.task_handlers import BaseTaskHandler
from utils.beaker_integration import check_beaker_available, download_job_results
from utils.data_processing import import_beaker_job_results

def display_header():
    """Display the dashboard header"""
    st.title("📊 Model Performance Dashboard")
    st.markdown("""
    Interactive dashboard for visualizing and comparing performance of language models across different datasets and domains.
    Select models from the sidebar to analyze their performance.
    """)

def sidebar_model_selection(all_models, model_datasets):
    """Create sidebar for model selection"""
    st.sidebar.title("Model Selection")
    
    # Initialize session state for models if needed
    if "model_selection" not in st.session_state:
        st.session_state["model_selection"] = all_models[:min(3, len(all_models))]
    
    # Model selection with Select All button
    st.sidebar.subheader("Models")
    
    # Create columns for model selection
    col1, col2 = st.sidebar.columns([3, 1])
    
    with col1:
        selected_models = st.multiselect(
            "Select Models to Compare",
            options=all_models,
            default=st.session_state["model_selection"],
            key="multiselect_models"
        )
        # Update session state with current selection
        st.session_state["model_selection"] = selected_models
    
    with col2:
        if st.button("Select All", key="btn_select_all_models"):
            st.session_state["model_selection"] = all_models
            st.rerun()
    
    # Get available datasets for selected models
    available_datasets = set()
    for model in selected_models:
        available_datasets.update(model_datasets[model])
    
    # Initialize session state for datasets if needed
    available_datasets_list = sorted(list(available_datasets))
    if "dataset_selection" not in st.session_state or not set(st.session_state["dataset_selection"]).issubset(available_datasets):
        st.session_state["dataset_selection"] = available_datasets_list[:min(2, len(available_datasets_list))]
    
    # Dataset selection with Select All button
    st.sidebar.subheader("Datasets")
    
    # Create columns for dataset selection
    col1, col2 = st.sidebar.columns([3, 1])
    
    with col1:
        selected_datasets = st.multiselect(
            "Select Datasets",
            options=available_datasets_list,
            default=st.session_state["dataset_selection"],
            key="multiselect_datasets"
        )
        # Update session state with current selection
        st.session_state["dataset_selection"] = selected_datasets
    
    with col2:
        if st.button("Select All", key="btn_select_all_datasets"):
            st.session_state["dataset_selection"] = available_datasets_list
            st.rerun()
    
    # Add documentation in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info("""
    This dashboard visualizes model performance across datasets and domains.
    
    **How to use:**
    1. Select models to compare
    2. Choose datasets to analyze
    3. Explore the visualizations
    
    **Data source:** Model prediction files in the outputs directory.
    """)
    
    return selected_models, selected_datasets

def beaker_job_import_component(input_dir):
    """
    Component for importing Beaker job results
    
    Args:
        input_dir (str): The current input directory for model results
        
    Returns:
        bool: True if data was imported and UI should refresh
    """
    st.sidebar.markdown("---")
    with st.sidebar.expander("💻 Import From Beaker", expanded=False):
        # Check if Beaker client is available
        beaker_available, beaker_status = check_beaker_available()
        
        if not beaker_available:
            st.warning(f"⚠️ {beaker_status}")
            st.info("To enable Beaker integration, install beaker-py with `pip install beaker-py` and configure your Beaker credentials.")
            return False
        
        st.success("✅ Beaker client ready")
        
        # Input for Beaker job ID
        job_id = st.text_input(
            "Beaker Job ID", 
            help="Enter the Beaker job ID to download evaluation results"
        )
        
        # Optional model name override
        model_name = st.text_input(
            "Model Name (optional)",
            help="Custom name for the model. If empty, will try to extract from the job results."
        )
        
        # Help information - using a help icon and tooltip instead of nested expander
        st.markdown("### ℹ️ Help")
        
        # Display help information directly without an expander
        st.info("""
        **Requirements for Beaker jobs:**
        
        1. The job must have a result dataset
        2. The result dataset should contain:
           - `metrics.json` file
           - Prediction files named like `task-*-dataset-predictions.jsonl`
           
        **Troubleshooting:**
        
        - Make sure your Beaker credentials are configured
        - Check that the job ID is correct
        - Verify the job has completed and has results
        - For API errors, ensure you have the latest beaker-py version
        """)
        
        # Download button
        if st.button("Download & Import Job Results"):
            if not job_id:
                st.error("Please enter a Beaker job ID")
                return False
            
            # Show progress
            progress_text = st.empty()
            progress_text.info(f"Downloading results for job {job_id}...")
            
            # Download the job results
            success, message, download_path = download_job_results(job_id)
            
            if not success:
                # Add more context to the error message for common problems
                if "Job with ID" in message and "not found" in message:
                    error_message = f"Error: {message}. Please verify the job ID is correct."
                elif "No result dataset found" in message:
                    error_message = f"Error: {message}. The job must have a completed result dataset."
                elif "dataset.stream_file" in message or "AttributeError" in message:
                    error_message = (
                        f"Error: {message}. "
                        "This may be due to an incompatible beaker-py version. "
                        "Try upgrading with 'pip install -U beaker-py'."
                    )
                else:
                    error_message = f"Error: {message}"
                
                progress_text.error(error_message)
                return False
            
            # Update progress
            progress_text.info(f"Processing downloaded results...")
            
            # Import the downloaded results to the expected directory structure
            import_success, import_message, import_path = import_beaker_job_results(
                download_path, 
                input_dir,
                model_name=model_name if model_name else None
            )
            
            if not import_success:
                progress_text.error(f"Import error: {import_message}")
                return False
            
            # Show success message
            progress_text.success(f"Successfully imported job results: {import_message}")
            
            # Return True to trigger a reload
            return True
        
    return False 