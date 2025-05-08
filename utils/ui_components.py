"""
UI component utilities for the model performance dashboard
"""
import streamlit as st
from data.task_handlers import BaseTaskHandler

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