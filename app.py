#!/usr/bin/env python3
"""
Main application for Model Performance UI
"""
import streamlit as st
import os
import argparse

# Import UI components
from utils.ui_components import display_header, sidebar_model_selection, beaker_job_import_component

# Import data loading functions
from data.loaders import load_model_data, load_domain_data

# Import visualization modules
from visualizations.model_comparison import display_model_comparison
from visualizations.length_analysis import display_length_analysis
from visualizations.domain_analysis import display_domain_analysis

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Model Performance Dashboard")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Default directory containing model prediction files",
        default=None
    )
    return parser.parse_args()

# Set page configuration
st.set_page_config(
    page_title="Model Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Display header
    display_header()
    
    # Set up cache and session state
    if 'input_dir' not in st.session_state:
        # Use command line argument if provided, otherwise use default
        if args.input_dir:
            st.session_state.input_dir = args.input_dir
        else:
            # Set default to look for outputs in parent directory
            default_output_dir = "../outputs"
            if os.path.exists(default_output_dir):
                st.session_state.input_dir = default_output_dir
            else:
                st.session_state.input_dir = "outputs"  # Fallback to current directory
                # Create the directory if it doesn't exist
                os.makedirs(st.session_state.input_dir, exist_ok=True)
    
    # Add a flag to force reload when Beaker data is imported
    if 'force_reload' not in st.session_state:
        st.session_state.force_reload = False
    
    # Input directory selection
    with st.expander("Settings", expanded=False):
        input_dir = st.text_input(
            "Input Directory", 
            value=st.session_state.input_dir,
            help="Directory containing model prediction files"
        )
        
        if input_dir != st.session_state.input_dir:
            st.session_state.input_dir = input_dir
            # Use a button to force cache invalidation
            st.button("Reload Data")
    
    # Beaker integration component in the sidebar
    beaker_imported = beaker_job_import_component(st.session_state.input_dir)
    if beaker_imported:
        st.session_state.force_reload = True
        st.rerun()
    
    # Load data
    with st.spinner("Loading model data..."):
        performance_df, all_models, model_datasets = load_model_data(st.session_state.input_dir)
        # Reset the force_reload flag after loading
        st.session_state.force_reload = False
    
    if performance_df.empty:
        st.error(f"No model prediction files found in '{st.session_state.input_dir}'")
        return
    
    # Create sidebar for model selection
    selected_models, selected_datasets = sidebar_model_selection(all_models, model_datasets)
    
    # Display visualizations
    if not selected_models or not selected_datasets:
        st.warning("Please select at least one model and one dataset.")
    else:
        # Display model comparison
        display_model_comparison(performance_df, selected_models, selected_datasets)
        
        # Display length analysis
        display_length_analysis(performance_df, selected_models, selected_datasets)
        
        # Collect predictions data for the histograms
        predictions_data = []
        for _, row in performance_df[
            (performance_df['model'].isin(selected_models)) & 
            (performance_df['dataset'].isin(selected_datasets))
        ].iterrows():
            predictions_data.append((row['directory'], row))
        
        # Load and display domain analysis
        with st.spinner("Loading domain data..."):
            domain_df, subdomain_df = load_domain_data(selected_models, selected_datasets, performance_df)
        
        # Display domain analysis for all selected datasets, including token length histograms
        display_domain_analysis(domain_df, subdomain_df, selected_datasets, predictions_data)

if __name__ == "__main__":
    main() 