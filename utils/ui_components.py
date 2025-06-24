"""
UI component utilities for the model performance dashboard
"""
import streamlit as st
import pandas as pd
from data.task_handlers import BaseTaskHandler
from utils.beaker_integration import (
    check_beaker_available, 
    download_experiment_results, 
    list_workspace_experiments,
    get_workspace_info
)
from utils.data_processing import (
    import_beaker_job_results,
    get_cache_info,
    clear_cache
)

def display_header():
    """Display the dashboard header"""
    st.title("📊 Model Performance Dashboard")
    st.markdown("""
    Interactive dashboard for visualizing and comparing performance of language models across different datasets and domains.
    Select models from the sidebar to analyze their performance.
    """)

def plot_selection_component(all_models, selected_models, selected_datasets):
    """Enhanced model selection component for plotting"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Plot Configuration")
    
    # Initialize session state for plotting selection
    if "plot_selected_models" not in st.session_state:
        st.session_state["plot_selected_models"] = selected_models[:min(3, len(selected_models))]
    
    # Model selection for plotting
    plot_models = st.sidebar.multiselect(
        "Models to Plot",
        options=selected_models,
        default=st.session_state["plot_selected_models"],
        help="Select specific models to include in visualizations",
        key="plot_model_selector"
    )
    
    # Update session state
    st.session_state["plot_selected_models"] = plot_models
    
    # Quick selection buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Plot All", key="plot_all_models"):
            st.session_state["plot_selected_models"] = selected_models
            st.rerun()
    
    with col2:
        if st.button("Clear Plots", key="clear_plot_models"):
            st.session_state["plot_selected_models"] = []
            st.rerun()
    
    return plot_models

def cache_management_component():
    """Cache management component"""
    st.sidebar.markdown("---")
    with st.sidebar.expander("🗂️ Cache Management", expanded=False):
        cache_info = get_cache_info()
        
        if cache_info["file_count"] > 0:
            st.write(f"**Cache Files:** {cache_info['file_count']}")
            st.write(f"**Cache Size:** {cache_info['size_mb']:.2f} MB")
            st.write(f"**Cache Types:** {', '.join(cache_info['cache_types'])}")
            
            if st.button("Clear All Cache", help="Clear all cached data"):
                if clear_cache():
                    st.success("Cache cleared successfully!")
                    st.rerun()
                else:
                    st.error("Failed to clear cache")
        else:
            st.info("No cached data")

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
    
    # Add plot selection component
    plot_models = plot_selection_component(all_models, selected_models, selected_datasets)
    
    # Add cache management component
    cache_management_component()
    
    # Add documentation in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info("""
    This dashboard visualizes model performance across datasets and domains.
    
    **How to use:**
    1. Select models to compare
    2. Choose datasets to analyze
    3. Select models to plot
    4. Explore the visualizations
    
    **Data source:** Model prediction files in the outputs directory.
    """)
    
    return selected_models, selected_datasets, plot_models

def beaker_job_import_component(input_dir):
    """
    Simple Beaker status component for sidebar
    
    Args:
        input_dir (str): The current input directory for model results
        
    Returns:
        bool: Always False since main import is handled in main area
    """
    st.sidebar.markdown("---")
    with st.sidebar.expander("💻 Beaker Status", expanded=False):
        # Check if Beaker client is available
        beaker_available, beaker_status = check_beaker_available()
        
        if not beaker_available:
            st.warning(f"⚠️ {beaker_status}")
            st.info("To enable Beaker integration, install beaker-py with `pip install beaker-py` and configure your Beaker credentials.")
        else:
            st.success("✅ Beaker client ready")
            
            # Show current workspace info
            success, message, workspace_name = get_workspace_info()
            if success:
                st.info(f"**Current workspace:** {workspace_name}")
            
            st.info("Use the 'Import From Beaker' section in the main area to browse and import experiments.")
    
    return False

def beaker_import_main_component(input_dir):
    """
    Main area component for importing Beaker experiments with intelligent caching, pagination, and search
    
    Args:
        input_dir (str): The current input directory for model results
        
    Returns:
        bool: True if data was imported and UI should refresh
    """
    st.header("🔬 Import From Beaker")
    
    # Check if Beaker client is available
    beaker_available, beaker_status = check_beaker_available()
    
    if not beaker_available:
        st.error(f"⚠️ {beaker_status}")
        st.info("To enable Beaker integration, install beaker-py with `pip install beaker-py` and configure your Beaker credentials.")
        return False
    
    # Import the new caching functions
    from utils.beaker_integration import get_paginated_experiments, clear_experiments_cache
    
    # Show current workspace info
    success, message, workspace_name = get_workspace_info()
    if success:
        st.info(f"**Current workspace:** {workspace_name}")
    
    # Create tabs for different import methods
    tab1, tab2 = st.tabs(["📋 Browse Experiments", "🔗 Direct Import"])
    
    with tab1:
        st.markdown("### Browse Workspace Experiments")
        
        # Configuration section
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Workspace input (optional)
            custom_workspace = st.text_input(
                "Workspace (optional)",
                help="Leave empty to use default workspace",
                key="workspace_input"
            )
        
        with col2:
            # Refresh button to check for new experiments
            if st.button("🔄 Check New", key="check_new_experiments", help="Check for new experiments since last load"):
                workspace_to_use = custom_workspace if custom_workspace.strip() else None
                
                with st.spinner("Checking for new experiments..."):
                    success, message, page_experiments, has_more, total_count = get_paginated_experiments(
                        workspace_to_use, 
                        page=st.session_state.get("experiment_page", 0), 
                        page_size=50, 
                        force_refresh=False
                    )
                
                if success:
                    st.session_state["current_page_experiments"] = page_experiments
                    st.session_state["has_more_pages"] = has_more
                    st.session_state["total_experiment_count"] = total_count
                    st.session_state["last_workspace"] = workspace_to_use
                    if "New experiments detected" in message:
                        st.success(f"🎉 {message}")
                    else:
                        st.info(message)
                else:
                    st.error(message)
        
        with col3:
            # Reset button to clear all state
            if st.button("🔄 Reset", key="reset_experiments", help="Clear all cached experiments and start fresh"):
                workspace_to_use = custom_workspace if custom_workspace.strip() else None
                clear_experiments_cache(workspace_to_use)
                
                # Clear all experiment-related session state
                keys_to_clear = ["current_page_experiments", "experiment_page", "selected_experiments", 
                               "experiment_search", "has_more_pages", "total_experiment_count", "last_workspace"]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Cache cleared! Click 'Load Experiments' to start fresh.")
        
        # Search bar
        search_query = st.text_input(
            "🔍 Search experiments by name",
            help="Enter text to filter experiments by name (case-insensitive)",
            key="experiment_search",
            value=st.session_state.get("experiment_search", "")
        )
        
        # Initialize pagination state
        if "experiment_page" not in st.session_state:
            st.session_state["experiment_page"] = 0
        
        # Load experiments section
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Initial load or force refresh button
            button_text = "🔄 Load Experiments" if "current_page_experiments" not in st.session_state else "🔄 Refresh All"
            button_help = "Load experiments from workspace" if "current_page_experiments" not in st.session_state else "Force refresh all experiments from scratch"
            
            if st.button(button_text, key="load_experiments", type="primary", help=button_help):
                workspace_to_use = custom_workspace if custom_workspace.strip() else None
                
                # Reset to first page when loading new experiments
                st.session_state["experiment_page"] = 0
                
                # Force refresh if this is a refresh operation
                force_refresh = "current_page_experiments" in st.session_state
                
                with st.spinner("Loading experiments..." if not force_refresh else "Refreshing all experiments..."):
                    success, message, page_experiments, has_more, total_count = get_paginated_experiments(
                        workspace_to_use, 
                        page=0, 
                        page_size=50, 
                        force_refresh=force_refresh
                    )
                
                if success:
                    st.session_state["current_page_experiments"] = page_experiments
                    st.session_state["has_more_pages"] = has_more
                    st.session_state["total_experiment_count"] = total_count
                    st.session_state["last_workspace"] = workspace_to_use
                    st.success(message)
                else:
                    st.error(message)
        
        with col2:
            # Previous page button
            if st.session_state.get("experiment_page", 0) > 0:
                if st.button("⬅️ Previous", key="prev_page"):
                    st.session_state["experiment_page"] -= 1
                    
                    # Load previous page from cache
                    workspace_to_use = st.session_state.get("last_workspace")
                    with st.spinner("Loading previous page..."):
                        success, message, page_experiments, has_more, total_count = get_paginated_experiments(
                            workspace_to_use, 
                            page=st.session_state["experiment_page"], 
                            page_size=50, 
                            force_refresh=False
                        )
                    
                    if success:
                        st.session_state["current_page_experiments"] = page_experiments
                        st.session_state["has_more_pages"] = has_more
                        st.session_state["total_experiment_count"] = total_count
                        st.info(message)
                    else:
                        st.session_state["experiment_page"] += 1  # Revert on error
                        st.error(message)
            else:
                st.button("⬅️ Previous", key="prev_page_disabled", disabled=True)
        
        with col3:
            # Next page button  
            has_more = st.session_state.get("has_more_pages", False)
            if has_more:
                if st.button("➡️ Next", key="next_page"):
                    st.session_state["experiment_page"] += 1
                    
                    # Load next page from cache
                    workspace_to_use = st.session_state.get("last_workspace")
                    with st.spinner("Loading next page..."):
                        success, message, page_experiments, has_more, total_count = get_paginated_experiments(
                            workspace_to_use, 
                            page=st.session_state["experiment_page"], 
                            page_size=50, 
                            force_refresh=False
                        )
                    
                    if success:
                        st.session_state["current_page_experiments"] = page_experiments
                        st.session_state["has_more_pages"] = has_more
                        st.session_state["total_experiment_count"] = total_count
                        st.info(message)
                    else:
                        st.session_state["experiment_page"] -= 1  # Revert on error
                        st.error(message)
            else:
                st.button("➡️ Next", key="next_page_disabled", disabled=True)
        
        # Show current page info and cache status
        if "current_page_experiments" in st.session_state:
            total_count = st.session_state.get("total_experiment_count", 0)
            current_page = st.session_state.get("experiment_page", 0) + 1
            page_size = len(st.session_state.get("current_page_experiments", []))
            
            st.caption(f"📄 Page {current_page} ({page_size} experiments) | 💾 Total cached: {total_count} experiments")
        
        # Display experiments if loaded
        if "current_page_experiments" in st.session_state and st.session_state["current_page_experiments"]:
            experiments = st.session_state["current_page_experiments"]
            
            # Apply search filter
            if search_query.strip():
                filtered_by_search = [
                    exp for exp in experiments 
                    if search_query.lower() in exp["name"].lower()
                ]
            else:
                filtered_by_search = experiments
            
            st.markdown(f"### Available Experiments ({len(filtered_by_search)}{f'/{len(experiments)} after search' if search_query.strip() else ''})")
            
            if search_query.strip() and not filtered_by_search:
                st.warning(f"No experiments found matching '{search_query}'. Try a different search term.")
                filtered_by_search = []
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                show_only_finalized = st.checkbox("Show only finalized", value=True, key="filter_finalized")
            with col2:
                show_only_with_results = st.checkbox("Show only with results", value=True, key="filter_results")
            with col3:
                select_all = st.checkbox("Select all visible", key="select_all_experiments")
            
            # Apply status filters
            filtered_experiments = filtered_by_search
            if show_only_finalized:
                filtered_experiments = [exp for exp in filtered_experiments if exp["finalized"]]
            if show_only_with_results:
                filtered_experiments = [exp for exp in filtered_experiments if exp["has_results"]]
            
            if not filtered_experiments:
                if search_query.strip():
                    st.warning("No experiments match the current search and filters.")
                else:
                    st.warning("No experiments match the current filters.")
            else:
                # Initialize session state for experiment selection
                if "selected_experiments" not in st.session_state:
                    st.session_state["selected_experiments"] = set()
                
                # Handle select all toggle
                if select_all:
                    st.session_state["selected_experiments"].update({exp["id"] for exp in filtered_experiments})
                elif "select_all_experiments" in st.session_state and not select_all:
                    # Only clear the ones that are currently visible
                    visible_ids = {exp["id"] for exp in filtered_experiments}
                    st.session_state["selected_experiments"] -= visible_ids
                
                # Display experiments as a list with checkboxes
                st.markdown("#### Select Experiments to Download:")
                
                selected_for_download = []
                
                # Create a container for the experiment list
                with st.container():
                    for i, exp in enumerate(filtered_experiments):
                        col1, col2 = st.columns([1, 10])
                        
                        with col1:
                            # Checkbox for selection
                            is_selected = st.checkbox(
                                "",
                                value=exp["id"] in st.session_state["selected_experiments"],
                                key=f"exp_checkbox_{exp['id']}",
                                label_visibility="collapsed"
                            )
                            
                            if is_selected:
                                st.session_state["selected_experiments"].add(exp["id"])
                                selected_for_download.append(exp)
                            else:
                                st.session_state["selected_experiments"].discard(exp["id"])
                        
                        with col2:
                            # Experiment details
                            status_icon = "✅" if exp["finalized"] else "⏳"
                            results_icon = "📊" if exp["has_results"] else "❌"
                            
                            # Highlight search terms
                            display_name = exp['name']
                            if search_query.strip():
                                # Simple highlighting by making search term bold
                                import re
                                pattern = re.compile(re.escape(search_query), re.IGNORECASE)
                                display_name = pattern.sub(f"**{search_query}**", display_name)
                            
                            st.markdown(f"""
                            **{display_name}** {status_icon} {results_icon}
                            - **ID:** `{exp['id']}`
                            - **Author:** {exp['author']} | **Created:** {exp['created']}
                            - **Description:** {exp['description'] if exp['description'] else 'No description'}
                            """)
                        
                        # Add separator between experiments
                        if i < len(filtered_experiments) - 1:
                            st.markdown("---")
                
                # Download section
                if st.session_state["selected_experiments"]:
                    st.markdown(f"### Download Selected Experiments ({len(st.session_state['selected_experiments'])})")
                    
                    # Model name override for all selected
                    model_name_prefix = st.text_input(
                        "Model Name Prefix (optional)",
                        help="Prefix for model names. If empty, will try to extract from experiment results. For multiple experiments, each will get a unique suffix.",
                        key="browse_model_name_prefix"
                    )
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("📥 Download Selected Experiments", key="download_selected_bulk", type="primary"):
                            # Get all selected experiments from cache (might span multiple pages)
                            all_selected_experiments = []
                            for exp_id in st.session_state["selected_experiments"]:
                                # Find experiment details (might be from current page or other cached pages)
                                exp_details = next((exp for exp in filtered_experiments if exp["id"] == exp_id), None)
                                if exp_details:
                                    all_selected_experiments.append(exp_details)
                                else:
                                    # Create minimal experiment info for experiments from other pages
                                    all_selected_experiments.append({"id": exp_id, "name": exp_id})
                            
                            return download_selected_experiments(
                                st.session_state["selected_experiments"], 
                                all_selected_experiments, 
                                input_dir, 
                                model_name_prefix
                            )
                    
                    with col2:
                        if st.button("🗑️ Clear Selection", key="clear_selection"):
                            st.session_state["selected_experiments"] = set()
                            st.rerun()
                else:
                    st.info("Select one or more experiments to download.")
        else:
            st.info("Click 'Load Experiments' to browse available experiments.")
    
    with tab2:
        st.markdown("### Direct Experiment Import")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Input for Beaker job ID
            job_id = st.text_input(
                "Beaker Experiment ID", 
                help="Enter the Beaker experiment ID to download evaluation results",
                key="direct_job_id"
            )
        
        with col2:
            # Optional model name override
            model_name = st.text_input(
                "Model Name (optional)",
                help="Custom name for the model. If empty, will try to extract from the job results.",
                key="direct_model_name"
            )
        
        # Download button
        if st.button("📥 Download & Import", key="download_direct", type="primary"):
            if not job_id:
                st.error("Please enter a Beaker experiment ID")
                return False
            
            return download_and_import_experiment(job_id, input_dir, model_name)
    
    # Help information
    with st.expander("ℹ️ Help & Requirements", expanded=False):
        st.info("""
        **Requirements for Beaker experiments:**
        
        1. The experiment must have a result dataset
        2. The result dataset should contain:
           - `metrics.json` file
           - Prediction files named like `task-*-dataset-predictions.jsonl`
           
        **Smart Caching System:**
        
        - **Intelligent Loading**: Only fetches new experiments since last load
        - **Fast Pagination**: Pages are served from cache for instant navigation
        - **New Experiment Detection**: Automatically detects and highlights new experiments
        - **Cache Management**: Use "Check New" to update cache or "Reset" to clear and start fresh
           
        **Navigation:**
        
        - **Load Experiments**: Initial load or force refresh all experiments
        - **Check New**: Check for new experiments since last load (incremental update)
        - **Reset**: Clear all cached data and start fresh
        - **Pagination**: Navigate cached experiments 50 per page
        - **Search**: Filter experiments by name (works on current page)
           
        **Filtering Options:**
        
        - **Finalized**: Only show completed experiments
        - **With Results**: Only show experiments that have result datasets
        - **Select All**: Quickly select/deselect all visible experiments
        
        **Troubleshooting:**
        
        - Make sure your Beaker credentials are configured
        - Check that the experiment ID is correct
        - Verify the experiment has completed and has results
        - For API errors, ensure you have the latest beaker-py version
        """)
    
    return False

def download_selected_experiments(selected_ids, filtered_experiments, input_dir, model_name_prefix):
    """
    Download multiple selected experiments
    
    Args:
        selected_ids (set): Set of selected experiment IDs
        filtered_experiments (list): List of all filtered experiments
        input_dir (str): Input directory for results
        model_name_prefix (str): Prefix for model names
    
    Returns:
        bool: True if any download was successful
    """
    selected_experiments = [exp for exp in filtered_experiments if exp["id"] in selected_ids]
    
    if not selected_experiments:
        st.error("No experiments selected for download")
        return False
    
    success_count = 0
    total_count = len(selected_experiments)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, exp in enumerate(selected_experiments):
        # Update progress
        progress = (i + 1) / total_count
        progress_bar.progress(progress)
        status_text.text(f"Downloading experiment {i+1}/{total_count}: {exp['name']}")
        
        # Generate model name
        if model_name_prefix:
            if total_count > 1:
                model_name = f"{model_name_prefix}_{i+1}"
            else:
                model_name = model_name_prefix
        else:
            model_name = None
        
        # Download experiment
        success = download_and_import_experiment(exp["id"], input_dir, model_name, silent=True)
        if success:
            success_count += 1
    
    # Final status
    progress_bar.progress(1.0)
    if success_count == total_count:
        status_text.success(f"✅ Successfully downloaded all {total_count} experiments!")
        # Clear selection after successful download
        st.session_state["selected_experiments"] = set()
        return True
    elif success_count > 0:
        status_text.warning(f"⚠️ Downloaded {success_count}/{total_count} experiments successfully")
        return True
    else:
        status_text.error(f"❌ Failed to download any experiments")
        return False

def download_and_import_experiment(experiment_id: str, input_dir: str, model_name: str = None, silent: bool = False) -> bool:
    """
    Helper function to download and import a Beaker experiment
    
    Args:
        experiment_id (str): The Beaker experiment ID
        input_dir (str): The input directory for importing results
        model_name (str, optional): Custom model name
    
    Returns:
        bool: True if successful
    """
    # Show progress
    progress_text = st.empty()
    progress_text.info(f"Downloading results for experiment {experiment_id}...")
    
    # Download the experiment results
    success, message, download_path = download_experiment_results(experiment_id)
    
    if not success:
        # Add more context to the error message for common problems
        if "Experiment with ID" in message and "not found" in message:
            error_message = f"Error: {message}. Please verify the experiment ID is correct."
        elif "No result dataset found" in message:
            error_message = f"Error: {message}. The experiment must have a completed result dataset."
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
    progress_text.success(f"Successfully imported experiment results: {import_message}")
    
    # Return True to trigger a reload
    return True 