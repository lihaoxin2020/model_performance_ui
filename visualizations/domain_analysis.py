"""
Domain analysis visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data.task_handlers import BaseTaskHandler

def display_dataset_domain_analysis(domain_df, subdomain_df, dataset, predictions_data):
    """Display domain analysis for a specific dataset"""
    # Skip if there's no data
    if domain_df.empty and subdomain_df.empty:
        return
        
    # Filter for this dataset
    dataset_domain_df = domain_df[domain_df['dataset'] == dataset] if not domain_df.empty else pd.DataFrame()
    dataset_subdomain_df = subdomain_df[subdomain_df['dataset'] == dataset] if not subdomain_df.empty else pd.DataFrame()
    
    if dataset_domain_df.empty and dataset_subdomain_df.empty:
        return
    
    # Get dataset info and handler
    handler = BaseTaskHandler.get_handler_for_dataset(dataset)
    
    # Get display names for this dataset
    domain_display = handler.get_domain_display_name()
    subdomain_display = handler.get_subdomain_display_name()
    
    # Create dataset-specific header
    from data.loaders import DATASET_INFO
    dataset_display = DATASET_INFO.get(dataset, {}).get("full_name", dataset)
    st.header(f"Domain Analysis - {dataset_display}")
    
    # Create tabs for different visualizations
    tabs = []
    
    # Add tabs based on available data (non-empty dataframes)
    if not dataset_domain_df.empty and len(dataset_domain_df['domain'].unique()) > 0:
        tabs.append(domain_display)
    if not dataset_subdomain_df.empty and len(dataset_subdomain_df['subdomain'].unique()) > 0:
        tabs.append(subdomain_display)
    
    # Always add Token Length and Data Table tabs
    tabs.append("Token Length Distribution")
    tabs.append("Data Table")
    
    # Skip if no domain visualization tabs (only token length and data table)
    if len(tabs) <= 2:
        st.warning(f"No domain data available for {dataset_display}.")
    
    domain_tabs = st.tabs(tabs)
    
    # Display domain plot if available
    tab_index = 0
    if not dataset_domain_df.empty and len(dataset_domain_df['domain'].unique()) > 0:
        with domain_tabs[tab_index]:
            display_domain_comparison(dataset_domain_df, "domain", domain_display)
            display_domain_heatmap(dataset_domain_df, "domain", domain_display)
        tab_index += 1
    
    # Display subdomain plot if available 
    if not dataset_subdomain_df.empty and len(dataset_subdomain_df['subdomain'].unique()) > 0:
        with domain_tabs[tab_index]:
            display_domain_comparison(dataset_subdomain_df, "subdomain", subdomain_display)
            display_domain_heatmap(dataset_subdomain_df, "subdomain", subdomain_display)
        tab_index += 1
    
    # Display token length distribution
    with domain_tabs[tab_index]:
        # Filter predictions for current dataset
        dataset_preds = [(directory, row) for directory, row in predictions_data if row['dataset'] == dataset]
        display_dataset_length_histograms(dataset_preds, dataset)
        
    tab_index += 1
    
    # Display data table
    with domain_tabs[tab_index]:
        # Determine which dataframe to show based on priority and availability
        if not dataset_subdomain_df.empty:
            display_domain_table(dataset_subdomain_df, "subdomain", subdomain_display)
        else:
            display_domain_table(dataset_domain_df, "domain", domain_display)

def display_domain_comparison(analysis_df, domain_col, display_name):
    """Display domain comparison bar chart"""
    # Set up domain selection with Select All button
    unique_domains = sorted(analysis_df[domain_col].unique())
    
    # Create a unique identifier for this specific domain selection
    # Use dataset name and domain column to ensure uniqueness
    dataset_name = analysis_df['dataset'].iloc[0] if not analysis_df.empty else "unknown"
    selection_id = f"{dataset_name}_{domain_col}_{display_name}"
    button_key = f"btn_{selection_id}"
    state_key = f"domains_{selection_id}"
    
    # Create columns for domain selection
    col1, col2 = st.columns([3, 1])
    
    # Initialize session state for domain selection if needed
    if state_key not in st.session_state:
        st.session_state[state_key] = unique_domains[:min(5, len(unique_domains))]
    
    with col1:
        selected_domains = st.multiselect(
            f"Select {display_name} to display",
            options=unique_domains,
            default=st.session_state[state_key],
            key=f"multiselect_{selection_id}"
        )
        
        # Update session state with user selection
        st.session_state[state_key] = selected_domains
    
    with col2:
        # Use a button to select all domains
        if st.button("Select All", key=button_key):
            st.session_state[state_key] = unique_domains
            # Force rerun to update the multiselect with all domains selected
            st.rerun()
    
    if not selected_domains:
        st.warning(f"Please select at least one {display_name.lower()} to display.")
        return
        
    # Filter data for selected domains
    filtered_analysis = analysis_df[analysis_df[domain_col].isin(selected_domains)]
    
    # Create bar chart
    fig = px.bar(
        filtered_analysis,
        x=domain_col,
        y="accuracy",
        color="model",
        barmode="group",
        title=f"Model Accuracy by {display_name}",
        labels={domain_col: display_name, "accuracy": "Accuracy", "model": "Model"},
        height=500,
        hover_data=["total"]
    )
    
    # Customize layout for better readability
    if len(selected_domains) > 5:
        fig.update_layout(
            xaxis_tickangle=-45,
            margin=dict(b=100)
        )
        
    st.plotly_chart(fig, use_container_width=True)

def display_domain_heatmap(analysis_df, domain_col, display_name):
    """Display domain heatmap"""
    # Create pivot table for heatmap
    pivot_df = analysis_df.pivot_table(
        index="model", 
        columns=domain_col, 
        values="accuracy",
        aggfunc="mean"
    )
    
    # Create sample size table for annotation
    sample_df = analysis_df.pivot_table(
        index="model", 
        columns=domain_col, 
        values="total",
        aggfunc="sum"
    )
    
    # Create heatmap
    fig = go.Figure()
    
    # Add heatmap trace
    fig.add_trace(go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="Accuracy"),
        hovertemplate=
        "<b>Model:</b> %{y}<br>" +
        f"<b>{display_name}:</b> %{{x}}<br>" +
        "<b>Accuracy:</b> %{z:.4f}<br>" +
        "<extra></extra>"
    ))
    
    # Add text annotation with accuracy and sample size
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            accuracy = pivot_df.iloc[i, j]
            sample_size = sample_df.iloc[i, j]
            if not pd.isna(accuracy):
                fig.add_annotation(
                    x=pivot_df.columns[j],
                    y=pivot_df.index[i],
                    text=f"{accuracy:.3f}<br>n={int(sample_size)}",
                    showarrow=False,
                    font=dict(color="black" if accuracy < 0.6 else "white", size=10)
                )
    
    fig.update_layout(
        title=f"Model Performance by {display_name}",
        xaxis_title=display_name,
        yaxis_title="Model",
        height=600
    )
    
    # Adjust for many domains
    if len(pivot_df.columns) > 10:
        fig.update_layout(
            xaxis_tickangle=-45,
            margin=dict(b=100)
        )
    
    st.plotly_chart(fig, use_container_width=True)

def display_domain_table(analysis_df, domain_col, display_name):
    """Display domain data table"""
    st.dataframe(
        analysis_df[[
            "model", domain_col, "accuracy", "correct", "total"
        ]].sort_values(by=["model", domain_col]),
        use_container_width=True,
        column_config={
            "model": "Model",
            domain_col: display_name,
            "accuracy": st.column_config.NumberColumn(
                "Accuracy",
                format="%.4f",
            ),
            "correct": "Correct",
            "total": "Total"
        }
    )

def display_dataset_length_histograms(dataset_predictions, dataset_name):
    """Display token length histograms for a specific dataset"""
    if not dataset_predictions:
        st.warning("No prediction data available for token length analysis.")
        return
    
    from data.loaders import DATASET_INFO
    dataset_display = DATASET_INFO.get(dataset_name, {}).get("full_name", dataset_name)
    st.subheader(f"Token Length Distribution for {dataset_display}")
    
    # Get all models from the prediction data
    models = set(row['model'] for _, row in dataset_predictions)
    
    # Create columns for the plots based on number of models
    num_models = len(models)
    if num_models > 3:
        cols = st.columns(3)  # Max 3 columns
    else:
        cols = st.columns(num_models)
    
    # Track which column to use
    col_idx = 0
    
    # For each model, create a histogram
    for model in sorted(models):
        # Get predictions for this model
        model_preds = []
        for directory, row in dataset_predictions:
            if row['model'] == model:
                # Load predictions
                from data.loaders import load_predictions
                predictions = load_predictions(directory)
                model_preds.extend(predictions)
        
        if not model_preds:
            continue
            
        # Process predictions to get token lengths and correctness
        token_lengths = []
        correctness = []
        
        for pred in model_preds:
            # Check if prediction has token length and correctness info
            if "num_tokens" in pred and "metrics" in pred:
                # Check for different possible match field names
                is_correct = False
                if "exact_match" in pred["metrics"]:
                    is_correct = pred["metrics"]["exact_match"] == 1
                elif "exact_match_flex" in pred["metrics"]:
                    is_correct = pred["metrics"]["exact_match_flex"] == 1
                elif "accuracy" in pred["metrics"]:
                    is_correct = pred["metrics"]["accuracy"] == 1
                
                token_lengths.append(pred["num_tokens"])
                correctness.append("Correct" if is_correct else "Incorrect")
        
        if not token_lengths:
            continue
            
        # Create DataFrame for plotting
        import pandas as pd
        import numpy as np
        plot_df = pd.DataFrame({
            "num_tokens": token_lengths,
            "correctness": correctness
        })
        
        # Find the maximum token length (for consistent x-axis)
        max_token = min(np.percentile(token_lengths, 98), 65536)  # Cap at 98th percentile
        
        # Calculate optimal bin size based on data distribution
        q75, q25 = np.percentile(token_lengths, [75, 25])
        bin_width = 2 * (q75 - q25) * len(token_lengths)**(-1/3)  # Freedman-Diaconis rule
        nbins = int(np.ceil((max_token) / max(bin_width, 100)))  # Ensure reasonable number of bins
        nbins = min(max(nbins, 20), 50)  # Between 20 and 50 bins
        
        # Use appropriate column
        with cols[col_idx % len(cols)]:
            # Create figure using Plotly
            import plotly.graph_objects as go
            fig = go.Figure()
            
            # Add histograms for correct and incorrect answers
            colors = {"Correct": "rgba(65, 105, 225, 0.7)", "Incorrect": "rgba(220, 20, 60, 0.7)"}
            for label, color in colors.items():
                subset = plot_df[plot_df["correctness"] == label]
                if not subset.empty:
                    fig.add_trace(go.Histogram(
                        x=subset["num_tokens"],
                        name=label,
                        marker_color=color,
                        opacity=0.7,
                        histnorm='probability density',
                        xbins=dict(
                            start=0,
                            end=max_token,
                            size=max_token / nbins
                        )
                    ))
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=model,
                    font=dict(size=16)
                ),
                xaxis_title=f"Token Count (capped at {int(max_token)})",
                yaxis_title="Relative frequency",
                barmode='overlay',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=450,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Show sample size info
            correct_count = plot_df[plot_df["correctness"] == "Correct"].shape[0]
            incorrect_count = plot_df[plot_df["correctness"] == "Incorrect"].shape[0]
            total_count = correct_count + incorrect_count
            st.caption(f"Sample size: {correct_count} correct ({correct_count/total_count:.1%}), {incorrect_count} incorrect ({incorrect_count/total_count:.1%})")
        
        # Move to next column
        col_idx += 1

def display_domain_analysis(domain_df, subdomain_df, selected_datasets, predictions_data):
    """Display domain analysis visualizations for all selected datasets"""
    if domain_df.empty and subdomain_df.empty and not predictions_data:
        st.warning("No domain data available for the selected models and datasets.")
        return
        
    # Create a separate analysis section for each dataset
    for dataset in selected_datasets:
        display_dataset_domain_analysis(domain_df, subdomain_df, dataset, predictions_data) 