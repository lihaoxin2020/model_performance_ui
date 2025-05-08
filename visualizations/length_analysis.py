"""
Output length analysis visualizations
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def display_length_analysis(performance_df, selected_models, selected_datasets):
    """Display output length analysis visualizations"""
    st.header("Output Length Analysis")
    
    # Filter data for selected models and datasets
    filtered_df = performance_df[
        (performance_df['model'].isin(selected_models)) & 
        (performance_df['dataset'].isin(selected_datasets))
    ]
    
    if filtered_df.empty:
        st.warning("No data available for the selected models and datasets.")
        return
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Length Distribution", "Length vs Accuracy"])
    
    with tab1:
        # Create bar chart with datasets on x-axis and models as bars
        fig = px.bar(
            filtered_df,
            x="display_dataset",
            y="avg_tokens",
            color="model",
            barmode="group",
            title="Average Output Length by Dataset and Model",
            labels={
                "display_dataset": "Dataset",
                "avg_tokens": "Average Number of Tokens",
                "model": "Model"
            },
            height=500
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title="Dataset",
            yaxis_title="Average Number of Tokens"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Create scatter plot of length vs accuracy
        fig = px.scatter(
            filtered_df,
            x="avg_tokens",
            y="accuracy",
            color="model",
            symbol="display_dataset",
            title="Output Length vs Accuracy",
            labels={
                "avg_tokens": "Average Number of Tokens",
                "accuracy": "Accuracy",
                "model": "Model",
                "display_dataset": "Dataset"
            },
            height=500
        )
        
        # Add trendline
        fig.update_traces(marker=dict(size=12))
        
        st.plotly_chart(fig, use_container_width=True)

def display_length_distribution_histograms(predictions_data, selected_models, selected_datasets):
    """Display histograms showing token length distribution with correct/incorrect coloring using Plotly"""
    st.header("Token Length Distribution by Correctness")
    
    # Create columns for the plots based on number of models
    num_models = len(selected_models)
    cols = st.columns(min(3, num_models))  # Max 3 columns
    
    # Track which column to use
    col_idx = 0
    
    # For each model, create a histogram
    for model in selected_models:
        # Get predictions for this model
        model_preds = []
        for dataset in selected_datasets:
            # Find the directory for this model and dataset
            directories = [d for _, d in predictions_data if d["model"] == model and d["dataset"] == dataset]
            
            # Process each directory
            for directory_info in directories:
                directory = directory_info["directory"]
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
                
                token_lengths.append(pred["num_tokens"])
                correctness.append("Correct" if is_correct else "Incorrect")
        
        if not token_lengths:
            continue
            
        # Create DataFrame for plotting
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