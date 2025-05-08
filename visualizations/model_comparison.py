"""
Model comparison visualizations
"""
import streamlit as st
import plotly.express as px

def display_model_comparison(performance_df, selected_models, selected_datasets):
    """Display model comparison visualizations"""
    st.header("Model Performance Comparison")
    
    # Filter data for selected models and datasets
    filtered_df = performance_df[
        (performance_df['model'].isin(selected_models)) & 
        (performance_df['dataset'].isin(selected_datasets))
    ]
    
    if filtered_df.empty:
        st.warning("No data available for the selected models and datasets.")
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Bar Chart", "Heatmap", "Data Table"])
    
    with tab1:
        # Create bar chart with datasets on x-axis
        fig = px.bar(
            filtered_df, 
            x="display_dataset", 
            y="accuracy", 
            color="model", 
            barmode="group",
            title="Model Accuracy by Dataset",
            labels={"display_dataset": "Dataset", "accuracy": "Accuracy", "model": "Model"},
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Create pivot table
        pivot_df = filtered_df.pivot(index="model", columns="display_dataset", values="accuracy")
        
        # Create heatmap
        fig = px.imshow(
            pivot_df,
            text_auto=".3f",
            color_continuous_scale="Blues",
            labels=dict(x="Dataset", y="Model", color="Accuracy"),
            title="Model Performance Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Show data table
        st.dataframe(
            filtered_df[["model", "display_dataset", "accuracy"]].sort_values(
                by=["display_dataset", "model"]
            ),
            use_container_width=True,
            column_config={
                "model": "Model",
                "display_dataset": "Dataset",
                "accuracy": st.column_config.NumberColumn(
                    "Accuracy",
                    format="%.4f",
                )
            }
        ) 