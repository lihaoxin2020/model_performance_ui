# Model Performance Analysis UI

An interactive dashboard for analyzing and comparing the performance of large language models across multiple datasets and evaluation tasks.

## Features

- **Interactive Model Comparison**: Compare performance of multiple models side-by-side
- **Output Length Analysis**: Visualize and analyze token length distributions
- **Domain-specific Performance**: Explore model strengths and weaknesses across different domains
- **Detailed Token Distribution Visualization**: See how token length correlates with correctness
- **Data Tables**: Access the raw data for detailed analysis

## Running the Dashboard

Simply run the provided script:

```bash
./run_ui.sh
```

This will:
1. Check for and install any required dependencies
2. Start the Streamlit server (default port: 8501)
3. Open the dashboard in your web browser

## Directory Structure

```
model_performance_ui/
├── app.py                # Main application entry point
├── run_ui.sh             # Script to run the application
├── setup.py              # Dependencies list
├── data/                 # Data loading and processing modules
├── visualizations/       # Visualization components
└── utils/                # Utility functions
```

## Usage

In the dashboard:

1. Use the sidebar to select models to compare
2. Choose datasets to analyze
3. Select domain analysis type (high-level or subdomains)
4. Navigate between different visualizations using the tabs

## Input Data Format

The system expects model prediction files in the following directory structure:

```
outputs/
└── lmeval-<model_name>-on-<dataset_name>-<hash>/
    ├── metrics.json
    └── task-000-<dataset_name>-predictions.jsonl
```

By default, the dashboard looks for an "outputs" directory at the same level as the model_performance_ui directory. You can customize this location in the Settings panel.

## Adding New Datasets

To add support for new datasets, modify the `get_dataset_domains` function in `data/loaders.py` with appropriate domain extraction logic for your dataset. 