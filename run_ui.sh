#!/bin/bash
# Run the Model Performance UI

# Set current directory as the working directory
cd "$(dirname "$0")"

# Set PYTHONPATH to include the current directory for imports
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Check for required packages
python -c "
import sys
required_packages = [
    'streamlit', 'plotly', 'pandas', 'numpy', 
    'seaborn', 'datasets', 'transformers', 'matplotlib'
]
missing_packages = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing_packages.append(pkg)
if missing_packages:
    print('Installing missing packages: ' + ', '.join(missing_packages))
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + missing_packages)
else:
    print('All required packages are already installed')
"

# Run the Streamlit app
echo "Starting Model Performance Dashboard..."
streamlit run app.py --server.address=0.0.0.0 --server.port=8501 