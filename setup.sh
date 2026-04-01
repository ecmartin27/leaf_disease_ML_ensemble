#!/bin/bash

# --- 1. Define Environment Name ---
ENV_NAME="leaf_disease_env"

echo "------------------------------------------------"
echo "🌱 Initializing Leaf Disease Model Setup..."
echo "------------------------------------------------"

# --- 2. Create Virtual Environment if it doesn't exist ---
if [ ! -d "$ENV_NAME" ]; then
    echo "📦 Creating virtual environment: $ENV_NAME..."
    python3 -m venv $ENV_NAME
else
    echo "✅ Virtual environment already exists."
fi

# --- 3. Activate the Environment ---
# This is critical so that 'pip' and 'python' refer to the local env
echo "🔄 Activating environment..."
source "$ENV_NAME/bin/activate"

# --- 4. Upgrade Pip and Install Dependencies ---
echo "⚙️  Installing/Updating dependencies (this may take a minute)..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# --- 5. Final Launch Check ---
if [ $? -eq 0 ]; then
    echo "🚀 Setup successful! Launching Streamlit..."
    python3 -m streamlit run app.py
else
    echo "❌ Setup failed. Please check your internet connection or requirements.txt."
    exit 1
fi
