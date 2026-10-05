# msc-dissertation-project

# Sustainable AI: Accuracy vs. Energy Benchmarking Platform

This repository contains the codebase for a M.Sc. Dissertation framework designed to measure the predictive efficacy and environmental cost (carbon emissions) of machine learning models. 

## Prerequisites
- **Python 3.9 - 3.12** - A local machine or VM (CodeCarbon requires local hardware access to read CPU sensors).

## Setup Instructions

### 1. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
`python -m venv venv`
`.\venv\Scripts\Activate.ps1`

### 2. Install Dependencies
`pip install -r requirements.txt`

### 3. Run the Backend API
In your first terminal, start the FastAPI engine:
`uvicorn api:app --reload`
*(Wait for "Application startup complete" message)*

### 4. Run the Frontend Dashboard
Open a **second** terminal, activate the virtual environment again, and run:
`streamlit run app.py`

Upload your CSV via the sidebar to begin benchmarking!
