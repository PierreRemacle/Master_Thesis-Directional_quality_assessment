# Master Thesis Dashboard: Path-Based Quality Metrics Visualization

## Overview
This Plotly Dash application is a comprehensive research tool designed to analyze and visualize path-based quality metrics for dimensionality reduction techniques. The dashboard provides an interactive platform for researchers to explore and compare different dimensionality reduction methods.

## Project Structure
- `src/app.py`: Main application entry point
- `PCA_MNIST_data/`: Example dataset folder
  - Contains pre-processed MNIST dataset for demonstration
  - Provides sample data for testing and visualization
- `master_thesis.pdf`: Full master's thesis document
  - Provides comprehensive background and detailed research findings

## Features
- Interactive dashboard with 6 specialized tabs
  - 5 analysis tabs
  - 6th tab containing the full master's thesis PDF
- Support for multiple dimensionality reduction techniques:
  - Isomap
  - t-SNE
  - UMAP
  - Multidimensional Scaling (MDS)
  - Principal Component Analysis (PCA)
- Custom path-based quality metric analysis
- Dynamic data upload and processing
- Comparative metric visualization
- Direct access to research documentation

## Running the Application
1. Ensure you have the required dependencies installed
2. Navigate to the project directory
3. Run the application:
   ```bash
   python src/app.py
   ```
4. Open a web browser and go to http://127.0.0.1:8050/

## Data Example
The `PCA_MNIST_data/` folder includes a pre-processed MNIST dataset, providing:
- A ready-to-use example for testing the dashboard
- Demonstration of data processing capabilities
- Benchmark dataset for dimensionality reduction techniques

## Requirements
- Python 3.8+
- Plotly Dash
- Numpy
- Scikit-learn
- Pandas

## Documentation
- Interactive dashboard provides detailed insights
- Full master's thesis available directly within the application's 6th tab
- Comprehensive explanation of research methodology and findings
