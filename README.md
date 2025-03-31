# Dimensionality Reduction and Visualization of Exoplanet Detection Data

## Project Overview
This project focuses on dimensionality reduction and visualization techniques applied to NASA’s Kepler Mission dataset. The dataset contains astrophysical observations of exoplanet candidates and confirmed planets, with numerous numerical features describing their properties. The primary goal is to analyze and visualize this high-dimensional data using techniques such as:

- **Principal Component Analysis (PCA)**
- **Linear Discriminant Analysis (LDA)**
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
- **Multidimensional Scaling (MDS)**
- **Singular Value Decomposition (SVD)**

Additionally, the project incorporates classification models (RandomForest, K-Nearest Neighbors) to assess the effectiveness of dimensionality reduction techniques.

## Project Structure
```
├── data/                  # Contains the cumulative.csv dataset
├── src/                   # Contains project implementation
│   ├── DR.ipynb           # Main project notebook with full implementation
├── venv/                  # Virtual environment (optional)
├── LICENSE                # MIT License
├── README.md              # Project documentation
├── requirements.txt       # List of required dependencies
```

## Dataset
The dataset used in this project, **cumulative.csv**, contains observations of planetary candidates from the Kepler mission. Key attributes include:
- **Orbital and Transit Features**: Period, duration, depth, and time reference
- **Planetary Properties**: Radius, equilibrium temperature, stellar flux
- **Stellar Properties**: Effective temperature, surface gravity, stellar radius
- **Target Variable**: `koi_disposition` (CONFIRMED, CANDIDATE, FALSE POSITIVE)

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebook `DR.ipynb` using Jupyter Notebook or Jupyter Lab.

## Implementation Details
The project follows these key steps:
1. **Data Preprocessing**: Handling missing values, encoding categorical variables, feature scaling.
2. **Exploratory Data Analysis (EDA)**: Understanding dataset distribution, correlations, and feature importance.
3. **Dimensionality Reduction**:
   - **PCA**: Reducing features while maintaining variance.
   - **LDA**: Supervised feature reduction for classification.
   - **t-SNE & MDS**: Visualizing high-dimensional data in 2D.
   - **SVD**: Decomposing data into singular vectors for efficient processing.
4. **Model Training & Evaluation**:
   - Training **RandomForest** and **KNN** classifiers.
   - Comparing accuracy, precision, recall, and F1-score across different feature reduction techniques.

## Results & Insights
- **PCA and SVD** retain most variance and perform well in classification tasks.
- **LDA** shows the best class separation due to supervision.
- **t-SNE and MDS** are useful for visualization but not for feature extraction.
- **RandomForest** achieves higher accuracy compared to KNN after dimensionality reduction.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
