# Dimensionality Reduction and Clustering of Exoplanet Detection Data

## Project Overview
This project focuses on applying **dimensionality reduction** and **clustering** techniques to data from **NASA’s Kepler Mission**. The dataset includes astrophysical observations of planetary candidates and confirmed exoplanets.

The main objectives are:
- Visualizing high-dimensional exoplanet data using techniques like:
  - **PCA**, **LDA**, **t-SNE**, **MDS**, **SVD**
- Discovering natural groupings among exoplanetary systems using clustering algorithms:
  - **KMeans**, **DBSCAN**, **Agglomerative Clustering**, **Birch**
- Evaluating clustering quality using metrics like **Silhouette Score**, **Calinski-Harabasz Index**, and **Davies-Bouldin Score**
- Assessing classification performance after dimensionality reduction using **RandomForest** and **KNN**

---

## Project Structure
```
├── data/                  # Contains the cumulative.csv dataset
├── src/                   # Contains project implementation
│   ├── DR.ipynb           # Dimensionality reduction notebook
│   ├── clustering.ipynb   # Clustering and visualization notebook
├── venv/                  # Virtual environment (optional)
├── LICENSE                # MIT License
├── README.md              # Project documentation
├── requirements.txt       # List of required dependencies
```

---

## Dataset Description

**Dataset**: [Kepler Exoplanet Search Results - cumulative.csv](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)

- **Total Rows**: 9,564  
- **Total Columns**: 50  
- **Target Variable**: `koi_disposition` (CONFIRMED, CANDIDATE, FALSE POSITIVE)

**Key Features**:
- **Planetary & Orbital Data**: Period, radius, duration, temperature, depth  
- **Stellar Data**: Stellar radius, effective temperature, surface gravity  
- **Target and Disposition Data**: Planet classification, detection methods

## Part 1: Dimensionality Reduction

### Techniques Implemented:
- **Principal Component Analysis (PCA)**  
- **Linear Discriminant Analysis (LDA)**  
- **t-SNE** (t-distributed stochastic neighbor embedding)  
- **Multidimensional Scaling (MDS)**  
- **Singular Value Decomposition (SVD)**

### Classification Models:
- **RandomForestClassifier**
- **K-Nearest Neighbors (KNN)**

### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-score (for classification tasks)
- Explained variance (for PCA/SVD)
- Visualization in 2D space (for t-SNE, MDS)

### Key Insights:
- **LDA** provided the best class separation due to its supervised nature.  
- **PCA** and **SVD** retained significant variance and worked well with classifiers.  
- **t-SNE** and **MDS** were effective for visualization but not for downstream ML tasks.  
- RandomForest outperformed KNN across most techniques.

---

## Part 2: Clustering and Unsupervised Learning

### Clustering Algorithms:
- **KMeans**
- **Agglomerative Clustering**
- **DBSCAN**
- **Birch**

### Evaluation Metrics:
- **Silhouette Score**
- **Calinski-Harabasz Index**
- **Davies-Bouldin Score**

### Dimensionality Reduction for Visualization:
- **PCA** and **UMAP** were used to project high-dimensional data to 2D before clustering and visualization.

### Key Insights:
- **KMeans** and **Agglomerative Clustering** produced interpretable clusters with good separation in PCA/UMAP projections.
- **DBSCAN** captured denser planetary systems but was sensitive to `eps` value.
- **Birch** showed competitive performance with minimal tuning.
- **UMAP** was more effective than t-SNE in preserving both local and global structure.

---

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. Create and activate a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the notebooks:
   ```bash
   jupyter notebook
   ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
