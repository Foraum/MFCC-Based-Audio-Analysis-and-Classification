# MFCC_Based Audio Analysis And Classification 
## Project Overview 
This project uses Mel-frequency cepstral coefficients (MFCCs) to analyze, cluster, and classify 115 audio files. By extracting and engineering advanced features, we aimed to distinguish between genres, artists, and musical characteristics, addressing challenges like high dimensionality and overlapping clusters.

For more details on the problem statement, refer to the [Project Question Paper PDF](Project_Question_Paper.pdf).

## Dataset Details 
- **Source**: 115 audio files converted into CSV files containing MFCC coefficients.
- **Format**: Each file has 20 rows and ~10,320 columns for a 2-minute song.
- **Content Categories:**
  1. Indian National Anthem
  2. Marathi Bhav Geet and Lavni
  3. Hindi songs by Asha Bhosle and Kishor Kumar
  4. English songs by Michael Jackson

## Feature Engineering 
### Challenges with Initial Features
Basic MFCCs, Delta, and Delta-Delta captured spectral and temporal dynamics but missed rhythmic and harmonic nuances, leading to overlapping clusters.

**Advanced Features Used**
1. Spectral Roll-off Frequency
2. Spectral Contrast
3. Tempo and Beat Strength
4. Spectral Flatness
5. Spectral Bandwidth
6. Zero-Crossing Rate (ZCR)
7. Global MFCC Features
8. Autocorrelation
9. Energy Band Features

## Clustering Methodology
### Steps for Clustering
- **Feature Reduction:**
 1. Removed highly correlated features using Variance Threshold.
 2. Applied SVD to reduce features to 116 columns, achieving a VIF of 1.
- **Clustering Algorithms:**
 1. Implemented K-Means for initial clustering.
 2. Used PCA to enhance clustering precision by minimizing collinearity.
 3. Visualized and refined clusters using t-SNE.
 4. Re-clustered into four groups after removing the distinct clusters of Michael Jackson and National Anthem to improve precision.

## Code Overview  

| **File Name**       | **Description**                                                                                 |  
|----------------------|-----------------------------------------------------------------------------------------------|  
| `(A)Maincode.ipynb`     | Performs clustering of all 115 songs into six groups based on MFCC features and advanced metrics. |  
| `(B)Reducdecode.ipynb`  | Re-clusters songs after removing Michael Jackson and National Anthem clusters for improved precision. |  


## Testing Methodology
- **Test Dataset**: Downloaded additional songs for validation, representing each artist and genre.
- **Cluster Evaluation**: Incorporated test songs into the clusters and assessed their alignment.
- **Validation Metrics**: Verified test songs’ placement in the correct clusters based on predefined attributes.

## Technical Details
### Tools and Libraries
**Python Libraries**: librosa, scikit-learn, pandas, matplotlib, numpy.<br>
**Environment**: Jupyter Notebook 
### Clustering Metrics
**Silhouette Score**: Measured the compactness of clusters.<br>
**DB Index**: Assessed cluster separation.

## Documentation
[Project Question Paper PDF](Project_Question_Paper.pdf): Contains the original problem statement.<br>
[Final Report](Final_Project_PPT.pdf): Detailed report of the project, including the methodology and results.

