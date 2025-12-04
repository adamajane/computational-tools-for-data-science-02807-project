# Music Similarity and Clustering System

This folder contains the completed implementation and analysis.

## Overview

**Notebooks (run in order):**

1. **1-data-cleaning-and-feature-vector.ipynb** – Data integration & feature engineering

   - Merges Spotify audio features with Last.fm popularity and tags (55,129 tracks)
   - Implements TF-IDF vectorization on tags with SVD compression (100 components)
   - Applies PCA to audio features (8 components) and standardization to popularity metrics
   - Creates 111-dimensional weighted feature matrix
   - Uses stratified 80/20 split to prevent feature transform leakage

2. **2-lsh_and_search.ipynb** – Locality-Sensitive Hashing & similarity search

   - Implements random hyperplane LSH (8 hash tables, 16 bits per table)
   - Benchmarks against exact cosine similarity: achieves 0.70 recall@10 with 33× speedup
   - Explores alternative approaches (MinHash on tags) and their limitations

3. **3-cluster-evaluation.ipynb** – K-Means clustering & evaluation

   - Clusters 55,129 tracks into 113 genres using K-Means
   - Evaluates quality with silhouette (0.331), purity (0.471), NMI (0.585), ARI (0.241)
   - Visualizes clusters in PCA, t-SNE, UMAP space
   - Explores LSH-aided K-Means variant

4. **4-final-recommendation-system.ipynb** – End-to-end recommender
   - Builds user profiles from listening history
   - Blends multiple signals: cosine similarity (60%), cluster priors (10%), genre priors (10%), popularity (10%), artist overlap (5%), seed similarity (5%)
   - Supports exact and LSH-based candidate generation
   - Provides interpretable score breakdowns for each recommendation

**Artifacts:**

- `artifacts/final_cleaned_dataset.csv` – Cleaned and deduplicated track metadata
- `artifacts/track_features.npz` – Pre-computed feature matrices (train, validation, and combined)
