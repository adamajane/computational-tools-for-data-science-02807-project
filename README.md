# Music Similarity and Clustering

## 02807 Computational Tools for Data Science

This repository contains the final project for 02807 Computational Tools for Data Science. We explore music similarity search and clustering using Spotify and Last.fm data.

## Project Overview

We built an end-to-end music recommendation system that:

- Merges **55,129 tracks** from Spotify audio features and Last.fm popularity/tags
- Implements **TF-IDF + SVD** to compress semantic tag information
- Applies **Locality-Sensitive Hashing (LSH)** for fast approximate similarity search (~33× speedup)
- Clusters tracks into 113 genres using **K-Means** with silhouette score 0.331
- Provides personalized recommendations blending cosine similarity, clustering, and popularity signals

## Data

Raw Spotify and Last.fm data is stored via Git LFS. To fetch it:

```bash
git lfs install
git lfs fetch --all
```

## How to Run

1. Run notebooks in order:

   - `main/1-data-cleaning-and-feature-vector.ipynb` – builds feature vectors
   - `main/2-lsh_and_search.ipynb` – implements LSH
   - `main/3-cluster-evaluation.ipynb` – evaluates clustering
   - `main/4-final-recommendation-system.ipynb` – end-to-end recommender
