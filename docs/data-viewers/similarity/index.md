# Similarity Viewer

[Run the Similarity Viewer](main.html)

This 2D chart shows a series of points.  Each point
corresponds to a single paper.  The distance between
any two points represents how similar the papers are.
Points that are close together indicated that the papers are similar.  Points that are far apart indicate that the papers are different.
Note that the X and Y dimensions don't actually mean anything concrete like the score.  They are just ways to group similar papers.

## Embeddings

The way we create this chart is to use large language models (LLMs) to create a data structure for each paper called an **embedding**.  An embedding is a vector of numbers that place each paper in a multi-dimensional space based on the words and concepts described in each paper.

## Sample Run

```sh
$ python create-embeddings.py 
```

```
Prostate Cancer Papers - Embedding Analysis
==================================================
Loading data from ../../data/output-full-scored.csv...
Loaded 7338 papers
Creating text embeddings...
  - Generating TF-IDF vectors...
  - TF-IDF matrix shape: (7338, 5000)
  - Applying UMAP reduction...
  - 2D embeddings shape: (7338, 2)
Performing clustering with 8 clusters...

Cluster Analysis:
==================================================
Cluster 0: 1019 papers
  Avg Score: 18.4
  Top Terms: biopsy, mri, imaging, patients, resonance
  Main Treatment: Unknown

Cluster 1: 1037 papers
  Avg Score: 9.3
  Top Terms: expression, ar, mir, pca, cell
  Main Treatment: Unknown

Cluster 2: 686 papers
  Avg Score: 16.3
  Top Terms: psma, sup, pet, ct, pet ct
  Main Treatment: Unknown

Cluster 3: 1089 papers
  Avg Score: 13.8
  Top Terms: risk, patients, pca, men, 95
  Main Treatment: Unknown

Cluster 4: 968 papers
  Avg Score: 7.6
  Top Terms: cells, cell, sub, cancer cells, activity
  Main Treatment: Unknown

Cluster 5: 814 papers
  Avg Score: 29.1
  Top Terms: patients, radiotherapy, dose, radiation, therapy
  Main Treatment: Unknown

Cluster 6: 951 papers
  Avg Score: 22.3
  Top Terms: patients, men, prostatectomy, radical, screening
  Main Treatment: Unknown

Cluster 7: 774 papers
  Avg Score: 17.7
  Top Terms: metastatic, patients, castration, bone, resistant
  Main Treatment: Unknown

Exporting to papers-embeddings.json...
✓ Successfully exported 7338 papers with embeddings

✓ Analysis complete!
✓ Embeddings and clusters saved to: papers-embeddings.json
✓ Ready for 2D similarity visualization!
```