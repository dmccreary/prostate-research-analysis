# Similarity Viewer

This 2D chart shows a series of points.  Each point
corresponds to a single paper.  The distance between
any two points represents how similar the papers are.
Points that are close together indicated that the papers are similar.  Points that are far apart indicate that the papers are different.
Note that the X and Y dimensions don't actually mean anything concrete like the score.  They are just ways to group similar papers.

## Embeddings

The way we create this chart is to use large language models (LLMs) to create a data structure for each paper called an **embedding**.  An embedding is a vector of numbers that place each paper in a multi-dimensional space based on the words and concepts described in each paper.  