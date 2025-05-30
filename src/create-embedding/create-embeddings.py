#!/usr/bin/env python3
"""
Prostate Cancer Papers - Text Embedding and Similarity Analysis

This script creates 2D embeddings of research papers based on their textual content,
allowing visualization of similar papers clustered together in space.

Features:
- TF-IDF vectorization of paper abstracts and titles
- PCA and t-SNE dimensionality reduction 
- UMAP for high-quality 2D embeddings
- Clustering analysis using K-means
- Export to JSON for interactive visualization

Requirements:
- scikit-learn
- umap-learn  
- pandas
- numpy
"""

import pandas as pd
import numpy as np
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP, fall back to t-SNE if not available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")
    print("Using t-SNE instead for dimensionality reduction.")

class PaperEmbeddingAnalyzer:
    def __init__(self):
        self.papers_df = None
        self.embeddings_2d = None
        self.clusters = None
        self.vectorizer = None
        self.cluster_labels = None
        
    def load_data(self, csv_file):
        """Load and preprocess the papers data."""
        print(f"Loading data from {csv_file}...")
        self.papers_df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.papers_df)} papers")
        
        # Clean and prepare text data
        self.papers_df['combined_text'] = self.papers_df.apply(
            lambda row: self._combine_text_fields(row), axis=1
        )
        
        # Filter out papers with insufficient text
        min_text_length = 50
        initial_count = len(self.papers_df)
        self.papers_df = self.papers_df[
            self.papers_df['combined_text'].str.len() >= min_text_length
        ]
        final_count = len(self.papers_df)
        
        if final_count < initial_count:
            print(f"Filtered out {initial_count - final_count} papers with insufficient text")
        
        return self.papers_df
    
    def _combine_text_fields(self, row):
        """Combine title, abstract, and other text fields for analysis."""
        fields = []
        
        # Add title (weighted more heavily)
        title = str(row.get('title', ''))
        if title and title != 'nan':
            fields.extend([title] * 3)  # Weight title 3x
        
        # Add abstract
        abstract = str(row.get('abstract', ''))
        if abstract and abstract != 'nan' and len(abstract) > 20:
            fields.append(abstract)
        
        # Add details (journal, year info)
        details = str(row.get('details', ''))
        if details and details != 'nan':
            fields.append(details)
        
        combined = ' '.join(fields)
        
        # Clean the text
        combined = re.sub(r'[^\w\s]', ' ', combined)  # Remove special chars
        combined = re.sub(r'\s+', ' ', combined)      # Normalize whitespace
        combined = combined.lower().strip()
        
        return combined
    
    def create_embeddings(self, method='umap', n_components=2, random_state=42):
        """Create 2D embeddings using TF-IDF + dimensionality reduction."""
        print("Creating text embeddings...")
        
        # Step 1: TF-IDF Vectorization
        print("  - Generating TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,          # Limit vocabulary size
            min_df=2,                   # Ignore terms in fewer than 2 documents
            max_df=0.95,                # Ignore terms in more than 95% of documents
            stop_words='english',       # Remove common English words
            ngram_range=(1, 2),         # Include unigrams and bigrams
            lowercase=True,
            strip_accents='unicode'
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(self.papers_df['combined_text'])
        print(f"  - TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Step 2: Dimensionality Reduction
        if method == 'umap' and UMAP_AVAILABLE:
            print("  - Applying UMAP reduction...")
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                random_state=random_state,
                verbose=False
            )
        elif method == 'tsne':
            print("  - Applying t-SNE reduction...")
            # First reduce to ~50 dimensions with PCA to speed up t-SNE
            if tfidf_matrix.shape[1] > 50:
                print("  - Pre-reducing with PCA...")
                pca = PCA(n_components=50, random_state=random_state)
                tfidf_matrix = pca.fit_transform(tfidf_matrix.toarray())
            
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(30, len(self.papers_df) // 4),
                random_state=random_state,
                verbose=0,
                max_iter=1000
            )
        else:
            print("  - Applying PCA reduction...")
            reducer = PCA(n_components=n_components, random_state=random_state)
        
        # Apply dimensionality reduction
        if method == 'umap' and UMAP_AVAILABLE:
            self.embeddings_2d = reducer.fit_transform(tfidf_matrix)
        elif method == 'tsne':
            self.embeddings_2d = reducer.fit_transform(tfidf_matrix.toarray() if hasattr(tfidf_matrix, 'toarray') else tfidf_matrix)
        else:
            self.embeddings_2d = reducer.fit_transform(tfidf_matrix.toarray())
        
        print(f"  - 2D embeddings shape: {self.embeddings_2d.shape}")
        return self.embeddings_2d
    
    def perform_clustering(self, n_clusters=8):
        """Perform clustering on the 2D embeddings."""
        print(f"Performing clustering with {n_clusters} clusters...")
        
        if self.embeddings_2d is None:
            raise ValueError("Must create embeddings first")
        
        # Standardize the embeddings for clustering
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings_2d)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(embeddings_scaled)
        
        # Add cluster labels to dataframe
        self.papers_df['cluster'] = self.cluster_labels
        
        # Analyze clusters
        self._analyze_clusters()
        
        return self.cluster_labels
    
    def _analyze_clusters(self):
        """Analyze and characterize each cluster."""
        print("\nCluster Analysis:")
        print("=" * 50)
        
        self.cluster_info = {}
        
        for cluster_id in range(max(self.cluster_labels) + 1):
            cluster_papers = self.papers_df[self.papers_df['cluster'] == cluster_id]
            
            # Get top terms for this cluster
            cluster_texts = cluster_papers['combined_text'].tolist()
            if cluster_texts:
                cluster_tfidf = self.vectorizer.transform(cluster_texts)
                mean_tfidf = np.mean(cluster_tfidf.toarray(), axis=0)
                
                # Get feature names and find top terms
                feature_names = self.vectorizer.get_feature_names_out()
                top_indices = np.argsort(mean_tfidf)[-10:][::-1]
                top_terms = [feature_names[i] for i in top_indices]
                
                # Calculate average score for cluster
                avg_score = cluster_papers['score'].mean()
                
                # Get treatment types in cluster
                treatment_counts = cluster_papers['treatment_type'].value_counts() if 'treatment_type' in cluster_papers.columns else {}
                
                self.cluster_info[cluster_id] = {
                    'size': len(cluster_papers),
                    'avg_score': avg_score,
                    'top_terms': top_terms,
                    'top_treatment': treatment_counts.index[0] if len(treatment_counts) > 0 else 'Unknown'
                }
                
                print(f"Cluster {cluster_id}: {len(cluster_papers)} papers")
                print(f"  Avg Score: {avg_score:.1f}")
                print(f"  Top Terms: {', '.join(top_terms[:5])}")
                print(f"  Main Treatment: {treatment_counts.index[0] if len(treatment_counts) > 0 else 'Unknown'}")
                print()
    
    def export_to_json(self, output_file):
        """Export embeddings and paper data to JSON for visualization."""
        print(f"Exporting to {output_file}...")
        
        if self.embeddings_2d is None:
            raise ValueError("Must create embeddings first")
        
        # Prepare papers data with embeddings
        papers_data = []
        
        for idx, row in self.papers_df.iterrows():
            paper_data = {
                'id': int(idx),
                'title': str(row.get('title', '')),
                'abstract': str(row.get('abstract', ''))[:300] + "..." if len(str(row.get('abstract', ''))) > 300 else str(row.get('abstract', '')),
                'details': str(row.get('details', '')),
                'author': str(row.get('author', '')),
                'url': str(row.get('url', '')),
                'score': float(row.get('score', 0)),
                'pmid': str(row.get('pmid', '')),
                'x': float(self.embeddings_2d[idx, 0]),
                'y': float(self.embeddings_2d[idx, 1]),
                'cluster': int(row.get('cluster', 0)) if 'cluster' in row else 0
            }
            
            # Add derived fields if they exist
            for field in ['treatment_type', 'study_type', 'quality_category', 'journal', 'year']:
                if field in row:
                    paper_data[field] = str(row[field]) if pd.notna(row[field]) else 'Unknown'
            
            papers_data.append(paper_data)
        
        # Create output structure
        output_data = {
            'metadata': {
                'total_papers': len(papers_data),
                'embedding_method': 'TF-IDF + UMAP' if UMAP_AVAILABLE else 'TF-IDF + t-SNE',
                'features': self.vectorizer.get_feature_names_out().tolist() if self.vectorizer else [],
                'n_clusters': len(self.cluster_info) if hasattr(self, 'cluster_info') else 0
            },
            'clusters': self.cluster_info if hasattr(self, 'cluster_info') else {},
            'papers': papers_data,
            'bounds': {
                'x_min': float(np.min(self.embeddings_2d[:, 0])),
                'x_max': float(np.max(self.embeddings_2d[:, 0])),
                'y_min': float(np.min(self.embeddings_2d[:, 1])),
                'y_max': float(np.max(self.embeddings_2d[:, 1]))
            }
        }
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully exported {len(papers_data)} papers with embeddings")
        return output_data

def main():
    """Main analysis pipeline."""
    print("Prostate Cancer Papers - Embedding Analysis")
    print("=" * 50)
    
    # Configuration
    input_file = '../../data/output-full-scored.csv'
    output_file = 'papers-embeddings.json'
    embedding_method = 'umap' if UMAP_AVAILABLE else 'tsne'
    n_clusters = 8
    
    # Initialize analyzer
    analyzer = PaperEmbeddingAnalyzer()
    
    # Load data
    papers_df = analyzer.load_data(input_file)
    
    # Create embeddings
    embeddings = analyzer.create_embeddings(method=embedding_method)
    
    # Perform clustering
    clusters = analyzer.perform_clustering(n_clusters=n_clusters)
    
    # Export results
    analyzer.export_to_json(output_file)
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ Embeddings and clusters saved to: {output_file}")
    print(f"✓ Ready for 2D similarity visualization!")

if __name__ == "__main__":
    main()