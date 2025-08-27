# src/synmind.py
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import defaultdict
from typing import List, Dict

# Load the embedding model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def cluster_keypoints(keypoints: List[str], num_clusters: int = 3) -> Dict[str, List[str]]:
    """
    Takes a list of keypoints and groups them into meaningful clusters.
    """
    if not keypoints or len(keypoints) < num_clusters:
        return {"Theme 1": keypoints}

    embeddings = embedding_model.encode(keypoints, convert_to_tensor=False)
    
    actual_num_clusters = min(num_clusters, len(keypoints))
    kmeans = KMeans(n_clusters=actual_num_clusters, random_state=42, n_init='auto')
    kmeans.fit(embeddings)
    
    clusters = defaultdict(list)
    for i, keypoint in enumerate(keypoints):
        cluster_label = kmeans.labels_[i]
        clusters[f"Theme {cluster_label + 1}"].append(keypoint)
        
    return dict(clusters)