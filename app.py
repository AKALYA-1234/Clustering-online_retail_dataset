import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score


# -------------------
# Main Clustering Function
# -------------------
def run_clustering(file, model_file, method, n_clusters, eps, min_samples):
    if file is None:
        return None, "⚠️ Please upload a dataset."

    # Load dataset
    df = pd.read_csv(r"C:\Users\DELL\Downloads\OnlineRetail.csv",encoding=latin1)
    numeric_df = df.select_dtypes(include=[np.number]).dropna()

    if numeric_df.shape[1] < 2:
        return None, "⚠️ Need at least 2 numeric columns for clustering."

    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(numeric_df)

    labels = None
    title = ""

    # -------------------
    # Custom Model Load
    # -------------------
    if model_file is not None:
        try:
            model = joblib.load(model_file.name)
            labels = model.fit_predict(X)
            title = f"Custom Model Loaded from {model_file.name}"
        except Exception as e:
            return None, f"❌ Error loading model: {str(e)}"

    else:
        # -------------------
        # Built-in Models
        # -------------------
        if method == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X)
            title = f"KMeans (k={n_clusters})"

        elif method == "Hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X)
            title = f"Hierarchical (k={n_clusters})"

        elif method == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            title = f"DBSCAN (eps={eps}, min_samples={min_samples})"

    # -------------------
    # Plotting
    # -------------------
    plt.figure(figsize=(6,4))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap="viridis", s=30)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig("cluster_result.png")
    plt.close()

    # -------------------
    # Status
    # -------------------
    score = "N/A"
    if labels is not None and len(set(labels)) > 1 and -1 not in labels:
        score = round(silhouette_score(X, labels), 3)

    return "cluster_result.png", f"✅ Done! {title} | Silhouette Score: {score}"


# -------------------
# Gradio UI
# -------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🔷 Clustering App with Model Selection & PKL Upload")

    with gr.Row():
        file = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
        model_file = gr.File(label="Upload Custom Model (.pkl)", file_types=[".pkl"])

    method = gr.Dropdown(
        ["KMeans", "Hierarchical", "DBSCAN"],
        label="Choose Model",
        value="KMeans"
    )

    with gr.Row():
        n_clusters = gr.Slider(2, 15, value=3, step=1, label="Number of Clusters (KMeans/Hierarchical)")
        eps = gr.Slider(0.1, 10.0, value=0.5, step=0.1, label="EPS (DBSCAN)")
        min_samples = gr.Slider(2, 20, value=5, step=1, label="Min Samples (DBSCAN)")

    btn = gr.Button("🚀 Run Clustering")

    status = gr.Label(label="Status")
    output_img = gr.Image(type="filepath", label="Cluster Visualization")

    btn.click(
        fn=run_clustering,
        inputs=[file, model_file, method, n_clusters, eps, min_samples],
        outputs=[output_img, status]
    )

demo.launch()
