import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from modules import projection_head
import SimCLR_dataset
import faiss
from tqdm import tqdm



def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity 
    """
    dot_product = np.dot(a, b) # x.y
    norm_a = np.linalg.norm(a) #|x|
    norm_b = np.linalg.norm(b) #|y|
    return dot_product / (norm_a * norm_b)


def main(cfg_path: str, ckpt_path: str, device: str):
    num_test_imgs = 10
    clusters = 5
    epochs = 100

    # Import configuration
    with open(cfg_path, 'r') as file:
        args = yaml.safe_load(file)

    # Define model
    model: nn.Module = projection_head.__dict__["SingleProjection"](
            args["in_features"], 
            args["hidden_features"],
            args["out_features"],
            args['dropout']
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Find ids
    dataset_path = Path(args["dataset_path"])
    val_ids = open(args["val_dataset"], 'r').read().splitlines()

    test_set = SimCLR_dataset.__dict__["EvalDataset"](
        dataset_path, 
        val_ids, 
        args['multiplier'],
        mock_length=len(val_ids)
    )

    all_metric = 0
    for _ in tqdm(range(epochs)):
        # Select 5 random classes
        selected_idx = random.sample(range(len(val_ids)), k=clusters)

        # Extract all features of each class
        out_features = [(idx, feat) for idx in selected_idx for feat in test_set[idx]]
        keys, out_features = list(zip(*out_features))
        
        with torch.no_grad():
            pred_features = model(torch.tensor(np.asarray(out_features))).detach().cpu().numpy()

        # K-means of pred_features
        # Create index
        n_init = 10
        max_iter = 300
        kmeans = faiss.Kmeans(d=pred_features.shape[1], k=clusters, niter=max_iter, nredo=n_init)
        kmeans.train(pred_features)
        # Find centroids
        centroids = kmeans.centroids
        
        # Find centroid appartenenza
        D, I = kmeans.index.search(pred_features, 1)

        metric = 0
        # Select test feature
        test_idxs = random.sample(range(len(keys)), num_test_imgs)
        for test_idx in test_idxs:
            test_key = keys[test_idx]
            test_feat = pred_features[test_idx]
            
            # Find closest centroid to test sample
            max_score = 0
            best_centroid = clusters
            for i, centroid in enumerate(centroids):
                score = cos_sim(test_feat, centroid)
                if score > max_score:
                    best_centroid = i
                    max_score = score
            
            # Calculate metric
            qn = 0
            rel = 0
            for k, i in zip(keys, I):
                if k == test_key:
                    qn += 1
                    if i == best_centroid:
                        rel += 1
            metric += rel / qn
        metric /= num_test_imgs

        all_metric += metric
    all_metric /= epochs
    print(f"Cumulative metric mP@5: {all_metric:.4f}")




if "__main__" in __name__:
    main(
        "/data/GoogleUniversalImageEmbedding/experiments/SimCLR-512-temp0.5/config.yaml",
        "/data/GoogleUniversalImageEmbedding/experiments/SimCLR-512-temp0.5/checkpoint_12000.tar",
        "cpu"
    )


