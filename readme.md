# Removing noisy links benefits link prediction in complex network(DLLP)

## Environment

- **Python**: 3.11.8  
- Install all dependencies:
  ```bash
  pip install -r requirements.txt

## Workflow

### 1. Generate Node Embeddings

Use a GCN model to learn node representations from the input graph.

```
python .\gnn_embed.py --method gcn
```

**Arguments:**

- `--method`: Specify the embedding method. Here, we use `gcn`.

------

### 2. Compute Link Loss Perturbation Influence

This step measures how the perturbation of each link affects the overall model loss. It helps identify **noisy links** in the network.

```
python .\DLLP_influence.py
```

