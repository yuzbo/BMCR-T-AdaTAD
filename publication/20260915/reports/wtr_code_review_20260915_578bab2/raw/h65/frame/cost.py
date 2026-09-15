"""Transparent operation accounting for frame engine."""
def empty_trace(depth=0):
    return {"q": [0]*depth, "kv": [0]*depth, "heavy_mlp": [0]*depth,
            "light": [0]*depth, "tia": [0]*depth, "routing_qk": [0]*depth,
            "depth_masks": [], "spatial_masks": [], "score_qk": [0]*depth}
