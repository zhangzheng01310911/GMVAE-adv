#!/usr/bin/env python3
from pathlib import Path

import numpy as np

rng = np.random.default_rng(11)
time, nodes, channels = 240, 12, 3
t = np.arange(time, dtype=np.float32)
values = np.empty((time, nodes, channels), dtype=np.float32)
for node in range(nodes):
    phase = node / nodes * np.pi
    values[:, node, 0] = 45 + 15 * np.sin(2 * np.pi * t / 24 + phase)
    values[:, node, 1] = 70 + 20 * np.cos(2 * np.pi * t / 24 + phase)
    values[:, node, 2] = 0.3 + 0.1 * np.sin(2 * np.pi * t / 12 + phase)
values += rng.normal(0, 0.5, values.shape)
adjacency = np.zeros((nodes, nodes), dtype=np.float32)
for node in range(nodes):
    adjacency[node, (node - 1) % nodes] = 1
    adjacency[node, (node + 1) % nodes] = 1
target = Path("work/synthetic_traffic.npz")
target.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(target, values=values, adjacency=adjacency)
print(target)
