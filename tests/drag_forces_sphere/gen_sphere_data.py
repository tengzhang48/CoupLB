"""Generate LAMMPS data file for icosahedral sphere markers."""
import numpy as np
import sys

def icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    raw = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            raw.append([0, s1, s2*phi])
            raw.append([s1, s2*phi, 0])
            raw.append([s2*phi, 0, s1])
    raw = np.array(raw, dtype=float)
    return raw / np.linalg.norm(raw[0])

def icosahedron_faces(verts):
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = set()
    for i in range(12):
        for j in range(i+1, 12):
            if abs(np.linalg.norm(verts[i]-verts[j]) - edge_len) < edge_len*0.01:
                edges.add((i,j))
    adj = {i: set() for i in range(12)}
    for i,j in edges:
        adj[i].add(j); adj[j].add(i)
    faces = set()
    for i in range(12):
        for j in adj[i]:
            if j<=i: continue
            for k in adj[i] & adj[j]:
                if k<=j: continue
                faces.add((i,j,k))
    return list(faces)

def subdivide(verts, faces, freq):
    if freq == 1: return verts
    all_verts = list(verts)
    def add_or_get(pt):
        pt = pt / np.linalg.norm(pt)
        for idx, v in enumerate(all_verts):
            if np.linalg.norm(pt - v) < 1e-10: return idx
        all_verts.append(pt)
        return len(all_verts) - 1
    for a,b,c in faces:
        va, vb, vc = verts[a], verts[b], verts[c]
        for i in range(freq+1):
            for j in range(freq+1-i):
                u, v = i/freq, j/freq
                add_or_get(u*va + v*vb + (1-u-v)*vc)
    return np.array(all_verts)

freq = int(sys.argv[1]) if len(sys.argv) > 1 else 4
R = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
box_half = float(sys.argv[3]) if len(sys.argv) > 3 else 16.0
mass_total = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

verts = icosahedron_vertices()
faces = icosahedron_faces(verts)
pts = subdivide(verts, faces, freq) * R
N = len(pts)
expected = 10*freq**2 + 2
assert N == expected, f"Expected {expected}, got {N}"

m = mass_total / N
L = box_half

with open("sphere_markers.data", "w") as f:
    f.write(f"LAMMPS data file - freq-{freq} icosahedral sphere, R={R}, {N} markers\n\n")
    f.write(f"{N} atoms\n\n")
    f.write(f"1 atom types\n\n")
    f.write(f"{-L:.6f} {L:.6f} xlo xhi\n")
    f.write(f"{-L:.6f} {L:.6f} ylo yhi\n")
    f.write(f"{-L:.6f} {L:.6f} zlo zhi\n\n")
    f.write(f"Masses\n\n")
    f.write(f"1 {m:.10f}\n\n")
    f.write(f"Atoms # atomic\n\n")
    for i, p in enumerate(pts):
        f.write(f"{i+1} 1 {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

print(f"Written sphere_markers.data: {N} markers, R={R}, freq={freq}")
print(f"  mass/marker = {m:.10f}  (total = {mass_total})")
