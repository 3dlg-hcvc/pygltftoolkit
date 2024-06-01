import sys

import numpy as np

sys.path.append("../../")

import pygltftoolkit as pygltk

DEMOS = [0, 4]

# Load the glTF 2.0 file
gltf = pygltk.load("./1d95eadf60b29d6d14d1969463abed7ab31ff800.glb")
print(gltf)

if 0 in DEMOS:
    # Load annotations
    gltf.load_stk_segmentation_openable("./1d95eadf60b29d6d14d1969463abed7ab31ff800.artpre.json")
    gltf.load_stk_articulation("./fpModel.1d95eadf60b29d6d14d1969463abed7ab31ff800.articulations.json")
    gltf.load_stk_precomputed_segmentation("./1d95eadf60b29d6d14d1969463abed7ab31ff800.connectivity.segs.json")
    print(gltf)

if 1 in DEMOS:
    # Recolor gltf
    face_colors = np.array([[1.0, 0.0, 0.0, 1.0]] * len(gltf.faces))
    gltf.color_faces(face_colors)

if 2 in DEMOS:
    interactive_vis = gltf.interactiveVisualizer()
    gltf.show(interactive_vis)

if 3 in DEMOS:
    static_vis = gltf.staticVisualizer()
    gltf.render(static_vis, "./temp.png")
    gltf = pygltk.load("./1d95eadf60b29d6d14d1969463abed7ab31ff800.glb")
    gltf.render(static_vis, "./temp2.png")

if 4 in DEMOS:
    gltf.sample_uniform(1000000, vertices=True, semantic_map={"drawer": 0, "door": 1, "lid": 2, "base": 3})
