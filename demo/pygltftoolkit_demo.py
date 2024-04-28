import sys

sys.path.append("../../")

import pygltftoolkit as pgltk

# Load the glTF 2.0 file
gltf = pgltk.load("./1d95eadf60b29d6d14d1969463abed7ab31ff800.glb")
print(gltf)

# Load annotations
gltf.load_stk_segmentation("./1d95eadf60b29d6d14d1969463abed7ab31ff800.artpre.json")
gltf.load_stk_articulation("./fpModel.1d95eadf60b29d6d14d1969463abed7ab31ff800.articulations.json")
gltf.load_stk_precomputed_segmentation("./1d95eadf60b29d6d14d1969463abed7ab31ff800.connectivity.segs.json")

print(gltf)
import pdb; pdb.set_trace()
