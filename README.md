# pygltftoolkit

## Table of Contents

- [About](#about)
- [Install](#install)
- [Usage](#usage)
- [Limitations and Development](#dev)

## About <a name = "about"></a>

pygltftoolkit aims to provide high-level API for loading and processing glTF 2.0 files. 

### Install <a name = "install"></a>

There are just a couple of dependencies to install:

```
python>=3.8
numpy
pygltflib
PIL
trimesh
```

If you want to use FPD to sample points
```
torch
```
and run
```
# PointOps from PointCept libbrary - https://github.com/Pointcept/Pointcept
cd pointops
python setup.py install
```

Please also install meshcat for visualization

```
git submodule update --init --recursive

cd meshcat-python
python setup.py install
```

## Usage <a name = "usage"></a>

Loading a file:

```
import pygltgtoolkit as pygltk
gltf = pygltk.load("/path/to/file.glb")

# You can print to verify the content
print(gltf)
```

pygltftoolkit provides an "unrolled" view of the scene in addition to basic graph structure

```
vertices = gltf.vertices
faces = gltf.faces
triangles = vertices[faces]
```

You can load annotations produced by the [Scene Toolkit](https://github.com/smartscenes/sstk) as follows:

```
gltf.load_stk_segmentation("/path/to/id.artpre.json")
gltf.load_stk_segmentation_openable("/path/to/id.artpre.json")  # This version involves specific label merging/propagation strategy from S2O
gltf.load_stk_articulation("/path/to/dataset.id.articulations.json")
gltf.load_stk_precomputed_segmentation("/path/to/id.connectivity.segs.json")
```

In order to preserve dependencies on graph structure when using "unrolled" view, a number of maps are stored. Note that these maps correspond to the nodes that contain meshes and meshes directly (that is parents are not accounted for). Maps exist for nodes, meshes and primitives (however maps are local for primitives as they don't have global id).

```
# Obtaining nodes containing meshes
nodes = np.unique(gltf.node_map)

# Working withs the corresponding triangles
for node_id in nodes:
    node_mask = gltf.node_map == node_id
    node_triangles = vertices[faces[node_mask]]
    # Process node_triangles ...
```

Mesh manipulation and modification.

```
# Change coordinate frame. Updates both gltf.vertices and matrices of all nodes
origin_coordinate_frame = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])

target_coordinate_frame = np.array([[0, 1, 0],
                                    [0, 0, 1],
                                    [-1, 0, 0]])

gltf.transform_coordinate_frame(origin_coordinate_frame, target_coordinate_frame)

# Rescale the scene by some factor
gltf.rescale(2)
```

## Limitations and Development <a name = "dev"></a>

There is a number of limitations that may or may not be lifted in the future:
* Only a single scene is supported. Multiple scenes are used rarely and there was even a proposal to remove the support for multiple scenes in a single file. See [discussion](https://github.com/KhronosGroup/glTF/issues/1542). 
* Skin, animation and sampler part of the glTF graph are not supported (and are not planned currently).
* Currently only the following attributes are extracted from primitives, if present: ["COLOR_0", "NORMAL", "POSITION", "TEXCOORD_0"]. 
* Limited number of texture types supported (excluding normalTexture, occlusionTexture, emissiveTexture)

Current TODOs (approximately in order of priority):
* Mesh geometry modification
* KDTree and KNN
* Extensions support
