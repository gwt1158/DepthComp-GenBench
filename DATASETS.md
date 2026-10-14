# Dataset sources and acknowledgements

DepthComp-GenBench does not redistribute dataset files. Download the data from
the sources below, review the applicable license or terms of use, and cite the
original dataset paper in publications that use the benchmark.

## Prepared benchmark data

| Benchmark role | Download | Notes |
| --- | --- | --- |
| NYU Depth V2 source domain | [Hugging Face](https://huggingface.co/datasets/heuliuh/nyu_depth_v2) | Prepare aligned RGB, target depth, fixed sparse depth, and masks as documented in the root README. |
| KITTI Depth Completion source domain | [Hugging Face](https://huggingface.co/datasets/dangth2004/KITTI-Depth-Estimation) · [Official benchmark](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php) | The official benchmark provides projected raw LiDAR and accumulated semi-dense annotations. |
| PSD OOD validation collection | [Hugging Face](https://huggingface.co/datasets/cslxx/PSD-val) | Prepared source for the OOD protocol. Follow the PSD split and sparse-input definitions. |
| ScanNet addition | [Gated Hugging Face mirror](https://huggingface.co/datasets/Pointcept/scannet-compressed) · [Official site](http://www.scan-net.org/) | Access requires acceptance of the ScanNet terms. Do not redistribute it. |

## OOD-16 original sources

The PSD OOD collection is the benchmark download. The links below identify the
original dataset owners for attribution, documentation, and license review.

| Domain | Original source | Protocol group |
| --- | --- | --- |
| VOID1500 | [VOID project and downloader](https://github.com/alexklwong/void-dataset) | Raw |
| SUN RGB-D | [Official project](https://rgbd.cs.princeton.edu/) | Raw |
| TOFDC | [Official dataset page](https://yanzq95.github.io/projectpage/TOFDC/index.html) | Raw |
| DIML | [DIML/CVL RGB-D dataset](https://dimlrgbd.github.io/) | Raw |
| DIODE indoor | [Official project](https://diode-dataset.org/) | Indoor |
| Middlebury | [Official stereo datasets](https://vision.middlebury.edu/stereo/data/) | Indoor |
| ETH3D | [Official dataset overview](https://www.eth3d.net/overview) | Indoor |
| ScanNet | [Official project](http://www.scan-net.org/) | Indoor; license-gated |
| DrivingStereo | [Official project](https://drivingstereo-dataset.github.io/) | Outdoor |
| Argoverse | [Argoverse 1](https://www.argoverse.org/av1.html) | Outdoor |
| Cityscapes | [Official dataset](https://www.cityscapes-dataset.com/) | Outdoor |
| DIODE outdoor | [Official project](https://diode-dataset.org/) | Outdoor |
| Hypersim | [Official repository](https://github.com/apple/ml-hypersim) | Synthetic |
| Virtual KITTI 2 | [Official project](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) | Synthetic |
| KITTI-360 | [Official project](https://www.cvlibs.net/datasets/kitti-360/) | Special |
| Stanford2D3D | [Official project](https://buildingparser.stanford.edu/dataset.html) | Special |

The OOD-16 order, expected sample counts, and display names are fixed in
`unified_evaluation/ood_registry.py`. ScanNet is the sixteenth licensed domain.
Use `eval_ood15` only when ScanNet access is unavailable, and label its result
as OOD-15 partial rather than Macro-16.

## Required attribution

When publishing results, cite the dataset papers for every source and test
domain used, as well as the PSD protocol paper:

> Shenglun Chen, Xinzhu Ma, Hong Zhang, Haojie Li, and Zhihui Wang,
> "Propagating Sparse Depth via Depth Foundation Model for Out-of-Distribution
> Depth Completion," IEEE Transactions on Image Processing, 2025.

Refer to each official page for its preferred BibTeX entry. Dataset access
does not grant permission to rehost or sublicense the original files.
