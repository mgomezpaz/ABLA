# Why Search Everywhere? Smart Membrane Segmentation for Particle Detection

## Abstract

Particle identification in cryo-electron tomography (cryo-ET) remains a significant challenge due to inherently low signal-to-noise ratios, variable particle characteristics, and the extensive search volumes required for analysis. We present ABLA (Automated Bacterial Labelling Annotations), an innovative pipeline that addresses these challenges by intelligently constraining the particle search space through precise membrane segmentation. By combining SAM2 (Segment Anything Model 2) with entropy-based slice selection and automated centroid detection, ABLA generates 3D masks of bacterial outer membranes, typically reducing the search volume to up to a quarter of the original tomogram, depending on bacterial size.

Shannon entropy helps identify optimal search regions, improving membrane-associated protein detection by eliminating areas where target proteins are unlikely to be found.
