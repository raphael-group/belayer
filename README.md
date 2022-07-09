# Belayer

Belayer: Modeling discrete and continuous spatial variation in gene expression from spatially resolved transcriptomics.
Cong Ma\*, Uthsav Chitra\*, Shirley Zhang, Ben Raphael

## Installation
Belayer depends on the following python packages: numpy, scipy, pandas, sklearn, [networkx](https://github.com/networkx/networkx), [glmpca](https://github.com/willtownes/glmpca-py).
Further installation TBD.

## Tutorial
See `tutorial.ipynb` for a complete tutorial of how to run Belayer on three different datasets. Note that the tutorial requires downloading two files from [here](https://drive.google.com/drive/folders/150OZEl5Np2rgvSJm4E8QCCtIOjJkZ39N?usp=sharing) - one file for the DLPFC tutorial and one for the mouse wound tutorial - and placing them in their respective folders.

## Usage
```
python belayer.py (-i <10x directory> | -s <count matrix file> <spatial coordinate file>) -m <running mode> -L <number layers> [options]
```

Details of required and optional input arguments:

Argument | Data type | Description
---      | :---:     | ---
-i (--indir) | str | Input 10X directory for ST data.
-s (--stfiles) | list of str | Input count matrix file followed by spatial coordinate file for ST data. Count matrix and spatial coordinate must have the same number of spots. Only one of -i and -s is allowed.
-m (--mode) | char | Running mode. A: axis-aligned layered tissue. R:rotated axis-aligned layered tissue. S:arbitrarily curved tissue supervised by annotated layers. L:layered tissue with linear layer boundaries.
-L (--nlayers) | int | Number of layers to infer.
-a (--annotation) | str | File of annotated layers for each spot when using S mode.
-o (--outprefix) | str | Output prefix.
-p (--platform) | str | Platform for spatial transcriptomics data. Only used when running mode is S.


## Output
+ \<outprefix>_layer.csv. This file contains the identified layers for each spot.
+ estimated piecewise function coefficients file TBD, see tutorial for details.

## Example
