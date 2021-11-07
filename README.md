# Belayer

An algorithm for modeling layered tissues. Work in progress!

## Installation
Belayer depends on the following python packages: numpy, scipy, pandas, sklearn, [networkx](https://github.com/networkx/networkx), [glmpca](https://github.com/willtownes/glmpca-py).
Further installation TBD.

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
+ estimated piecewise function coefficients file TBD.

## Example