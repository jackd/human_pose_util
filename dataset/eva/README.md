Provides access to human eva 1 dataset, either through raw data or the precalculated/compressed hdf5 version.

# Usage

## `raw`
This file is based on the uncompressed data that can be downloaded [here](http://humaneva.is.tue.mpg.de/datasets_human_1).

Ensure the `HUMAN_EVA_1_PATH` environment variable is set to the root directory of the uncompressed eva dataset. It should be structured like:
- Background
  - Background_1_(BW1).avi
  - ...
- S1
  - Calibration_DATA
    - BW1.cal
    ...
  - Image_Data
    - Box_1_(BW2).avi
    ...
  - Mocap_Data
    - Box_1.c3d
    ...
  - Sync_Data
    - Box_1_(BW1).ofs
- S2
...
- S4

From command line, use
```
export HUMAN_EVA_1_PATH=/path/to/human_eva_1/uncompressed/directory
```
Add this line to your `~/.bashrc` to avoid having to run for each new terminal.

While the raw version of the dataset cache's results within the one session, the calculation time is not trivial. For extended use, we recommend the hdf5 version provided by `hdf5.py`.

## `hdf5`
This file provides an equivalent dataset stored in `hdf5` format. It also includes a conversion script to convert from the `raw.py` version. The conversion is almost entirely based on [this work](https://github.com/ynaka81/DeepPose).
