# HUMAN POSE UTIL

Various utilities for human pose estimation.

## Setup

Clone the repository and install using `pip install -e`.

```bash
cd path/to/parent_folder
git clone https://github.com/jackd/human_pose_util.git
pip install -e human_pose_util
```

## Datasets

This repository comes with support for [Human3.6M](http://vision.imar.ro/human3.6m/description.php) (h3m) and [HumanEva_I](http://humaneva.is.tue.mpg.de/datasets_human_1) (eva) datasets. Due to licensing issues these are not provided here - see the respective websites for details.

### Setting up datasets

#### [Human3.6M](http://vision.imar.ro/human3.6m/description.php) (h3m)

To work with the Human3.6M dataset, you must have the relevant `.cdf` files in an uncompressed local directory, referenced here as `MY_H3M_DIRECTORY`. For licensing reasons, we cannot provide the raw Human3.6m data. Please consult the [website](http://vision.imar.ro/human3.6m/description.php) to source the original data. This directory must have the following structure:

```
- MY_H3M_DIRECTORY
  - D2_positions
    - S1
      - Directions.54138969.cdf
      - ...
    - S5
      - ...
    ...
  - D3_positions
    - S1
    ...
  - D3_positions_mono
    - S1
    ...
  - Videos
    - S1
    ...
```

`Videos` aren't used in module, though the dataset has a `video_path` attribute which assumes the above structure.

To let the scripts know where to find the data, run the following in a terminal

```bash
export H3M_PATH=/path/to/MY_H3M_DIRECTORY
```

Consider adding this line to your `.bashrc` if you will be using this a lot.

#### [HumanEva_I](http://humaneva.is.tue.mpg.de/datasets_human_1) (eva)

To work with the HumanEva_I dataset, you must have the uncompressed data available in `MY_EVA_1_DIR` which should have the following structure:

```
- MY_EVA_1_DIR
  - S1
    - Calibration_Data
      - BW1.cal
      ...
    - Image_Data
      - Box_1_(BW2).avi
      ...
    - Mocap_Data
      - Box_1.c3d
      - Box_1.mat
      ...
    - Sync_Data
      - Box_1_(BW1).ofs
      ...
  - S2
    ...
  ...
```

`Image_Data` is not used in this module, thought the dataset has a `video_path` attribute which assumes the above structure.

To let scripts know where to find the data, run the following in a terminal

```bash
export H3M_PATH=/home/jackd/Development/datasets/human3p6m/data
```

Consider adding this line to your `.bashrc` if you will be using this a lot.

### Registering a new dataset

A new dataset can be registered using

```python
human_pose_util.register.dataset_register[dataset_id] = {
    'train': train_datastet,
    'eval': eval_dataset,
}
```

If your dataset uses a different skeleton from those provided (see `human_pose_util.skeleton.Skeleton`), you'll need to precede this with a similar skeleton registration line

```python
human_pose_util.register.skeleton_register[my_skeleton_id] = my_skeleton
```

After that, training/inference can procede as normal.

See `human_pose_util.dataset.h3m` and `human_pose_util.dataset.eva` for examples.

### Visualizations

Visualizations are dependent on `matplotlib` or `glumpy`, though these are not included in the dependencies.

Installing `matlplotlib`:

```bash
pip install matplotlib
```

Installing `glumpy`:

```bash
git clone https://github.com/glumpy/glumpy.git
cd glumpy
python setup.py install
```

### TODO

- Continue removing all dataset.interface stuff. Mostly there with h3m, but start with `dataset.h3m.pose_sequence`. It's overly complicated - just map lists of dictionaries (and use hdf5 instead of dicts if memory becomes too intensive - unlikely for just pose stuff, though maybe necessary for images/heatmaps).
- `dataset/mpi_inf/README.md` TODOs
