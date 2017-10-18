# [MPI_INF_3DHP Dataset](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)
We port the `mpi_inf_3dhp` dataset to the common interface used throughout this repository. As with all datasets here, we do not supply the data, nor pretend to represent those who do. See the [website](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) for full details/licenses.

## Setup
1. Download the install scripts and utility files from the official [website](http://gvv.mpi-inf.mpg.de/3dhp-dataset/). Extract the contents to some directoy, `/path/to/mpi_inf_3dhp_dir`.
2. Follow the instructions in the `README.md` just extracted. In particular, you'll have to change `ready_to_download` to `1`, and optionally download all subjects (rather than just 1 and 2 as is the default). For these instructions, we assume you do not change the `destination`.
3. Make the download script executable and run it.
```
cd /path/to/mpi_inf_3dhp_dir
chmod +x get_dataset.sh
./get_dataset.sh
```
4. Set the `MPI_INF_PATH` to the destination (`/path/to/mpi_inf_3dhp_dir` unless you changed the `conf.ig` `destination` value).
```
export MPI_INF_PATH=/path/to/mpi_inf_3dhp_dir
```
Consider putting this in your `~/.bashrc` file, otherwise you'll have to repeat this step for each new terminal.
5. Run `dataset.py` as a script. This should convert the `.mat` and `.calibration` files to `hdf5` format.

## Status
Still an active work in progress.

TODO:
* port `get_sequence_info` (and `get_camera_set`?)
* verify projections/camera properties
* work out what `univ_annot3` actually means. Same across all cameras?
* transformed datasets/registrations
