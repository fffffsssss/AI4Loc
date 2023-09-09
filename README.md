# AI4Loc
Artificial intelligence for single molecule localization microscopy (SMLM), using deep learning methods to localize 
single molecules


## Installation
The code was tested on Windows 11 with Python and PyTorch. Packages required can be found in `requirements.txt`. 
The following software / hardware is tested:
* Python=3.9.7
* CUDA 11.8
* Intel i7-11800H
* NVIDIA RTX 3080 Laptop GPU

To build the Anaconda environment, run the following command:
```commandline
conda create --name ailoc python=3.9.7 
```
Then activate the environment:
```commandline
conda activate ailoc
```
Install the required packages:
```commandline
pip install -r requirements.txt
```

The downloaded files shall be organized as the following hierarchy:
```
├── root
│   ├── ailoc
│   │   ├── common                       // common tools used by all algorithms
│   │   │   ├── beads_calibration        // calibrate the PSF, copy from smap_py
│   │   │   │   ├── main_calibration.py  // the entry of beads calibration
│   │   │   │   ├── ...
│   │   │   ├── analyzer.py              // utilize xxloc to analyze experimental data
│   │   │   ├── xxloc.py                 // the abstract class xxloc
│   │   │   ├── ...                      // other common modules, such as notebook_gui, utilities...
│   │   ├── simulation                   // simulation tools
│   │   │   ├── simulator.py             // the simulator class for generating SMLM data
│   │   │   ├── ...                      // other simulation modules such as PSF, camera, etc.
│   │   ├── deeploc                      // the implementation of deeploc
│   │   │   ├── deeploc.py               // deeploc class, which inherits from xxloc
│   │   │   ├── network.py               // the network structure of deeploc, also the forward function
│   │   │   ├── loss.py                  // the loss function of deeploc
│   │   ├── syncloc                      // the implementation of syncloc
│   │   │   ├── ...
│   │   ├── ...                          // other xxloc algorithms, fd-deeploc, etc.
│   ├── datasets                         // store the datasets
│   ├── results                          // store the results
│   ├── demo_pipelines                   // notebooks and scripts for demo, including learning, analysis, etc.
│   │   ├── notebook
│   │   ├── pyscripts
```