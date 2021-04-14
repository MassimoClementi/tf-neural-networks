# tf-neural-networks
Scripts, projects and exercises that are based on the TensorFlow framework


## Structure of the folder at 2021-04-14
```
    .
    ├── LICENSE
    ├── README.md
    ├── datasets
    │   └── weatherAUS.csv
    ├── python_scripts
    │   ├── 0_Basics_and_MLP.py
    │   ├── 1_IRIS_dataset_and_MLP.py
    │   ├── 2_MNIST_dataset_and_CNNs.py
    │   ├── 3_CIFAR_and_tf_data_module.py
    │   └── archive
    ├── requirements.txt
    ├── saved_models
    │   ├── 2_MNIST_dataset_and_CNNs
    │   └── 3_CIFAR_and_tf_data_module
    ├── setup.sh
    ├── side_projects
    │   └── MNIST-opencv
    └── venv
        ├── bin
        ├── include
        ├── lib
        ├── pyvenv.cfg
        └── share

```

## Configure virtual environment

First make sure that all the scripts and the python files have execution permissions. If that is not the case just use `chmod +x`. Then run `setup.sh` to setup the virtual environment.


## Usage

Open the main folder with VScode and use the embedded _interactive python shell_ (similar to a Jupiter Notebook) to run the cells in the scripts, which can be found in either `python_scripts` or `side_projects` folders.

Remember to check the code for links to eventual datasets that require a manual download.