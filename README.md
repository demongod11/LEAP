# LEAP: Learning-guided Quality Cut Selection for Faster Technology Mapping

This repository contains the code of the LEAP framework, a novel ML-guided cut choices for ASIC technology mapping


## Installing dependencies

We recommend using [Anaconda](https://www.anaconda.com/) environment to install dependencies for running our framework and models. The dependencies used can be found in the *leap_env.yml* file provided in our repository. 
The environment can be recreated using the command ```conda env create -f leap_env.yml```.

Make sure that that the cudatoolkit version in the gpu matches with the pytorch-geometric's (and dependencies) CUDA version.


## Organisation

	├── LEAP
	│   ├── abc			                    # Modified code of abc
	│   ├── cut-sol    
    |   |   ├── model_weights               # Weights of Cut Classifier and Delay Predictor
    |   |   ├── plots                       # Various plots of results for visualisation
    |   |   ├── cs_dataset.py               # Loads dataset
    |   |   ├── evaluate_class.py           # Cut Classifier - testing pipeline
    |   |   ├── evaluate_qor.py             # Delay Predictor - testing pipeline
    |   |   ├── train_class.py              # Cut Classifier - training pipeline
    |   |   ├── train_qor.py                # Delay Predictor - training pipeline
    |   |   ├── model.py                    # Cut Classifier and Delay Predicor models
    |   |   └── generate_final_cuts.cpp     # Sweep algorithm
	│   ├── data			                # Data of all the designs                                                                                                                   
	│   ├── generate_all_cuts.cpp			# Generates all_cuts.csv from cuts.csv
    |   └── leap_env.yml                    # Dependency file

**Note -** The data folders for ```hyp``` and ```multiplier``` are large in size and hence not uploaded here. You can download them from [here](https://iitgoffice-my.sharepoint.com/:f:/g/personal/r_chigarapally_iitg_ac_in/EjekCNp3SD9KuCc9iRqjD_oBUwOutkQGO-x8ckG3FhG43w?e=bCugCA)


## ABC

The ```abc``` present in this repository is the modified abc. This abc has 2 extra commands. They are ```generate_nc_info``` and ```selective_map```. The usage of both the commands is specified below - 

1. ```generate_nc_info -n nodes.csv -c cuts.csv``` - This command generates 2 files namely ```nodes.csv``` and ```cuts.csv``` which contains the node level and cut level information of all the nodes and cuts of the circuit.

2. ```selective_map -c few_cuts.csv -m few_maps.csv``` - This command takes an input ```few_cuts.csv``` and performs mapping using only the cuts specified in this file. The adjacency list of the final mapping is stored in ```few_maps.csv```.


## ABC-Unlimited

The code for ```abc-unlimited``` can be found [here](https://github.com/demongod11/abc-unlimited.git). 

**Note -** All the commands of abc-unlimted are same as that of the vanilla-abc. The only differnce is that abc-unlimited considers the entire exhaustive set of cuts for performing the technology mapping.


## Dataset Generation

To train the models on a design, we need a dataset which contains truth tables and it's corresponding delay values of the design. Example file can be found at ```data/square/cut_stats.csv```. -1 in the cut_delay column represents that the associated cut cannot be implemented by the supergate library. 

To generate this file follow the steps below -
1. Go to ```abc/src/selectiveMap/selectiveMap.c``` and uncomment the lines 407-411.
2. Build ```abc``` by using ```make``` command.
3. Now perform ```selective_map``` using ```all_cuts.csv``` (can be generated using ```generate_all_cuts.cpp```) for a design to get it's corresponding ```cut_stats.csv```.

**Note -** After generating the dataset, comment the lines 407-411 and then build the ```abc``` again.


## Training

Follow the steps mentioned below to train the models on a design (ex. ```square```) - 

1. Go to ```abc``` folder and start *abc* - ```./abc```.
2. Load the standard cell library - ```read_lib asap7_clean.lib```.
3. Load the design into *abc* and perform structural hashing - ```read ../data/square/square.v;strash```.
4. Generate *nodes.csv* and *cuts.csv* - ```generate_nc_info -n nodes.csv -c cuts.csv```.
5. Now perform the ```Dataset Generation``` method as specified in the above section. At the end of this step, a file named ```cut_stats.csv``` should have been generated.
6. Now execute the ```train_class.py``` file to train the ```Cut Classifier```
    ```PYTHONUNBUFFERED=1 nohup python train_class.py > train_class_out.log 2> train_class_err.log &```
    The corresponding logs will be stored in ```train_class_out.log``` and ```train_class_err.log```.
7. Now execute the ```train_qor.py``` file to train the ```Delay Predictor```
    ```PYTHONUNBUFFERED=1 nohup python train_qor.py > train_qor_out.log 2> train_qor_err.log &```
    The corresponding logs will be stored in ```train_qor_out.log``` and ```train_qor_err.log```.
    The model weights for the ```Cut Classifier``` and ```Delay Predictor``` will be stored in ```model_weights/class_model``` and ```model_weights/qor_model``` respectively.

*Class Training Loss.png*, *Class Validation Loss.png*, *QoR Training Loss.png* and *QoR Validation Loss.png* would also be generated.


## Testing

Follow the steps mentioned below to test the models on a design (ex. ```c6288```) - 

1. Go to ```abc``` folder and start *abc* - ```./abc```.
2. Load the standard cell library - ```read_lib asap7_clean.lib```.
3. Load the design into *abc* and perform structural hashing - ```read ../data/square/square.v;strash```.
4. Generate *nodes.csv* and *cuts.csv* - ```generate_nc_info -n nodes.csv -c cuts.csv```.
5. Now execute the ```evaluate_class.py``` file to infer the ```Cut Classifier```
    ```PYTHONUNBUFFERED=1 nohup python evaluate_class.py > ../data/c6288/evaluate_class_out.log 2> ../data/c6288/evaluate_class_err.log &```    
    The corresponding logs will be stored in *evaluate_class_out.log* and *evaluate_class_err.log*.
6. Wait till the above process gets completed. After the execution has been completed, a file named *interim_cut_delays.csv* would be generated.
7. Now execute the ```evaluate_qor.py``` file to infer the ```Delay Predictor```
    ```PYTHONUNBUFFERED=1 nohup python evaluate_qor.py > ../data/c6288/evaluate_qor_out.log 2> ../data/c6288/evaluate_qor_err.log &```
    The corresponding logs will be stored in *evaluate_qor_out.log* and *evaluate_qor_err.log*.
8. Wait till the above process gets completed. After the execution has been completed, a file named *cut_delays.csv* would be generated.
9. Now execute the ```generate_final_cuts.cpp``` file which has the **Sweep** algorithm implementation. This would generate a ```final_cuts_10.csv``` file.
10. Now perform selective mapping using this file - ```selective_map -c final_cuts_10.csv -m final_maps_10.csv```.
11. Get the final area and delay values of this mapping using this command - ```topo;stime```.



<!-- ## How to cite

If you use this code/dataset, please cite:

```

``` -->
