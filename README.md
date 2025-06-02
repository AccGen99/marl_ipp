# Scalable Multi-Robot Informative Path Planning for Target Mapping via Deep Reinforcement Learning

This repository contains the code of our paper titled "Scalable Multi-Robot Informative Path Planning for Target Mapping via Deep Reinforcement Learning".
Our paper can be found [here](https://arxiv.org/pdf/2409.16967).

This codebase builds upon the author's previous works titled "CAtNIPP: Context-aware attention-based network for informative path planning" by Cao et al. published in the Proceedings of the Conference on Robot Learning (CoRL, 2023), and "Deep Reinforcement Learning with Dynamic Graphs for Adaptive Informative Path Planning" by Vashisth et al. published in the IEEE Robotics and Automation Letters (RAL, 2024). 
We found the above works and open-sourced code to be extremely helpful towards development of our approach and advancing RL-based IPP approaches. Please acknowledge this by citing the above works as well:

```commandline
@inproceedings{cao2023catnipp,
  title={{CAtNIPP: Context-aware attention-based network for informative path planning}},
  author={Cao, Yuhong and Wang, Yizhuo and Vashisth, Apoorva and Fan, Haolin and Sartoretti, Guillaume Adrien},
  booktitle=corl,
  year={2023}
}

@article{vashisth2024ral,
  author={Vashisth, Apoorva and R{\"u}ckin, Julius and Magistri, Federico and Stachniss, Cyrill and Popovic, Marija},
  journal={IEEE Robotics and Automation Letters (RA-L)}, 
  title={{Deep Reinforcement Learning with Dynamic Graphs for Adaptive Informative Path Planning}}, 
  year={2024},
  pages={1-8},
}
```

## Setting up code

Make a new conda environment -

```
conda create -n ipp python=3.9
```

Our approach is based on pytorch. Other than pytorch, please install the following packages -

```
ray
imageio
scipy
matplotlib
shapely
scikit-learn
```

Activate the conda environment before training or inference -

```
conda activate ipp
```

## Training

Train the model by running the following command -

```
python driver.py
```
To specify the number of parallel environment instances, change the variable ```NUM_META_AGENT``` in [```parameters.py```](parameters.py)

To specify the number of robots in the multi-robot system associated with each environment instance, change the variable ```NUM_AGENTS``` in [```parameters.py```](parameters.py)

To specify the test environment change the variable ```TEST_TYPE``` in [```test_parameters.py```](test_parameters.py) to one of ```random``` or ```grid```.

You can change the range of target detection in sensor module by changing values of ```DEPTH``` parameter in [```parameters.py```](parameters.py). Note that the environment built in python considers an occupancy grid of 50 cells in each of 3 directions and the ```DEPTH``` variable specifies the depth of sensor frustum in terms of number of grid cells.

## Inference

Run the model for test -

```
python test_driver.py
```
To specify the size of the environment, change the variable ```ENV_SIZE``` in [```test_parameters.py```](parameters.py)

To specify the number of robots in the multi-robot system associated with each environment instance, change the variable ```NUM_AGENTS``` in [```test_parameters.py```](parameters.py)

## Key Files

* driver.py - Driver of program. Holds global network.
* runner.py - Compute node for training. Maintains a single meta agent containing one instance of environment.
* worker.py - A single agent in a the IPP instance.
* parameter.py - Parameters for training and test.
* env.py - Define the environment class.
