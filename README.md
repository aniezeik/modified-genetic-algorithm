# Radio Resource Management for 6G Networks: A Cell less Approach

This repository implements the research paper by Kooshki et al.[1]. The code implements a modified genetic algorithm which is used to optimize scheduling and resource management in cell-free mobile networks, resulting in improved system capacity for both indoor and outdoor environments when compared with legacy cellular scheduler. 


## To run the simulation

 
git clone https://github.com/aniezeik/modified-genetic-algorithm.git  
cd modified-genetic-algorithm  
python genetic_algorithm.py


### Expected Output

The simulation generates two CDF charts that show the system capacity improvements for:
- Indoor environment for the different number of RU and UE combinations.
- Outdoor(umi - urban micro) environment for the different number of RU and UE combinations.

## Project Structure

The codebase is organized into four main modules:

### `config.py`
Contains the configuration parameters for the simulation.

### `network_layout.py`
Generates the network topologies for the simulation environment:
- **Outdoor environment**: Hexagonal grid layout
- **Indoor environment**: Rectangular grid layout
This module also has a function to calculate the distance between the RUs and the UEs, which is needed for the pathloss calculations.

### `propagation_environment.py`
This module handles the radio propagation modeling. It contains:
- Pathloss calculation for indoor and outdoor environments following the ITU standards.
- Multipath fading generation using rayleigh fading model.
- Reference Signal Received Power (RSRP) derivation.

### `genetic_algorithm.py`
This module, which is the core of the simulations contains the modified genetic algorithm  and the legacy scheduling implementation. It contains different functions which are listed below:
- Fitness function calculation
- Initial population generation
- Roulette wheel selection
- Single-point crossover 
- Adaptive mutation
- Legacy round-robin for comparison
- Main simulation definition

A *pdf report* is also part of this repository. It details the implementation design decisions and also gives answers to the questions as part of the assignment.


### Important notation to consider in the code base:
M = Number of RUs  
N = Number of UEs  
K = Number of RBs  
T = Number of TTIs (which is 1 in this simulation)  


## Requirements
The only requirements are numpy and matplotlib


## References
[1] F. Kooshki, M. A. Rahman, M. M. Mowla, A. G. Armada, and A. Flizikowski, “Efficient radio resource
management for future 6g mobile networks: A cell-less approach,” IEEE Networking Letters, vol. 5,
no. 2, pp. 95–99, 2023.






