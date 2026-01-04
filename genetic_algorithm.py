

# This code implements a genetic algorithm for resource allocation in mobile networks.
# It includes functions for to compute the fitness, initialize the population, perform selection, and other genetic operations.
# The main execution block runs the MGA for different network scenarios and compares it to legacy round-robin scheduling.
# Executing this code results in CDF plots of system capacity improvements for the urban micro (outdoor) and indoor environments.


def compute_fitness(individual, number_rus, Gain, P_per_RB_W, noise_W, RB_bw=180e3):
    """
    Computes fitness (total system throughput) for a given individual.
    Handles power splitting when multiple UEs share a PRB on the same RU.
    Includes interference from other RUs on the same PRB.

    Args:
        individual: (K, M + N) allocation matrix
        number_rus: number of RUs
        Gain: channel gain tensor (M, K, N, T)
        P_per_RB_W: transmit power per RB (W)
        noise_W: noise power (W)
        RB_bw: RB bandwidth (Hz)

    Returns:
        fitness: (M, K, N, T) array representing throughput for each RU-UE-RB-TTI
        fitness_sum: scalar representing total system throughput
    """

    # Extract UE-RU and UE-RB allocation matrices
    UE_RU = individual[:, :number_rus]  # (K, M)
    UE_RB = individual[:, number_rus:]  # (K, N)

    # Expand for broadcasting to enable multiplication with the Gain tensor
    UE_RU_expanded = UE_RU.T[:, :, None, None]   # (M, K, 1, 1)
    UE_RB_expanded = UE_RB[None, :, :, None]     # (1, K, N, 1)

    # Allocate Gain based on UE-RU and UE-RB associations 
    Gain_allocated = Gain * UE_RU_expanded * UE_RB_expanded     # (M, K, N, T)

    # Count number of UEs per PRB per RU 
    ue_count_per_rb = np.sum(UE_RU_expanded * UE_RB_expanded, axis=1, keepdims=True)  # (M, 1, N, T)
     
    # Avoid division by zero by setting zero counts to one (will not affect since Gain_allocated will be zero there)
    ue_count_per_rb = np.where(ue_count_per_rb == 0, 1, ue_count_per_rb)
    
    # Power per UE: divide total PRB power by number of UEs using that PRB
    P_per_ue_rb = P_per_RB_W / ue_count_per_rb  # (M, 1, N, T)

    # Power received by each UE from each RU on each PRB
    received_power = P_per_ue_rb * Gain_allocated       # (M, K, N, T)

    # Total power per PRB (sum over all RUs AND all UEs on each PRB) 
    total_rx_power_per_prb = np.sum(received_power, axis=(0, 1), keepdims=True)   # shape: (1, 1, N, T)  
    
    # Interference: total received power on PRB minus the power received by the specific UE from its RU on that PRB
    interference = total_rx_power_per_prb - received_power  # (M, K, N, T)

    # sinr in Watts
    sinr_w = received_power / (interference + noise_W ) 
    
    # Calculate fitness (throughput) per UE-RU-RB
    fitness = RB_bw * np.log2(1.0 + sinr_w)

    # Calculate total fitness (sum over all UEs, RUs, and RBs)
    fitness_sum = np.sum(fitness) 

    return fitness, fitness_sum


def create_initial_population(population_size, number_ues, number_rus, number_rbs, rsrp_dBm):
    """
    Creates an initial population for the genetic algorithm.

    Args:
        population_size: number of individuals in the population
        number_ues: number of UEs
        number_rus: number of RUs
        number_rbs: number of RBs
        rsrp_dBm: RSRP values (M, K) array

    Returns:
        population: list of individuals (each individual is a (K, M + N) allocation matrix)
    """  
    # Initialize list to hold population
    population = []   

    # --- UE–RU association (associate UE to RU with maximum RSRP) ---
    UE_RU = np.zeros((number_ues, number_rus), dtype=int)   
    for k in range(number_ues):
        # Find RU with highest RSRP for UE k. rsrp_dBm shape: (M, K)
        best_ru = np.argmax(rsrp_dBm[:, k])     
        # Associate UE k to RU with the highest RSRP
        UE_RU[k, best_ru] = 1                   

    # --- Create population with UE-RU and UE-RB allocations ---
    for p in range(population_size): 
        # UE–RB allocation matrix      
        UE_RB = np.zeros((number_ues, number_rbs), dtype=int)   
        # random order to determine round robin allocation
        ue_order = np.random.permutation(number_ues)  
              
        # --- Round robin RB allocation among UEs associated to each RU ---
        for m in range(number_rus):
            # check which UEs are associated to RU m
            associated_ues = [k for k in ue_order if UE_RU[k, m] == 1]   
            # If no UEs are associated, skip this RU
            if not associated_ues:
                continue
            num_associated = len(associated_ues)
            # Determine number of iterations for round-robin allocation
            num_iterations = max(number_rbs, num_associated)    

            # ---Allocate RBs in round-robin fashion---
            for i in range(num_iterations):  
                #Cycle through associated UEs              
                ue_idx = associated_ues[i % num_associated]     
                # cycle through RBs
                rb_idx = i % number_rbs  
                #Allocate RB to UE                       
                UE_RB[ue_idx, rb_idx] = 1                       

        # Combine UE–RU and UE–RB into single individual
        individual = np.concatenate((UE_RU, UE_RB), axis=1) 
        # Add individual to population
        population.append(individual)

    return population


def constraint_three_penalty(population):
    """
    To satisfy minimum rate constraints for each UE and penalize those that do not meet the threshold,
    this function makes the rate of the UE zero if it does not meet the threshold.

    Args:
        population: list of individuals (each individual is a (K, M + N) allocation matrix)

    Returns:
    fitness_sum: list of total fitness for each individual in the population taking into account constraint 3
    """ 

    # Initialize list to hold fitness sums
    fitness_sum = []
    for i in range(len(population)):
        # Get the datarate for the UE-RU-RB allocations in the individual  
        fitness, _ = compute_fitness(population[i], number_rus, G, P_per_RB_W, noise_W)
        # fitness has shape (M, K, N, T)
        # Sum over M (RUs), N (RBs), and T (TTIs) to get per-UE rates
        per_user_rate = np.sum(fitness, axis=(0, 2, 3))  # Shape: (K,)
        # Minimum rate threshold in bits/s - 0.3Mbps
        per_user_rate_threshold = 300000     

        # Apply penalty for UEs not meeting the minimum rate
        for j in range(len(per_user_rate)): 
            if per_user_rate[j] < per_user_rate_threshold:
                # Penalize by setting rate to zero if below threshold
                per_user_rate[j] = 0  
        # Sum the per-UE rates to get total fitness for the individual           
        fitness_sum.append(np.sum(per_user_rate))   
    return fitness_sum


def roulette_wheel_selection(population):
    """
    Performs roulette wheel selection on the population based on fitness values.

    Args:
        population: list of individuals (each individual is a (K, M + N) allocation matrix)

    Returns:
        selected_population: list of selected individuals after roulette wheel selection
        best_individual: the best individual in the population before selection
    """
    # Compute total fitness for each individual taking into account constraint 3
    fitness_sum = constraint_three_penalty(population)     

    # Elitism: Identify the best individual in the population before selection to retain the best over generations
    best_individual_index = np.argmax(fitness_sum)
    best_individual = population[best_individual_index]

    #---Selection---
    # Total fitness of the population
    total_fitness = np.sum(fitness_sum) 
    # Selection probabilities proportional to fitness    
    probabilities = fitness_sum / total_fitness 
    selected_population = []
    # Select individuals based on probabilities
    for _ in range(len(population)):
        # random selected index based on probabilities
        selected_index = np.random.choice(len(population), p=probabilities)    
        # Append selected individual to new population 
        selected_population.append(np.copy(population[selected_index]))
    
    return selected_population, best_individual


def single_point_crossover(population, crossover_probability, number_rus):

    """
    Performs single-point crossover on the population.

    Args:
        population: list of individuals (each individual is a (K, M + N) allocation matrix)
        crossover_probability: probability of performing crossover

    Returns:
        new_population_after_crossover: list of individuals after crossover
    """

    # Initialize list to hold new population after crossover
    new_population_after_crossover = [] 
    # Process parents in pairs
    for i in range(0, len(population), 2):
        parent1 = np.copy(population[i])
        parent2 = np.copy(population[i+1 if i+1 < len(population) else 0])  
        
        random_num = np.random.rand()
        # Crossover only in the UE-RB allocation part
        crossover_area = parent1.shape[1] - number_rus  

        if random_num < crossover_probability:
            # select a random crossover point
            point = np.random.randint(1, crossover_area) 
            # Perform column crossover in the UE-RB part only. Swap columns after the crossover point
            parent1[:, number_rus + point:], parent2[:, number_rus + point:] = parent2[:, number_rus + point:].copy(), parent1[:, number_rus + point:].copy()
        
        new_population_after_crossover.extend([parent1, parent2])
    
    return new_population_after_crossover


def mutation(individual, mutation_probability, number_rus):
    """
    Performs bit-flip mutation on the RB allocation part of an individual.

    Args:
        individual: Binary array (K, M + N) where first M columns are UE-RU, last N columns are UE-RB
        mutation_probability: Probability of flipping each bit in the RB part
        number_rus: number of RUs (to identify where RB part starts)

    Returns:
        mutated: Individual with RB allocation mutated
    """
    # Create a copy of the individual to mutate so that the original is not altered
    mutated = individual.copy() 
    # Only mutate the RB part (columns from after number_rus onwards)
    rb_part = mutated[:, number_rus:] 
    # Create an array of random values to determine which rb bits to flip based on the probability
    mask = np.random.rand(*rb_part.shape) < mutation_probability
    # Flip bits in RB part according to the mask 
    rb_part[mask] = 1 - rb_part[mask] 
    # Update the mutated individual
    mutated[:, number_rus:] = rb_part
    
    return mutated

  

def run_ga(generations, population_size, number_ues, number_rus, number_rbs, rsrp_dBm, G, P_per_RB_W, noise_W, RB_bw):
    """
    Runs the Modified Genetic Algorithm (MGA) for resource allocation.

    Args:
        generations: number of generations to run the algorithm
        population_size: number of individuals in the population
        number_ues: number of UEs
        number_rus: number of RUs
        number_rbs: number of RBs
        rsrp_dBm: RSRP values (M, K) array

    Returns:
        best_idx: index of the best individual in the final population
        overall_best_individual_fitness: fitness value of the best individual
    """
    # Step 1: Create initial population
    population = create_initial_population(population_size, number_ues, number_rus, number_rbs, rsrp_dBm)
    
    for gen in range(generations):
        # --- Step 2: Selection ---
        # Perform roulette wheel selection and also get the best individual before selection for elitism
        selected_population, best_individual_before_selection = roulette_wheel_selection(population)

        # --- Step 3: Crossover ---
        # Perform single-point crossover on selected population
        new_population_after_crossover = single_point_crossover(selected_population, crossover_probability=0.7, number_rus=number_rus)
       
        # --- Step 4: Mutation on the children from the crossover ---
        
        # Initialize list to hold the new population after mutation
        next_population = []

        # Compute fitness for all children from the crossover
        fitness_sum_children = np.array([
            compute_fitness(ind, number_rus, G, P_per_RB_W, noise_W)[1] 
            for ind in new_population_after_crossover
        ]) 
        
        # maximum and average fitness of the children for adaptive mutation probability
        max_fitness = np.max(fitness_sum_children)
        average_fitness = np.mean(fitness_sum_children)
       
        
        for i in range(len(new_population_after_crossover)):
            # Calculate mutation probability based on the fitness
            mutation_probability = 0.1 * ((max_fitness - fitness_sum_children[i]) / ((max_fitness - average_fitness) + 1e-6))

            # Perform mutation on the individual
            mutated = mutation(new_population_after_crossover[i], mutation_probability, number_rus)
            # Add mutated individual to the next population
            next_population.append(mutated)
        
        # Add the best individual from this generation to ensure elitism
        next_population.append(best_individual_before_selection)

        # --- Update population for next generation ---
        population = next_population

    # --- Return best solution from the final generation ---
    # Compute fitness for all individuals in the final population
    fitness_list = np.array([
        compute_fitness(ind, number_rus, G, P_per_RB_W, noise_W)[1] 
        for ind in population
    ])
    # Find index of the best individual in the final population
    best_idx = int(np.argmax(fitness_list))
    # Get the fitness of the best individual
    overall_best_individual_fitness = fitness_list[best_idx]
    
    return best_idx, overall_best_individual_fitness


def round_robin_scheduling(number_rus, number_ues, number_rbs, rsrp_dBm, G, P_per_RB_W, noise_W):
    """
    Implements round-robin scheduling as a baseline.
    
    Algorithm:
    1. Associate each UE to RU with best RSRP 
    2. For each RU, cycle through its associated UEs and PRBs in a round robin fashion to allocate PRBs.
    
    Args:
        number_rus: number of RUs
        number_ues: number of UEs
        number_rbs: number of RBs
        rsrp_dBm: RSRP values (M, K)
        G: channel gain tensor (M, K, N, T)
        P_per_RB_W: transmit power per RB (W)
        noise_W: noise power (W)

    Returns:
        fitness_value_rr: scalar, total system throughput
    """
    
    UE_RU = np.zeros((number_ues, number_rus), dtype=int)
    for k in range(number_ues):
        # Find RU with highest RSRP for UE k. rsrp_dBm shape: (M, K)
        best_ru = np.argmax(rsrp_dBm[:, k])    
        # Associate UE k to RU with highest RSRP
        UE_RU[k, best_ru] = 1               
      
    # UE–RB allocation matrix
    UE_RB = np.zeros((number_ues, number_rbs), dtype=int)   
    
    # --- Round Robin per RU ---
    for m in range(number_rus):
        # check which UEs are associated to RU m
        associated_ues = [k for k in range(number_ues) if UE_RU[k, m] == 1]   
        if not associated_ues: 
            continue
        num_associated = len(associated_ues)
        num_iterations = max(number_rbs, num_associated)    

        # ---Allocate RBs in round-robin fashion---
        for i in range(num_iterations):      
            #Cycle through associated UEs           
            ue_idx = associated_ues[i % num_associated]  
            #Cycle through RBs   
            rb_idx = i % number_rbs   
            #Allocate RB to UE                      
            UE_RB[ue_idx, rb_idx] = 1

    # --- Create allocation matrix and compute throughput ---
    round_robin_matrix = np.concatenate((UE_RU, UE_RB), axis=1)
    # Compute total system throughput for round robin scheduling
    fitness_value_rr = compute_fitness(
        individual=round_robin_matrix,
        number_rus=number_rus,
        Gain=G,
        P_per_RB_W=P_per_RB_W,
        noise_W=noise_W
    )[1]                
    
    return fitness_value_rr





# --- Main simulation execution block ---
if __name__ == "__main__":

    import config
    import propagation_environment as pe
    import network_layout
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    
    number_rbs = config.NUM_RBS        
    number_ttis = config.NUM_TTIS       
    noise_W = config.noise_W
    RB_bw = config.RB_bw

    # To store the capacity improvements for the different scenarios
    system_capacity_improvement = {
        "umi": {},
        "indoor": {}
    }
    
    # Main loop over environments, RU and UE configurations, and seeds
    for env, cfg in config.ENV_CONFIG.items():
        if env == "umi":
            P_per_RB_W = config.P_per_RB_W_UMI
        else:
            P_per_RB_W = config.P_per_RB_W_INDOOR
        
        for number_rus, number_ues in zip(cfg["NUM_RUS"], cfg["NUM_UES"]):
            for seed in config.SEEDS:

                #--- Set random seeds for reproducibility ---
                np.random.seed(seed)
                random.seed(seed)
                               
                # --- Generate network layout and distances ---
                ru_x, ru_y, ue_x, ue_y, distances = network_layout.get_distances(env, number_rus, number_ues)
                
                # ---compute the channel gain which is a function of path loss and multipath fading ---
                h, PL_linear = pe.build_channel_gain_tensor(
                    number_rus, number_ues, number_rbs, number_ttis, env, distances
                )

                # --- compute channel power gains (linear) from amplitude h ---
                G = (h.astype(np.float64))**2 * PL_linear[:, :, None, None]  # shape (M,K,N,T)

                # ---Compute RSRP values---
                rsrp_dBm = pe.compute_rsrp(G, P_per_RB_W)
                
                # --- Run Genetic Algorithm ---
                best_population_index, fitness_value = run_ga(
                    generations=8,
                    population_size=12,
                    number_ues=number_ues,
                    number_rus=number_rus,
                    number_rbs=number_rbs,
                    rsrp_dBm=rsrp_dBm,
                    G=G,
                    P_per_RB_W=P_per_RB_W,
                    noise_W=noise_W,
                    RB_bw=RB_bw
                )
                
                # --- Run Round Robin for comparison ---
                fitness_value_rr = round_robin_scheduling(
                    number_rus=number_rus,
                    number_ues=number_ues,
                    number_rbs=number_rbs,
                    rsrp_dBm=rsrp_dBm,
                    G=G,
                    P_per_RB_W=P_per_RB_W,
                    noise_W=noise_W
                    
                )

                # --- Compute capacity improvement ---
                capacity_improvement = (((fitness_value - fitness_value_rr) / fitness_value_rr) * 100)
                           
                # --- Store results ---
                key = (number_rus, number_ues)
                if key not in system_capacity_improvement[env]:
                    system_capacity_improvement[env][key] = []

                # --- Store gain for this seed ---
                system_capacity_improvement[env][key].append(capacity_improvement)
                

# --- Plot CDFs of capacity improvements ---
#for env in system_capacity_improvement.keys():
for env_idx, env in enumerate(system_capacity_improvement.keys()):
    # Create subplot for each environment
    plt.subplot(1, 2, env_idx + 1)

    # Define line styles based on number of RUs
    line_style_map = {
        3: '-',      # Solid line for 3 RUs
        4: '-',      # Solid line for 4 RUs (indoor)
        8: '--',      # Dashed line for 8 RUs
        12: ':'     # Dotted line for 12 RUs
    }

    # for each (RU, UE) configuration, plot the CDF of capacity improvements
    for (num_rus, num_ues), gains in system_capacity_improvement[env].items():

        capacity_gains = np.array(gains)

        if len(capacity_gains) == 0:
            continue

        # Sort capacity gains for CDF computation
        capacity_gains_sorted = np.sort(capacity_gains)
        # Compute CDF values
        cdf = np.arange(1, len(capacity_gains_sorted) + 1) / len(capacity_gains_sorted)

        # Get line style based on number of RUs
        line_style = line_style_map.get(num_rus, '-')

        label = f"RUs={num_rus}, UEs={num_ues}"
        plt.plot(capacity_gains_sorted, cdf, label=label, linestyle=line_style, linewidth=2)

    plt.grid(True, alpha=0.3)
    plt.xlabel("System Capacity Improvement (%)", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.title(env.capitalize(), fontsize=14)
    plt.legend(fontsize=10, loc='best')
plt.tight_layout()
plt.show()




        
