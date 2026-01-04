import numpy as np
import matplotlib.pyplot as plt
import math


def get_distances(environment, number_rus, number_ues):
    """
    This function generates RU and UE positions based on the specified environment ("umi" or "indoor"),
    and computes the distances between each RU and UE. A hexagonal layout is used for RUs in the "umi" environment,
    while a rectangular layout is used for the "indoor" environment.

    Parameters:
        environment (str): The environment type ("umi" or "indoor").
        number_rus (int): The number of RUs.
        number_ues (int): The number of UEs.

    Returns:
        tuple: Contains the x and y coordinates of RUs and UEs, and the distance matrix.
    """


    # -----------------------------
    # Generation of hexagonal RU layout for the urban micro (umi)
    # -----------------------------
    if environment == "umi":
        num_rings = math.ceil((-3 + math.sqrt(12 * number_rus - 3)) / 6)  # Calculate number of rings needed for the hexagon
        ISD = 200.0  # inter-site distance in meters
        dx = ISD * np.sqrt(3) / 2   
        dy = ISD

        ru_x = []
        ru_y = []

        # --- Generation of the full hex grid ---
        for row in range(-num_rings, num_rings + 1):
            for col in range(-num_rings, num_rings + 1):
                x = col * dx + (row % 2) * dx / 2      # offset for even rows
                y = row * dy                           # y coordinate
                ru_x.append(x)
                ru_y.append(y)

        ru_x = np.array(ru_x)
        ru_y = np.array(ru_y)

        # --- Sort by distance from origin (center first) ---
        r = np.sqrt(ru_x**2 + ru_y**2)
        idx = np.argsort(r)

        ru_x = ru_x[idx][:number_rus]
        ru_y = ru_y[idx][:number_rus]

        # --- Define area for UE placement ---
        # margin around the RU area for UE placement. ISD to simulate far edge UEs from the RUs
        margin = ISD                     
        xmin, xmax = ru_x.min() - margin, ru_x.max() + margin
        ymin, ymax = ru_y.min() - margin, ru_y.max() + margin

        # --- Random UE placement within the defined area ---
        ue_x = np.random.uniform(xmin, xmax, number_ues)
        ue_y = np.random.uniform(ymin, ymax, number_ues)


    # -----------------------------
    # Generation of the rectangular RU layout for indoor
    # -----------------------------
    elif environment == "indoor":
        Width = 240   # width of the rectangle in meters
        Height = 50   # height of the rectangle in meters
        ISD_indoor = 20  # intersite distance between RUs in meters

        # To calculate the total number of RUs in each dimension
        num_x = math.ceil(Width / ISD_indoor)
        num_y = math.ceil(Height / ISD_indoor)

        # Generate RU positions based on the total number of RUs needed in each dimension
        ru_x = np.arange(num_x) * ISD_indoor
        ru_y = np.arange(num_y) * ISD_indoor

        # Create all combinations of (x, y) and reshape to (num_rus, 2)
        ru_positions = np.array(np.meshgrid(ru_x, ru_y)).T.reshape(-1, 2)
    
        #Separate x and y coordinates
        ru_x = ru_positions[:, 0]
        ru_y = ru_positions[:, 1]

        # Sort by distance from origin (center first)  
        r = np.sqrt(ru_x**2 + ru_y**2)
        idx = np.argsort(r)

        # Grid generation may produce more RUs than needed. Select only the required number. Those closest to origin.
        ru_x = ru_x[idx][:number_rus]
        ru_y = ru_y[idx][:number_rus]

        # Define area for UE placement
        xmin, xmax = ru_x.min(), ru_x.max() 
        ymin, ymax = ru_y.min(), ru_y.max() 

        # --- Random UE placement within the defined area ---
        ue_x = np.random.uniform(xmin, xmax, number_ues)
        ue_y = np.random.uniform(ymin, ymax, number_ues)

        # --- Stack to get (num_ues, 2) array of UE positions ---
        ue_positions = np.vstack((ue_x, ue_y)).T

        # --- Unstack to get separate x/y arrays ---
        ue_x = ue_positions[:, 0] 
        ue_y = ue_positions[:, 1]

    # -----------------------------
    # Generation of the distance matrix
    # -----------------------------
    distances = np.zeros((number_rus, number_ues))
    for i in range(number_rus):
        for j in range(number_ues):
            distances[i, j] = np.sqrt((ru_x[i] - ue_x[j])**2 + (ru_y[i] - ue_y[j])**2)
    
    return ru_x, ru_y, ue_x, ue_y, distances



