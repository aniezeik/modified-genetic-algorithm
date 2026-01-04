# Configuration parameters for the simulation

import numpy as np

# Simulation seeds for reproducibility
SEEDS = [int(s) for s in np.arange(1, 51, 1)]

# Environment-specific configurations for number of RUs and UEs
ENV_CONFIG = {
    "umi": {
        "NUM_RUS": [3, 3, 3, 8, 8, 8, 12, 12, 12],
        "NUM_UES": [9, 30, 90, 24, 80, 240, 36, 120, 360],
    },
    "indoor": {
        "NUM_RUS": [4, 4, 4, 8, 8, 8, 12, 12, 12],
        "NUM_UES": [9, 30, 90, 24, 80, 240, 36, 120, 360],
    }
}
''
# Transmit power and noise configurations
NUM_RBS = 10 
# Number of TTIs is 1
NUM_TTIS = 1
# Total transmit power in dBm for indoor
P_TX_TOTAL_DBM_INDOOR = 20.0        
P_TX_TOTAL_DBM_UMI = 33.0     
# Convert transmit power from dBm to Watts
P_TX_TOTAL_W_INDOOR = 10**((P_TX_TOTAL_DBM_INDOOR - 30) / 10)
P_TX_TOTAL_W_UMI = 10**((P_TX_TOTAL_DBM_UMI - 30) / 10)
# Power per Resource Block
P_per_RB_W_INDOOR = P_TX_TOTAL_W_INDOOR / NUM_RBS   
P_per_RB_W_UMI = P_TX_TOTAL_W_UMI / NUM_RBS 
# Resource Block bandwidth in Hz 
RB_bw = 180*1e3  
# Noise figure (NF) in dB
NF_dB = 9.0 
# Thermal noise power calculation 
kT_dBm_Hz = -174.0  
# Total noise power in dBm
noise_dBm_Hz = kT_dBm_Hz + NF_dB    
noise_dBm = noise_dBm_Hz + 10.0 * np.log10(RB_bw)  
# Convert noise power from dBm to Watts
noise_W = 10**((noise_dBm - 30.0) / 10.0)  





