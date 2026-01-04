import numpy as np


def shadow_fading(sigma_db):
    return np.random.normal(0, sigma_db)


def path_loss_inh_nlos(d_m, f_ghz = 0.9, ht_ut=1.5):
    """
    Calculate path loss for indoor non-line-of-sight (nLOS) scenario.
    Args:
        d_m (np.ndarray): Distance in meters    
        f_ghz (float): Carrier Frequency in GHz (900 MHz)
        ht_ut (float): Height of the User Terminal in meters
    """

    # Derive the 3D distance. Height of the UE is 1.5m and height of the RU is 3m
    ht_ru = 3.0
    d_m = np.sqrt(d_m ** 2 + (ht_ru - ht_ut) ** 2)

    # Path Loss calculation for nLOS according to ITU-R guidelines
    PL_dB = 43.3 * np.log10(d_m) + 11.5 + 20 * np.log10(f_ghz)  
    shadow = shadow_fading(3)
    PL_dB += shadow
    return PL_dB


def path_loss_umi_nlos(d_m, f_ghz = 0.9, ht_ut=1.5):
    """
    Calculate path loss for urban micro non-line-of-sight (nLOS) scenario.

    Args:
        d_m (np.ndarray): Distance in meters    
        f_ghz (float): Carrier Frequency in GHz (900 MHz)
        ht_ut (float): Height of the User Terminal in meters
    """ 
    # Derive the 3D distance. 
    d_m = np.sqrt(d_m ** 2 + (15 - ht_ut) ** 2)

    # Path Loss calculation
    PL_dB = 36.7 * np.log10(d_m) + 22.7 + 26 * np.log10(f_ghz) - 0.3 * (ht_ut - 1.5)
    shadow = shadow_fading(4)
    PL_dB += shadow

    return PL_dB


def build_channel_gain_tensor(number_rus, number_ues, number_rbs, number_ttis, env, distances):
    """"
    Build the channel gain tensor for the given environment and parameters.

    Args:
        number_rus (int): Number of RUs
        number_ues (int): Number of UEs
        number_rbs (int): Number of RBs
        number_ttis (int): Number of TTIs
        env (str): Environment type ("indoor" or "umi")
        distances (np.ndarray): Array of distances between RUs and UEs
        
    Returns:
        h (np.ndarray): Channel gain tensor
        PL_linear (np.ndarray): Path loss in linear scale

    """

    # Calculate path loss based on environment
    if env == "indoor":
        PL_dB = path_loss_inh_nlos(distances)
    else:
        PL_dB = path_loss_umi_nlos(distances)

    PL_linear = 10 ** (-PL_dB / 10.0)

    # Initialize channel gain tensor
    h = np.zeros((number_rus, number_ues, number_rbs, number_ttis), dtype=np.float32)

    # Generate small-scale fading (multipath) using rayleigh fading model
    if env == "indoor":

        # Generate Rayleigh fading for each RU-UE link. Reuse the same fading across RBs to estimate flat fading.
        h_indoor = np.abs(np.sqrt(0.5) * (np.random.randn(number_rus,number_ues) + 1j * np.random.randn(number_rus,number_ues)))
        h = h_indoor[:, :, None, None] 
        
    elif env == "umi":      
        # Generate rayleigh fading for each RU-UE link and each RB. Assume flat fading within each RB and add diversity across RBs.
        h_umi = np.abs(np.sqrt(0.5) * (np.random.randn(number_rus, number_ues, number_rbs) + 1j * np.random.randn(number_rus, number_ues, number_rbs)))
        h = h_umi[:, :, :, None] 
       
    return h, PL_linear


def compute_rsrp(G,P_per_RB_W):
    """
    Estimate the Reference Signal Received Power (RSRP).

    Args:
        G (np.ndarray): Channel gain (shape: M,K,N,T)
        P_per_RB_W (np.ndarray): Power per Resource Block (RB) (shape: M,K,N,T)

    Returns:
        rsrp_dBm (np.ndarray): RSRP in dBm (shape: M,K,N)
    """
    received_power = P_per_RB_W * G  # shape (M,K,N,T)

    # Average over RBs since number of TTIs is 1. Simulation uses RBs to estimate the RSRP over using the power received by the reference signal in the resource elements.
    rsrp_W = np.mean(received_power, axis=2)    # shape (M,K)
    rsrp_dBm = 10 * np.log10(rsrp_W ) + 30
    return rsrp_dBm