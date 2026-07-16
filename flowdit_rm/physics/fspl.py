import numpy as np
import matplotlib.pyplot as plt

from flowdit_rm.physics.tx_detector import detect_tx_from_png

def generate_3d_fspl_tensor(
    shape: tuple,
    coords: np.ndarray,
    powers: np.ndarray,
    terrain_yx: np.ndarray,
    rx_heights_agl: list = [1.5, 30.0, 200.0],
    #freq_mhz: float = 150.0,
    pixel_res_m: float = 10.0,
    tx_height_agl: float = 1.5,
    dynamic_range_db: float = 100.0, # Receive adaptive gamma value from the caller
) -> np.ndarray:
    """
    Generate a realistic 3D FSPL physical-prior tensor using DEM terrain elevation and multiple height layers.
    
    Args:
        shape: target matrix shape (H, W)
        coords: Transmittercoordinate数组，shape (N, 2)，format (y, x)
        powers: 每transmitters的峰值功率
        terrain_yx: terrain elevation matrix，shape (H, W)
        rx_heights_agl: receiver above-ground height list to generate (例如 1.5m, 30m, 200m)
        freq_mhz: signal frequency (MHz)
        pixel_res_m: real horizontal physical distance represented by each pixel (米)
        tx_height_agl: Transmitter离地高度 (米)
        
    Returns:
        fspl_tensor: shape (len(rx_heights_agl), H, W) 3D theoretical signal-strength tensor
    """
    H, W = shape
    num_layers = len(rx_heights_agl)
    
    # Initialize output tensor (3 channels, H, W) to a very small value
    fspl_tensor = np.full((num_layers, H, W), -150.0)
    
    # Note translated to English.
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    
    for (tx_y, tx_x), p_tx in zip(coords, powers):
        # Note translated to English.
        z_tx_absolute = terrain_yx[tx_y, tx_x] + tx_height_agl
        
        # 2. Compute 2D horizontal physical distance (meters)
        dist_m_2d = np.sqrt((y_grid - tx_y)**2 + (x_grid - tx_x)**2) * pixel_res_m
        
        # 3. Iterate over three receiver height slices (1.5 m, 30 m, 200 m)
        for i, h_rx in enumerate(rx_heights_agl):
            # Absolute receiver-elevation matrix over the full image
            z_rx_absolute = terrain_yx + h_rx
            
            # Compute true 3D height difference (meters)
            height_diff_m = z_tx_absolute - z_rx_absolute
            
            # Note translated to English.
            dist_m_3d = np.sqrt(dist_m_2d**2 + height_diff_m**2)
            
            # ---------------------------------------------------------
            # Core correction: anchor-based relative path-loss model
            # ---------------------------------------------------------
            # Set reference distance d_ref to one pixel width in physical units, e.g. 10 meters
            d_ref = pixel_res_m 
            
            # Clamp minimum distance to d_ref to remove singularities and anchor attenuation
            effective_dist = np.maximum(dist_m_3d, d_ref)
            
            # Directly compute relative attenuation, avoiding dependence on frequency f and constant 32.44
            relative_loss = 20 * np.log10(effective_dist / d_ref)
            
            # # Start from extracted peak power and subtract relative attenuation
            # p_rx = p_tx - relative_loss
            # # ---------------------------------------------------------
            
            # # Multi-source fusion by taking maximum signal strength
            # fspl_tensor[i] = np.maximum(fspl_tensor[i], p_rx)
            # ---------------------------------------------------------
            # Core scale alignment: map dB attenuation to [0, 255] pixel space
            # ---------------------------------------------------------
            #dynamic_range_db = 100.0  # Assume 100 dB attenuation spans the brightest to darkest image values
            
            # Convert attenuation into pixel-domain units; 100 dB corresponds to 255 pixel levels
            # Use the adaptive dynamic_range_db value passed by the caller
            pixel_loss = relative_loss / dynamic_range_db
            
            # 2. Compute the final prior map in [0, 255] and clamp the minimum to 0
            p_rx = np.maximum(p_tx - pixel_loss, 0.0)
            # ---------------------------------------------------------
            
            # Multi-source fusion by taking maximum signal strength
            fspl_tensor[i] = np.maximum(fspl_tensor[i], p_rx)
            
    return fspl_tensor

# ================= Simulated test call =================
if __name__ == "__main__":
    # 1. Set the input image path; replace with a real radio-map path
    sample_img_path = "/workspace/PPData5D-success/png/06.DenseUrban/T06C0D0002_n00_f00_ss_z00.png"

    # Note translated to English.
    print("Calling tx_detector for blind extraction...")
    coords, powers, thresh, orig_matrix = detect_tx_from_png(
        png_path=sample_img_path,
        percentile=98.0,
        min_distance=10,
    )
    print(f"Detection complete. Automatic threshold: {thresh:.2f}, found {len(coords)} transmitters。")

    # 3. Extract terrain information
    terrain_path = "/workspace/PPData5D-success/npz/T06C0D0002_n00_bdtr.npz"
    terrain_data = np.load(terrain_path, allow_pickle=True)
    terrain_yx = terrain_data["terrain_yx"]

    # Extract a 3-layer FSPL tensor
    fspl_3d = generate_3d_fspl_tensor(
        shape=orig_matrix.shape,
        coords=coords,
        powers=powers,
        terrain_yx=terrain_yx,
        rx_heights_agl=[1.5, 30.0, 200.0]
    )
    
    # Note translated to English.
    heights = [1.5, 30.0, 200.0]
    global_min = float(np.min(fspl_3d))
    global_max = float(np.max(fspl_3d))

    for i, h in enumerate(heights):
        # 2. Force vmin and vmax so all three images share the same color scale
        im = axes[i + 2].imshow(
            fspl_3d[i], 
            cmap="jet", 
            vmin=global_min, 
            vmax=global_max
        )
        print(f"Saved {out_path}")