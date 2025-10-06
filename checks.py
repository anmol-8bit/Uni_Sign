import pickle
import numpy as np

def check_pkl_normalization(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    keypoints = data['keypoints']
    if len(keypoints) == 0:
        return None
        
    print(f"File: {pkl_path}")
    print(f"Number of frames: {len(keypoints)}")
    
    # Get all coordinates
    all_coords = np.concatenate(keypoints, axis=0)  # Shape: (frames, 133, 2)
    print(f"All coords shape: {all_coords.shape}")
    
    # Extract x,y coordinates (last dimension is 2: x, y)
    coords_xy = all_coords  # Shape: (frames, 133, 2)
    
    # Remove zero coordinates - check if both x and y are zero
    # valid_mask should have shape (frames, 133)
    valid_mask = (coords_xy[:, :, 0] != 0) | (coords_xy[:, :, 1] != 0)  # Shape: (frames, 133)
    valid_coords = coords_xy[valid_mask]  # This will flatten to (N, 2)
    
    if len(valid_coords) == 0:
        print("No valid coordinates found")
        return None
        
    min_coords = valid_coords.min()
    max_coords = valid_coords.max()
    
    print(f"Min: {min_coords:.3f}, Max: {max_coords:.3f}")
    print(f"Image-relative (0-1): {max_coords <= 1.0 and min_coords >= 0.0}")
    print(f"Pose-relative (-1 to 1): {max_coords <= 1.0 and min_coords >= -1.0}")
    print("=" * 50)


# Check a few files from each dataset
check_pkl_normalization("dataset/CSL_News/pose_format/Dragon-TV_20240319_48337-48687_213183.pkl")
check_pkl_normalization("dataset/ASL/pose_format/--6bmFM9wT4_0001_006.170-011.420_Hello_everyone._Welcome_to_Sign1News._I_.pkl")