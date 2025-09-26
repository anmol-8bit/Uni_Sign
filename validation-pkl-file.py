import os
import pickle
import numpy as np
import glob
from pathlib import Path
import sys

def validate_single_pkl(pkl_path):
    """Validate a single PKL file"""
    try:
        # Try to load the file
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if it has the required keys
        if not isinstance(data, dict):
            return False, "Not a dictionary"
        
        required_keys = ['keypoints', 'scores']
        for key in required_keys:
            if key not in data:
                return False, f"Missing key: {key}"
        
        keypoints = data['keypoints']
        scores = data['scores']
        
        # Check if keypoints and scores have same length
        if len(keypoints) != len(scores):
            return False, f"Keypoints ({len(keypoints)}) and scores ({len(scores)}) length mismatch"
        
        if len(keypoints) == 0:
            return False, "No frames found"
        
        # Check keypoints format for each frame
        for i, (kpts, scrs) in enumerate(zip(keypoints, scores)):
            # Convert to numpy if needed
            kpts = np.array(kpts)
            scrs = np.array(scrs)
            
            # Check if we have people detected
            if len(kpts.shape) == 3:  # [num_people, num_keypoints, 2]
                num_people, num_keypoints, coords = kpts.shape
                
                # Check expected format
                if coords != 2:
                    return False, f"Frame {i}: Expected 2 coordinates (x,y), got {coords}"
                
                if num_keypoints != 133:
                    return False, f"Frame {i}: Expected 133 keypoints, got {num_keypoints}"
                
                # Check scores shape matches
                if scrs.shape != (num_people, num_keypoints):
                    return False, f"Frame {i}: Scores shape {scrs.shape} doesn't match keypoints {kpts.shape[:2]}"
                
                # Check coordinate ranges (should be normalized 0-1)
                if np.any(kpts < -0.1) or np.any(kpts > 1.1):
                    return False, f"Frame {i}: Keypoints not properly normalized (should be 0-1 range)"
                
                # Check score ranges (should be 0-1)
                if np.any(scrs < 0) or np.any(scrs > 1):
                    return False, f"Frame {i}: Scores not in valid range (should be 0-1)"
            
            elif len(kpts.shape) == 2 and kpts.shape[0] == 0:
                # No people detected in this frame - this is valid
                continue
            else:
                return False, f"Frame {i}: Unexpected keypoints shape {kpts.shape}"
        
        return True, f"Valid - {len(keypoints)} frames, avg {np.mean([len(k) for k in keypoints]):.1f} people/frame"
        
    except Exception as e:
        return False, f"Error loading file: {str(e)}"

def validate_pose_directory(pose_dir, sample_size=100):
    """Validate multiple PKL files in a directory"""
    pkl_files = glob.glob(os.path.join(pose_dir, "*.pkl"))
    
    if len(pkl_files) == 0:
        print(f"No PKL files found in {pose_dir}")
        return
    
    print(f"Found {len(pkl_files)} PKL files in {pose_dir}")
    
    # Sample files if too many
    if len(pkl_files) > sample_size:
        import random
        pkl_files = random.sample(pkl_files, sample_size)
        print(f"Sampling {sample_size} files for validation")
    
    valid_count = 0
    invalid_count = 0
    errors = {}
    
    print("\nValidating files...")
    for i, pkl_file in enumerate(pkl_files):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(pkl_files)}")
        
        is_valid, message = validate_single_pkl(pkl_file)
        
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            filename = os.path.basename(pkl_file)
            errors[filename] = message
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files checked: {len(pkl_files)}")
    print(f"Valid files: {valid_count}")
    print(f"Invalid files: {invalid_count}")
    print(f"Success rate: {valid_count/len(pkl_files)*100:.1f}%")
    
    # Show errors
    if errors:
        print(f"\nERRORS FOUND:")
        for filename, error in list(errors.items())[:10]:  # Show first 10 errors
            print(f"  {filename}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more errors")
    
    return valid_count, invalid_count, errors

def detailed_file_inspection(pkl_path):
    """Detailed inspection of a single PKL file"""
    print(f"Detailed inspection of: {pkl_path}")
    print("="*50)
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"File size: {os.path.getsize(pkl_path) / 1024:.1f} KB")
        print(f"Data type: {type(data)}")
        print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        if 'keypoints' in data and 'scores' in data:
            keypoints = data['keypoints']
            scores = data['scores']
            
            print(f"Number of frames: {len(keypoints)}")
            
            # Analyze first few frames
            for i in range(min(3, len(keypoints))):
                kpts = np.array(keypoints[i])
                scrs = np.array(scores[i])
                
                print(f"\nFrame {i}:")
                print(f"  Keypoints shape: {kpts.shape}")
                print(f"  Scores shape: {scrs.shape}")
                
                if len(kpts.shape) == 3:
                    print(f"  Number of people: {kpts.shape[0]}")
                    print(f"  Keypoints per person: {kpts.shape[1]}")
                    print(f"  Coordinate range: [{kpts.min():.3f}, {kpts.max():.3f}]")
                    print(f"  Score range: [{scrs.min():.3f}, {scrs.max():.3f}]")
                    print(f"  Average confidence: {scrs.mean():.3f}")
                print("w_h : ", data["w_h"])
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Detailed inspection of specific file
        pkl_path = sys.argv[1]
        detailed_file_inspection(pkl_path)
    else:
        # Validate directory
        pose_dir = "/data_benchmark/Uni-Sign/dataset/ASL/pose_format"
        validate_pose_directory(pose_dir)