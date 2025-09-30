#!/usr/bin/env python3
"""
Quick single-file pose validation
Usage: python quick_pose_check.py video.mp4 pose.pkl
"""

import cv2
import pickle
import numpy as np
import sys
import os

def quick_pose_check(video_path, pkl_path, output_path=None):
    """Quick validation of a single video-pose pair"""
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"quick_check_{base_name}.mp4"
    
    print(f"🎯 Quick Pose Check")
    print(f"   Video: {video_path}")
    print(f"   PKL: {pkl_path}")
    print(f"   Output: {output_path}")
    
    # Load pose data
    try:
        with open(pkl_path, 'rb') as f:
            pose_data = pickle.load(f)
        print(f"✅ PKL loaded: {len(pose_data['keypoints'])} frames")
    except Exception as e:
        print(f"❌ Error loading PKL: {e}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video")
        return False
    
    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height}, {fps}fps, {total_frames} frames")
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    pose_frames = len(pose_data['keypoints'])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw pose if available
        if frame_idx < pose_frames:
            keypoints = pose_data['keypoints'][frame_idx]
            scores = pose_data['scores'][frame_idx]
            
            # Handle multiple people
            if len(keypoints.shape) == 3:
                keypoints = keypoints[0]  # First person
                scores = scores[0]
            
            # Draw keypoints
            for i, (kpt, score) in enumerate(zip(keypoints, scores)):
                if score > 0.3:  # Confidence threshold
                    x = int(kpt[0] * width)
                    y = int(kpt[1] * height)
                    
                    # Color coding
                    if i < 17:  # Body
                        color = (0, 255, 0)
                    elif i < 91:  # Hands
                        color = (255, 0, 0)
                    else:  # Face
                        color = (0, 0, 255)
                    
                    cv2.circle(frame, (x, y), 2, color, -1)
            
            # Add info
            cv2.putText(frame, f"Frame: {frame_idx+1}/{pose_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Keypoints: {len(keypoints)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"   Progress: {frame_idx}/{total_frames}")
    
    cap.release()
    out.release()
    
    print(f"✅ Quick check completed: {output_path}")
    return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python quick_pose_check.py video.mp4 pose.pkl")
        print("Example: python quick_pose_check.py sample.mp4 sample.pkl")
        return
    
    video_path = sys.argv[1]
    pkl_path = sys.argv[2]
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    if not os.path.exists(pkl_path):
        print(f"❌ PKL file not found: {pkl_path}")
        return
    
    quick_pose_check(video_path, pkl_path)

if __name__ == "__main__":
    main()
