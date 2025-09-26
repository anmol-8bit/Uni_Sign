import os
from pathlib import Path
import sys
import time

def setup_pose_directories(base_path):
    """
    Create pose_asl directory structure mirroring the clips structure
    """
    clips_path = Path(base_path) / "clips"
    pose_path = Path(base_path) / "pose_asl"
    
    print(f"YouTube ASL Pose Directory Setup")
    print(f"================================")
    print(f"Source: {clips_path}")
    print(f"Target: {pose_path}")
    
    if not clips_path.exists():
        print(f"ERROR: Clips directory not found: {clips_path}")
        return False
    
    # Create base pose_asl directory
    pose_path.mkdir(exist_ok=True)
    print(f"✓ Created base directory: {pose_path}")
    
    # Get list of video directories first
    print("Scanning video directories...")
    video_dirs_list = [d for d in clips_path.iterdir() if d.is_dir()]
    print(f"Found {len(video_dirs_list)} video directories to process")
    
    # Process directories
    video_dirs = []
    total_clips = 0
    start_time = time.time()
    
    for i, video_id_dir in enumerate(video_dirs_list):
        video_clips_dir = video_id_dir / "clips"
        
        if video_clips_dir.exists():
            

            clip_files = list(video_clips_dir.glob("*.mp4"))
            clip_count = len(clip_files)
            total_clips += clip_count
            
            pose_video_dir = pose_path / video_id_dir.name / "clips"
            pose_video_dir.mkdir(parents=True, exist_ok=True)
            
            video_dirs.append({
                'video_id': video_id_dir.name,
                'clip_count': clip_count
            })
            
            # Progress update
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(video_dirs_list) - i - 1) / rate
                print(f"Progress: {i+1}/{len(video_dirs_list)} ({(i+1)/len(video_dirs_list)*100:.1f}%) "
                      f"- ETA: {remaining/60:.1f} min - Clips found: {total_clips}")
    
    elapsed = time.time() - start_time
    print(f"\n=== SETUP COMPLETE ===")
    print(f"Time taken: {elapsed/60:.1f} minutes")
    print(f"Video directories processed: {len(video_dirs)}")
    print(f"Total clips found: {total_clips:,}")
    print(f"Average clips per video: {total_clips/len(video_dirs):.1f}")
    
    # Show distribution
    clip_counts = [v['clip_count'] for v in video_dirs]
    print(f"Clip count distribution:")
    print(f"  Min: {min(clip_counts)}")
    print(f"  Max: {max(clip_counts)}")
    print(f"  Median: {sorted(clip_counts)[len(clip_counts)//2]}")
    
    # Show some examples
    print(f"\nSample video directories:")
    for i, vdir in enumerate(video_dirs[:5]):
        print(f"  {vdir['video_id']}: {vdir['clip_count']} clips")
    
    return video_dirs, total_clips

if __name__ == "__main__":
    base_path = "/data/youtube-asl-download"
    
    try:
        video_dirs, total_clips = setup_pose_directories(base_path)
        
        # Save detailed summary
        summary_file = "setup_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"YouTube ASL Directory Setup Summary\n")
            f.write(f"===================================\n")
            f.write(f"Base path: {base_path}\n")
            f.write(f"Video directories: {len(video_dirs):,}\n")
            f.write(f"Total clips: {total_clips:,}\n")
            f.write(f"Average clips per video: {total_clips/len(video_dirs):.1f}\n\n")
            
            # Statistics
            clip_counts = [v['clip_count'] for v in video_dirs]
            f.write(f"Clip count statistics:\n")
            f.write(f"  Min: {min(clip_counts)}\n")
            f.write(f"  Max: {max(clip_counts)}\n")
            f.write(f"  Median: {sorted(clip_counts)[len(clip_counts)//2]}\n\n")
            
            f.write("All video directories:\n")
            for vdir in video_dirs:
                f.write(f"{vdir['video_id']}: {vdir['clip_count']} clips\n")
        
        print(f"\n✓ Detailed summary saved to: {summary_file}")
        
        # Save a quick video list for processing
        video_list_file = "video_directories.txt"
        with open(video_list_file, "w") as f:
            for vdir in video_dirs:
                f.write(f"{vdir['video_id']}\n")
        
        print(f"✓ Video directory list saved to: {video_list_file}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)