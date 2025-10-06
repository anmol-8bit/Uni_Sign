#!/usr/bin/env python3
"""
Video and Pose Analysis Tool
Analyzes MP4 and PKL files to extract statistics and export to CSV
"""

import cv2
import pickle
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoPoseAnalyzer:
    """Analyzes video and pose files to extract comprehensive statistics"""
    
    def __init__(self, base_dir: str = "./dataset"):
        self.base_dir = Path(base_dir)
        self.results = []
        
    def analyze_video(self, video_path: Path) -> Dict:
        """Extract statistics from MP4 video file"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return None
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Get file size
            file_size = video_path.stat().st_size
            
            cap.release()
            
            return {
                'file_path': str(video_path),
                'file_name': video_path.name,
                'file_type': 'video',
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration_seconds': duration,
                'file_size_bytes': file_size,
                'file_size_mb': file_size / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video {video_path}: {e}")
            return None
    
    def analyze_pose(self, pkl_path: Path) -> Dict:
        """Extract statistics from PKL pose file"""
        try:
            with open(pkl_path, 'rb') as f:
                pose_data = pickle.load(f)
            
            # Extract basic info
            keypoints = pose_data.get('keypoints', [])
            scores = pose_data.get('scores', [])
            
            num_frames = len(keypoints)
            
            # Calculate pose statistics
            pose_stats = self._calculate_pose_statistics(keypoints, scores)
            
            # Get file size
            file_size = pkl_path.stat().st_size
            
            return {
                'file_path': str(pkl_path),
                'file_name': pkl_path.name,
                'file_type': 'pose',
                'num_frames': num_frames,
                'file_size_bytes': file_size,
                'file_size_mb': file_size / (1024 * 1024),
                **pose_stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pose {pkl_path}: {e}")
            return None
    
    def _calculate_pose_statistics(self, keypoints: List, scores: List) -> Dict:
        """Calculate detailed pose statistics"""
        if not keypoints or not scores:
            return {}
        
        # Flatten all keypoints and scores across frames
        all_keypoints = []
        all_scores = []
        
        for frame_kpts, frame_scores in zip(keypoints, scores):
            if len(frame_kpts.shape) == 3:  # Multiple people
                frame_kpts = frame_kpts[0]  # First person
                frame_scores = frame_scores[0]
            
            all_keypoints.append(frame_kpts)
            all_scores.append(frame_scores)
        
        all_keypoints = np.concatenate(all_keypoints, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        
        # Calculate statistics
        num_keypoints = all_keypoints.shape[1] if len(all_keypoints.shape) > 1 else 0
        
        # Confidence statistics
        valid_scores = all_scores[all_scores > 0]
        avg_confidence = np.mean(valid_scores) if len(valid_scores) > 0 else 0
        min_confidence = np.min(valid_scores) if len(valid_scores) > 0 else 0
        max_confidence = np.max(valid_scores) if len(valid_scores) > 0 else 0
        
        # Keypoint visibility
        visible_keypoints = np.sum(all_scores > 0.3)  # Threshold for visibility
        total_keypoints = len(all_scores)
        visibility_ratio = visible_keypoints / total_keypoints if total_keypoints > 0 else 0
        
        # Movement analysis (if multiple frames)
        movement_stats = self._analyze_movement(keypoints)
        
        return {
            'num_keypoints': num_keypoints,
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'visible_keypoints': visible_keypoints,
            'visibility_ratio': visibility_ratio,
            **movement_stats
        }
    
    def _analyze_movement(self, keypoints: List) -> Dict:
        """Analyze movement patterns in keypoints"""
        if len(keypoints) < 2:
            return {'movement_analysis': 'insufficient_frames'}
        
        # Calculate movement between consecutive frames
        movements = []
        for i in range(1, len(keypoints)):
            prev_kpts = keypoints[i-1]
            curr_kpts = keypoints[i]
            
            if len(prev_kpts.shape) == 3:
                prev_kpts = prev_kpts[0]
                curr_kpts = curr_kpts[0]
            
            # Calculate Euclidean distance for each keypoint
            diff = curr_kpts - prev_kpts
            distances = np.sqrt(np.sum(diff**2, axis=1))
            movements.extend(distances)
        
        if not movements:
            return {'movement_analysis': 'no_movement_data'}
        
        movements = np.array(movements)
        
        return {
            'avg_movement': np.mean(movements),
            'max_movement': np.max(movements),
            'movement_std': np.std(movements),
            'movement_analysis': 'completed'
        }
    
    def find_matching_files(self) -> List[Tuple[Path, Optional[Path]]]:
        """Find matching video and pose file pairs"""
        video_files = []
        pose_files = []
        
        # Find all video files
        for ext in ['*.mp4', '*.avi', '*.mov']:
            video_files.extend(self.base_dir.rglob(ext))
        
        # Find all pose files
        pose_files.extend(self.base_dir.rglob('*.pkl'))
        
        # Create mapping of base names
        video_map = {}
        for video in video_files:
            base_name = video.stem
            video_map[base_name] = video
        
        pose_map = {}
        for pose in pose_files:
            base_name = pose.stem
            pose_map[base_name] = pose
        
        # Find matches
        matches = []
        for base_name in video_map:
            video_path = video_map[base_name]
            pose_path = pose_map.get(base_name)
            matches.append((video_path, pose_path))
        
        # Add unmatched pose files
        for base_name in pose_map:
            if base_name not in video_map:
                matches.append((None, pose_map[base_name]))
        
        return matches
    
    def analyze_all(self) -> pd.DataFrame:
        """Analyze all video and pose files"""
        logger.info(f"Scanning directory: {self.base_dir}")
        
        matches = self.find_matching_files()
        logger.info(f"Found {len(matches)} file pairs/singles")
        
        all_results = []
        
        for video_path, pose_path in matches:
            # Analyze video
            if video_path:
                video_stats = self.analyze_video(video_path)
                if video_stats:
                    all_results.append(video_stats)
                    logger.info(f"Analyzed video: {video_path.name}")
            
            # Analyze pose
            if pose_path:
                pose_stats = self.analyze_pose(pose_path)
                if pose_stats:
                    all_results.append(pose_stats)
                    logger.info(f"Analyzed pose: {pose_path.name}")
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        if not df.empty:
            # Add dataset information
            df['dataset'] = df['file_path'].apply(self._extract_dataset_name)
            
            # Sort by dataset and file type
            df = df.sort_values(['dataset', 'file_type', 'file_name'])
        
        return df
    
    def _extract_dataset_name(self, file_path: str) -> str:
        """Extract dataset name from file path"""
        path_parts = Path(file_path).parts
        for part in path_parts:
            if part in ['CSL_News', 'CSL_Daily', 'WLASL']:
                return part
        return 'unknown'
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics from the analysis"""
        if df.empty:
            return {}
        
        summary = {}
        
        # Video statistics
        video_df = df[df['file_type'] == 'video']
        if not video_df.empty:
            summary['video_stats'] = {
                'total_videos': len(video_df),
                'total_duration_seconds': video_df['duration_seconds'].sum(),
                'avg_duration_seconds': video_df['duration_seconds'].mean(),
                'total_frames': video_df['total_frames'].sum(),
                'avg_frames_per_video': video_df['total_frames'].mean(),
                'resolution_distribution': video_df.groupby(['width', 'height']).size().to_dict(),
                'fps_distribution': video_df['fps'].value_counts().to_dict()
            }
        
        # Pose statistics
        pose_df = df[df['file_type'] == 'pose']
        if not pose_df.empty:
            summary['pose_stats'] = {
                'total_pose_files': len(pose_df),
                'total_pose_frames': pose_df['num_frames'].sum(),
                'avg_frames_per_pose': pose_df['num_frames'].mean(),
                'avg_confidence': pose_df['avg_confidence'].mean(),
                'avg_visibility_ratio': pose_df['visibility_ratio'].mean(),
                'keypoint_distribution': pose_df['num_keypoints'].value_counts().to_dict()
            }
        
        # Dataset breakdown
        summary['dataset_breakdown'] = df.groupby('dataset').size().to_dict()
        
        return summary
    
    def export_to_csv(self, df: pd.DataFrame, output_path: str = "video_pose_analysis.csv"):
        """Export analysis results to CSV"""
        df.to_csv(output_path, index=False)
        logger.info(f"Results exported to: {output_path}")
    
    def export_summary(self, summary: Dict, output_path: str = "analysis_summary.json"):
        """Export summary statistics to JSON"""
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary exported to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze video and pose files')
    parser.add_argument('--base_dir', default='./dataset', 
                       help='Base directory containing dataset folders')
    parser.add_argument('--output_csv', default='video_pose_analysis.csv',
                       help='Output CSV file path')
    parser.add_argument('--output_summary', default='analysis_summary.json',
                       help='Output summary JSON file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if base directory exists
    if not os.path.exists(args.base_dir):
        logger.error(f"Base directory does not exist: {args.base_dir}")
        logger.info("Please ensure the dataset directory exists with the following structure:")
        logger.info("  ./dataset/CSL_News/rgb_format/")
        logger.info("  ./dataset/CSL_News/pose_format/")
        logger.info("  ./dataset/CSL_Daily/rgb_format/")
        logger.info("  ./dataset/CSL_Daily/pose_format/")
        logger.info("  ./dataset/WLASL/rgb_format/")
        logger.info("  ./dataset/WLASL/pose_format/")
        return
    
    # Run analysis
    analyzer = VideoPoseAnalyzer(args.base_dir)
    df = analyzer.analyze_all()
    
    if df.empty:
        logger.warning("No files found to analyze")
        return
    
    # Generate summary
    summary = analyzer.generate_summary_stats(df)
    
    # Export results
    analyzer.export_to_csv(df, args.output_csv)
    analyzer.export_summary(summary, args.output_summary)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*50)
    
    if 'video_stats' in summary:
        vs = summary['video_stats']
        logger.info(f"Videos: {vs['total_videos']} files, {vs['total_duration_seconds']:.1f}s total")
        logger.info(f"Average duration: {vs['avg_duration_seconds']:.1f}s")
        logger.info(f"Total frames: {vs['total_frames']}")
    
    if 'pose_stats' in summary:
        ps = summary['pose_stats']
        logger.info(f"Pose files: {ps['total_pose_files']} files, {ps['total_pose_frames']} frames")
        logger.info(f"Average confidence: {ps['avg_confidence']:.3f}")
        logger.info(f"Average visibility: {ps['avg_visibility_ratio']:.3f}")
    
    logger.info(f"Dataset breakdown: {summary.get('dataset_breakdown', {})}")

if __name__ == "__main__":
    main()
