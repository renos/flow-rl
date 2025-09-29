#!/usr/bin/env python3
"""
Generate example videos from hierarchical flow policies using gen_frames_hierarchical.
"""

import os
from flowrl.utils.test import gen_frames_hierarchical, render_video


def generate_policy_videos(
    policy_path: str,
    output_dir: str = "videos",
    max_frames: int = 2000,
    goal_states: list = None,
    num_videos: int = 3
):
    """
    Generate multiple example videos from a hierarchical flow policy.
    
    Args:
        policy_path: Path to trained policy directory
        output_dir: Directory to save videos
        max_frames: Maximum frames per video
        goal_states: List of goal states to attempt (if None, uses [5, 8, 11])
        num_videos: Number of videos to generate per goal state
    """
    if goal_states is None:
        goal_states = [5, 8, 11]  # Different difficulty levels
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating videos for policy: {policy_path}")
    print(f"Output directory: {output_dir}")
    print(f"Goal states: {goal_states}")
    print(f"Videos per goal state: {num_videos}")
    
    for goal_state in goal_states:
        print(f"\n=== Generating videos for goal state {goal_state} ===")
        
        for video_idx in range(num_videos):
            try:
                print(f"  Generating video {video_idx + 1}/{num_videos}...")
                
                # Generate frames using existing function
                frames, states, env_states, actions = gen_frames_hierarchical(
                    policy_path=policy_path,
                    max_num_frames=max_frames,
                    goal_state=goal_state,
                    num_envs=64  # More environments for higher success chance
                )
                
                if len(frames) == 0:
                    print(f"    Failed to generate trajectory for goal state {goal_state}")
                    continue
                
                # Create video filename
                policy_name = os.path.basename(policy_path.rstrip('/'))
                video_filename = f"{policy_name}_goal{goal_state}_video{video_idx + 1}.mp4"
                video_path = os.path.join(output_dir, video_filename)
                
                # Render video
                render_video(frames, states, video_path)
                
                print(f"    ✅ Saved video: {video_path}")
                print(f"       Frames: {len(frames)}, Max state: {max(states)}")
                
            except Exception as e:
                print(f"    ❌ Error generating video {video_idx + 1}: {e}")
                continue
    
    print(f"\n🎬 Video generation completed! Check {output_dir}/ for videos.")


def generate_comparison_videos(
    policy_paths: list,
    output_dir: str = "comparison_videos",
    goal_state: int = 8,
    max_frames: int = 2000
):
    """
    Generate comparison videos from multiple policies for the same goal state.
    
    Args:
        policy_paths: List of paths to trained policies
        output_dir: Directory to save videos
        goal_state: Target goal state for comparison
        max_frames: Maximum frames per video
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating comparison videos for goal state {goal_state}")
    print(f"Policies: {len(policy_paths)}")
    print(f"Output directory: {output_dir}")
    
    for i, policy_path in enumerate(policy_paths):
        try:
            print(f"\n--- Policy {i+1}/{len(policy_paths)}: {policy_path} ---")
            
            # Generate frames
            frames, states, env_states, actions = gen_frames_hierarchical(
                policy_path=policy_path,
                max_num_frames=max_frames,
                goal_state=goal_state,
                num_envs=64
            )
            
            if len(frames) == 0:
                print(f"❌ Failed to generate trajectory for {policy_path}")
                continue
            
            # Create video filename
            policy_name = os.path.basename(policy_path.rstrip('/'))
            video_filename = f"comparison_goal{goal_state}_{policy_name}.mp4"
            video_path = os.path.join(output_dir, video_filename)
            
            # Render video
            render_video(frames, states, video_path)
            
            print(f"✅ Saved: {video_path}")
            print(f"   Frames: {len(frames)}, Max state: {max(states)}")
            
        except Exception as e:
            print(f"❌ Error with {policy_path}: {e}")
            continue
    
    print(f"\n🎬 Comparison videos completed! Check {output_dir}/ for videos.")


def test_video_generation():
    """Test video generation with the saved policy."""
    policy_path = "/home/renos/flow-rl/exp/different_network/1_policies"
    
    print("=== Testing Video Generation ===")
    
    generate_policy_videos(
        policy_path=policy_path,
        output_dir="test_videos",
        max_frames=1500,  # Shorter for testing
        goal_states=[5, 8],  # Just two goal states for testing
        num_videos=2  # Just 2 videos per goal state
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate videos from hierarchical flow policies")
    parser.add_argument("--policy_path", type=str, required=True,
                       help="Path to trained policy directory")
    parser.add_argument("--output_dir", type=str, default="videos",
                       help="Directory to save videos")
    parser.add_argument("--max_frames", type=int, default=2000,
                       help="Maximum frames per video")
    parser.add_argument("--goal_states", type=int, nargs="+", default=[5, 8, 11],
                       help="Goal states to attempt")
    parser.add_argument("--num_videos", type=int, default=3,
                       help="Number of videos per goal state")
    parser.add_argument("--test", action="store_true",
                       help="Run test with predefined settings")
    
    args = parser.parse_args()
    
    if args.test:
        test_video_generation()
    else:
        generate_policy_videos(
            policy_path=args.policy_path,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            goal_states=args.goal_states,
            num_videos=args.num_videos
        )