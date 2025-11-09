"""
Parallel skill training components for Flow-RL.

This module provides infrastructure for training multiple skills in parallel:
- training_setup: Prepare training runs with network remapping
- training_processor: Post-training processing with frame-count heuristic
- scheduler: Tmux-based parallel skill orchestration
"""

from .training_setup import prepare_training_run, build_remapping
from .training_processor import process_completed_training
from .scheduler import SkillScheduler

__all__ = [
    'prepare_training_run',
    'build_remapping',
    'process_completed_training',
    'SkillScheduler',
]
