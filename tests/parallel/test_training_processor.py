"""
Unit tests for training_processor.py

Tests the frame-count heuristic and checkpoint merging logic.
"""

import pytest
import pickle
import tempfile
from pathlib import Path

from flowrl.parallel.training_processor import (
    process_completed_training,
    save_expert_to_skills,
    load_global_checkpoint,
    save_global_checkpoint,
)


class TestFrameCountHeuristic:
    """Test the frame-count heuristic for resolving conflicts."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)

        # Create initial completed skills
        self.setup_initial_skills()

    def setup_initial_skills(self):
        """Create initial skills with expert networks."""
        skills_dir = self.base_dir / "skills"

        # Skill 0: Collect_Wood (50M frames)
        skill_0_dir = skills_dir / "0_Collect_Wood"
        expert_0_dir = skill_0_dir / "expert_0_policy"
        expert_0_dir.mkdir(parents=True, exist_ok=True)

        expert_0_data = {
            "params": {"actor": "wood_params_v1", "version": 1},
            "metadata": {
                "skill_name": "Collect_Wood",
                "global_expert_idx": 0,
                "total_frames": 50_000_000,
            }
        }
        with open(expert_0_dir / "params.pkl", 'wb') as f:
            pickle.dump(expert_0_data, f)

        # Skill 1: Collect_Stone (40M frames)
        skill_1_dir = skills_dir / "1_Collect_Stone"
        expert_1_dir = skill_1_dir / "expert_1_policy"
        expert_1_dir.mkdir(parents=True, exist_ok=True)

        expert_1_data = {
            "params": {"actor": "stone_params_v1", "version": 1},
            "metadata": {
                "skill_name": "Collect_Stone",
                "global_expert_idx": 1,
                "total_frames": 40_000_000,
            }
        }
        with open(expert_1_dir / "params.pkl", 'wb') as f:
            pickle.dump(expert_1_data, f)

        # Create global checkpoint
        global_ckpt = {
            "skills": {
                "Collect_Wood": {
                    "expert_idx": 0,
                    "path": "skills/0_Collect_Wood/",
                    "total_frames": 50_000_000,
                },
                "Collect_Stone": {
                    "expert_idx": 1,
                    "path": "skills/1_Collect_Stone/",
                    "total_frames": 40_000_000,
                }
            },
            "db": {}
        }
        save_global_checkpoint(global_ckpt, self.base_dir)

    def create_training_run(self, skill_name, global_expert_idx, expert_params,
                           global_to_local, local_to_global, initial_frame_counts,
                           total_timesteps):
        """Helper to create a completed training run."""
        run_folder = self.base_dir / "training_runs" / f"skill_{global_expert_idx}_{skill_name}"
        run_folder.mkdir(parents=True, exist_ok=True)

        policies_folder = run_folder / f"{global_expert_idx}_policies"
        policies_folder.mkdir(parents=True, exist_ok=True)

        # Create final checkpoint
        checkpoint = {
            "expert_params": expert_params,
            "remapping_metadata": {
                "global_to_local": global_to_local,
                "local_to_global": local_to_global,
                "initial_frame_counts": initial_frame_counts,
                "skill_name": skill_name,
                "global_expert_idx": global_expert_idx,
            },
            "skill_with_consumption": {"gain": {"test": 1}},
            "total_timesteps": total_timesteps,
        }

        with open(policies_folder / "checkpoint_final.pkl", 'wb') as f:
            pickle.dump(checkpoint, f)

        return run_folder

    def test_new_skill_saves_expert(self):
        """Test that a new skill's expert is saved correctly."""
        # Create training run for new skill (expert 2)
        expert_params = {
            "expert_0": {"actor": "wood_params_v1"},  # Loaded from skill 0
            "expert_1": {"actor": "stone_params_v1"},  # Loaded from skill 1
            "expert_2": {"actor": "pickaxe_params_new", "version": 1},  # New
        }

        run_folder = self.create_training_run(
            skill_name="Make_Pickaxe",
            global_expert_idx=2,
            expert_params=expert_params,
            global_to_local={0: 0, 1: 1, 2: 2},
            local_to_global={0: 0, 1: 1, 2: 2},
            initial_frame_counts={0: 50_000_000, 1: 40_000_000, 2: 0},
            total_timesteps=80_000_000
        )

        # Process
        results = process_completed_training(
            run_folder=run_folder,
            skill_name="Make_Pickaxe",
            global_expert_idx=2,
            base_dir=self.base_dir,
            use_file_lock=False
        )

        # Verify new expert was saved
        assert results["expert_updates"][2]["action"] == "new"
        assert results["expert_updates"][2]["frames"] == 80_000_000

        # Check file exists
        expert_2_path = self.base_dir / "skills" / "2_Make_Pickaxe" / "expert_2_policy" / "params.pkl"
        assert expert_2_path.exists()

        with open(expert_2_path, 'rb') as f:
            saved_expert = pickle.load(f)

        assert saved_expert["params"]["actor"] == "pickaxe_params_new"
        assert saved_expert["metadata"]["total_frames"] == 80_000_000

    def test_expert_updated_when_more_frames(self):
        """Test that expert is updated when new version has more frames."""
        # Train skill that updates expert 0 from 50M to 150M frames
        expert_params = {
            "expert_0": {"actor": "wood_params_v2", "version": 2},  # Updated version
            "expert_1": {"actor": "stone_params_v1"},  # Unchanged
            "expert_2": {"actor": "pickaxe_params_new"},  # New
        }

        run_folder = self.create_training_run(
            skill_name="Make_Pickaxe",
            global_expert_idx=2,
            expert_params=expert_params,
            global_to_local={0: 0, 1: 1, 2: 2},
            local_to_global={0: 0, 1: 1, 2: 2},
            initial_frame_counts={0: 50_000_000, 1: 40_000_000, 2: 0},
            total_timesteps=100_000_000  # All 3 experts trained for 100M frames
        )

        # Process
        results = process_completed_training(
            run_folder=run_folder,
            skill_name="Make_Pickaxe",
            global_expert_idx=2,
            base_dir=self.base_dir,
            use_file_lock=False
        )

        # Expert 0: 50M + 100M = 150M (should update)
        assert results["expert_updates"][0]["action"] == "updated"
        assert results["expert_updates"][0]["frames"] == 150_000_000

        # Expert 1: 40M + 100M = 140M (should update)
        assert results["expert_updates"][1]["action"] == "updated"
        assert results["expert_updates"][1]["frames"] == 140_000_000

        # Expert 2: new
        assert results["expert_updates"][2]["action"] == "new"

        # Verify expert 0 was actually updated on disk
        expert_0_path = self.base_dir / "skills" / "0_Collect_Wood" / "expert_0_policy" / "params.pkl"
        with open(expert_0_path, 'rb') as f:
            saved_expert = pickle.load(f)

        assert saved_expert["params"]["version"] == 2  # New version
        assert saved_expert["metadata"]["total_frames"] == 150_000_000

    def test_expert_kept_when_fewer_frames(self):
        """Test that existing expert is kept when new version has fewer frames."""
        # Simulate parallel training where another skill already updated expert 0 to 180M
        global_ckpt = load_global_checkpoint(self.base_dir)
        global_ckpt["skills"]["Collect_Wood"]["total_frames"] = 180_000_000
        save_global_checkpoint(global_ckpt, self.base_dir)

        # Also update the saved expert to reflect this
        expert_0_path = self.base_dir / "skills" / "0_Collect_Wood" / "expert_0_policy" / "params.pkl"
        with open(expert_0_path, 'rb') as f:
            expert_0_data = pickle.load(f)
        expert_0_data["params"]["version"] = 3  # Even newer version
        expert_0_data["metadata"]["total_frames"] = 180_000_000
        with open(expert_0_path, 'wb') as f:
            pickle.dump(expert_0_data, f)

        # Now process a training run that only got expert 0 to 150M
        expert_params = {
            "expert_0": {"actor": "wood_params_v2", "version": 2},
            "expert_1": {"actor": "stone_params_v1"},
            "expert_2": {"actor": "sword_params_new"},
        }

        run_folder = self.create_training_run(
            skill_name="Make_Sword",
            global_expert_idx=3,
            expert_params=expert_params,
            global_to_local={0: 0, 1: 1, 3: 2},
            local_to_global={0: 0, 1: 1, 2: 3},
            initial_frame_counts={0: 50_000_000, 1: 40_000_000, 3: 0},
            total_timesteps=100_000_000
        )

        # Process
        results = process_completed_training(
            run_folder=run_folder,
            skill_name="Make_Sword",
            global_expert_idx=3,
            base_dir=self.base_dir,
            use_file_lock=False
        )

        # Expert 0: 50M + 100M = 150M < 180M (should keep existing)
        assert results["expert_updates"][0]["action"] == "kept"
        assert results["expert_updates"][0]["frames"] == 180_000_000

        # Verify expert 0 still has version 3
        with open(expert_0_path, 'rb') as f:
            saved_expert = pickle.load(f)
        assert saved_expert["params"]["version"] == 3
        assert saved_expert["metadata"]["total_frames"] == 180_000_000

    def test_parallel_conflict_resolution(self):
        """Test realistic parallel training conflict scenario."""
        # Two skills train in parallel, both update shared expert 0

        # Skill A finishes first: 50M → 180M (trained 130M frames)
        expert_params_a = {
            "expert_0": {"actor": "wood_from_skill_a", "trained_frames": 130_000_000},
            "expert_1": {"actor": "stone_v1"},
            "expert_2": {"actor": "pickaxe_new"},
        }

        run_folder_a = self.create_training_run(
            skill_name="Make_Pickaxe",
            global_expert_idx=2,
            expert_params=expert_params_a,
            global_to_local={0: 0, 1: 1, 2: 2},
            local_to_global={0: 0, 1: 1, 2: 2},
            initial_frame_counts={0: 50_000_000, 1: 40_000_000, 2: 0},
            total_timesteps=130_000_000
        )

        # Process skill A
        results_a = process_completed_training(
            run_folder=run_folder_a,
            skill_name="Make_Pickaxe",
            global_expert_idx=2,
            base_dir=self.base_dir,
            use_file_lock=False
        )

        assert results_a["expert_updates"][0]["action"] == "updated"
        assert results_a["expert_updates"][0]["frames"] == 180_000_000

        # Skill B finishes second: 50M → 160M (trained 110M frames)
        # This should NOT overwrite expert 0 because 160M < 180M
        expert_params_b = {
            "expert_0": {"actor": "wood_from_skill_b", "trained_frames": 110_000_000},
            "expert_2": {"actor": "iron_from_b"},
            "expert_3": {"actor": "sword_new"},
        }

        run_folder_b = self.create_training_run(
            skill_name="Make_Sword",
            global_expert_idx=3,
            expert_params=expert_params_b,
            global_to_local={0: 0, 2: 1, 3: 2},
            local_to_global={0: 0, 1: 2, 2: 3},
            initial_frame_counts={0: 50_000_000, 2: 80_000_000, 3: 0},
            total_timesteps=110_000_000
        )

        # Process skill B
        results_b = process_completed_training(
            run_folder=run_folder_b,
            skill_name="Make_Sword",
            global_expert_idx=3,
            base_dir=self.base_dir,
            use_file_lock=False
        )

        # Expert 0: kept at 180M (from skill A)
        assert results_b["expert_updates"][0]["action"] == "kept"
        assert results_b["expert_updates"][0]["frames"] == 180_000_000

        # Verify expert 0 still has params from skill A
        expert_0_path = self.base_dir / "skills" / "0_Collect_Wood" / "expert_0_policy" / "params.pkl"
        with open(expert_0_path, 'rb') as f:
            saved_expert = pickle.load(f)
        assert saved_expert["params"]["actor"] == "wood_from_skill_a"


class TestArchivalAndCleanup:
    """Test archival and cleanup functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)

    def test_archives_training_artifacts(self):
        """Test that training artifacts are archived to skills/ folder."""
        # Create training run
        run_folder = self.base_dir / "training_runs" / "skill_0_Test_Skill"
        run_folder.mkdir(parents=True, exist_ok=True)

        policies_folder = run_folder / "0_policies"
        policies_folder.mkdir(parents=True, exist_ok=True)

        # Create artifacts
        (run_folder / "0.py").write_text("# Module code")
        (run_folder / "training.log").write_text("Training logs")
        (run_folder / "video.mp4").write_text("fake video")

        # Create checkpoint
        checkpoint = {
            "expert_params": {"expert_0": {"actor": "params"}},
            "remapping_metadata": {
                "global_to_local": {0: 0},
                "local_to_global": {0: 0},
                "initial_frame_counts": {0: 0},
                "skill_name": "Test_Skill",
                "global_expert_idx": 0,
            },
            "total_timesteps": 10_000_000,
        }

        with open(policies_folder / "checkpoint_final.pkl", 'wb') as f:
            pickle.dump(checkpoint, f)

        # Process
        process_completed_training(
            run_folder=run_folder,
            skill_name="Test_Skill",
            global_expert_idx=0,
            base_dir=self.base_dir,
            use_file_lock=False
        )

        # Verify artifacts were archived
        skill_folder = self.base_dir / "skills" / "0_Test_Skill"
        assert (skill_folder / "0.py").exists()
        assert (skill_folder / "training.log").exists()
        assert (skill_folder / "video.mp4").exists()

        # Verify training_run was cleaned up
        assert not run_folder.exists()


class TestExpertSaving:
    """Test save_expert_to_skills utility."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)

    def test_save_expert_creates_structure(self):
        """Test that save_expert_to_skills creates correct folder structure."""
        expert_params = {"actor": "test_params", "critic": "test_critic"}

        expert_folder = save_expert_to_skills(
            global_expert_idx=5,
            skill_name="Test Skill",
            expert_params=expert_params,
            total_frames=100_000_000,
            base_dir=self.base_dir
        )

        # Verify folder structure
        assert expert_folder.exists()
        assert expert_folder.name == "expert_5_policy"
        assert (expert_folder / "params.pkl").exists()

        # Verify saved data
        with open(expert_folder / "params.pkl", 'rb') as f:
            saved_data = pickle.load(f)

        assert saved_data["params"] == expert_params
        assert saved_data["metadata"]["skill_name"] == "Test Skill"
        assert saved_data["metadata"]["global_expert_idx"] == 5
        assert saved_data["metadata"]["total_frames"] == 100_000_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
