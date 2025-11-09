"""
Unit tests for training_setup.py

Tests the core remapping logic and training run preparation.
"""

import pytest
import pickle
import tempfile
from pathlib import Path

from flowrl.parallel.training_setup import (
    build_remapping,
    prepare_training_run,
    generate_module_with_remapping,
    load_expert_params,
    load_expert_metadata,
)


class TestBuildRemapping:
    """Test the build_remapping function."""

    def test_simple_remapping_contiguous_indices(self):
        """Test remapping with contiguous global expert indices."""
        # Setup: experts 0, 1 already exist, adding expert 2
        completed_skills = {
            "Collect_Wood": {"expert_idx": 0},
            "Collect_Stone": {"expert_idx": 1},
        }
        dependencies = ["Collect_Wood", "Collect_Stone"]
        new_expert_idx = 2

        # Execute
        g2l, l2g = build_remapping(dependencies, new_expert_idx, completed_skills)

        # Verify
        assert g2l == {0: 0, 1: 1, 2: 2}
        assert l2g == {0: 0, 1: 1, 2: 2}

    def test_sparse_remapping_non_contiguous_indices(self):
        """Test remapping with sparse global expert indices."""
        # Setup: experts 0, 5 exist, adding expert 7
        # This should remap to local indices 0, 1, 2
        completed_skills = {
            "Collect_Wood": {"expert_idx": 0},
            "Collect_Iron": {"expert_idx": 5},
        }
        dependencies = ["Collect_Wood", "Collect_Iron"]
        new_expert_idx = 7

        # Execute
        g2l, l2g = build_remapping(dependencies, new_expert_idx, completed_skills)

        # Verify
        assert g2l == {0: 0, 5: 1, 7: 2}
        assert l2g == {0: 0, 1: 5, 2: 7}

    def test_single_dependency(self):
        """Test remapping with only one dependency."""
        completed_skills = {
            "Collect_Wood": {"expert_idx": 0},
        }
        dependencies = ["Collect_Wood"]
        new_expert_idx = 1

        g2l, l2g = build_remapping(dependencies, new_expert_idx, completed_skills)

        assert g2l == {0: 0, 1: 1}
        assert l2g == {0: 0, 1: 1}

    def test_no_dependencies(self):
        """Test remapping with no dependencies (first skill)."""
        completed_skills = {}
        dependencies = []
        new_expert_idx = 0

        g2l, l2g = build_remapping(dependencies, new_expert_idx, completed_skills)

        assert g2l == {0: 0}
        assert l2g == {0: 0}

    def test_missing_dependency_raises_error(self):
        """Test that missing dependency raises ValueError."""
        completed_skills = {
            "Collect_Wood": {"expert_idx": 0},
        }
        dependencies = ["Collect_Wood", "Nonexistent_Skill"]
        new_expert_idx = 2

        with pytest.raises(ValueError, match="not found in completed_skills"):
            build_remapping(dependencies, new_expert_idx, completed_skills)

    def test_remapping_maintains_ordering(self):
        """Test that remapping creates contiguous local indices regardless of global order."""
        # Dependencies specified in non-sorted order
        completed_skills = {
            "Skill_A": {"expert_idx": 10},
            "Skill_B": {"expert_idx": 3},
            "Skill_C": {"expert_idx": 7},
        }
        dependencies = ["Skill_A", "Skill_B", "Skill_C"]
        new_expert_idx = 15

        g2l, l2g = build_remapping(dependencies, new_expert_idx, completed_skills)

        # Should sort by global index: 3, 7, 10, 15 → 0, 1, 2, 3
        assert g2l == {3: 0, 7: 1, 10: 2, 15: 3}
        assert l2g == {0: 3, 1: 7, 2: 10, 3: 15}


class TestModuleGeneration:
    """Test module generation with remapping."""

    def test_generate_module_with_remapping(self):
        """Test that generated module includes remapping metadata."""
        skill_name = "Make_Pickaxe"
        skill_data = {
            "code": "def task_0_reward(state):\n    return 1.0\n"
        }
        global_to_local = {0: 0, 1: 1, 2: 2}
        local_to_global = {0: 0, 1: 1, 2: 2}
        global_expert_idx = 2

        module_content = generate_module_with_remapping(
            skill_name,
            skill_data,
            global_to_local,
            local_to_global,
            global_expert_idx
        )

        # Verify header
        assert f"# Auto-generated training module for: {skill_name}" in module_content
        assert f"# Global expert index: {global_expert_idx}" in module_content

        # Verify remapping dicts
        assert "GLOBAL_TO_LOCAL = {0: 0, 1: 1, 2: 2}" in module_content
        assert "LOCAL_TO_GLOBAL = {0: 0, 1: 1, 2: 2}" in module_content

        # Verify LLM code is included
        assert "def task_0_reward(state):" in module_content

    def test_sparse_remapping_in_module(self):
        """Test module generation with sparse remapping."""
        skill_data = {
            "code": "# Task logic here\n"
        }
        global_to_local = {0: 0, 5: 1, 10: 2}
        local_to_global = {0: 0, 1: 5, 2: 10}

        module_content = generate_module_with_remapping(
            "Test_Skill",
            skill_data,
            global_to_local,
            local_to_global,
            10
        )

        assert "GLOBAL_TO_LOCAL = {0: 0, 5: 1, 10: 2}" in module_content
        assert "LOCAL_TO_GLOBAL = {0: 0, 1: 5, 2: 10}" in module_content

    def test_missing_code_raises_error(self):
        """Test that missing code in skill_data raises ValueError."""
        skill_data = {}  # No 'code' key
        global_to_local = {0: 0}
        local_to_global = {0: 0}

        with pytest.raises(ValueError, match="No LLM-generated code found"):
            generate_module_with_remapping(
                "Test_Skill",
                skill_data,
                global_to_local,
                local_to_global,
                0
            )


class TestPrepareTrainingRun:
    """Test the full prepare_training_run function."""

    def setup_method(self):
        """Setup test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)

        # Create skills/ directory structure with dummy experts
        self.setup_completed_skills()

    def setup_completed_skills(self):
        """Create dummy completed skills with expert networks."""
        skills_dir = self.base_dir / "skills"

        # Skill 0: Collect_Wood
        skill_0_dir = skills_dir / "0_Collect_Wood"
        expert_0_dir = skill_0_dir / "expert_0_policy"
        expert_0_dir.mkdir(parents=True, exist_ok=True)

        expert_0_data = {
            "params": {"actor": "dummy_wood_params"},
            "metadata": {
                "skill_name": "Collect_Wood",
                "global_expert_idx": 0,
                "total_frames": 50_000_000,
            }
        }
        with open(expert_0_dir / "params.pkl", 'wb') as f:
            pickle.dump(expert_0_data, f)

        # Skill 1: Collect_Stone
        skill_1_dir = skills_dir / "1_Collect_Stone"
        expert_1_dir = skill_1_dir / "expert_1_policy"
        expert_1_dir.mkdir(parents=True, exist_ok=True)

        expert_1_data = {
            "params": {"actor": "dummy_stone_params"},
            "metadata": {
                "skill_name": "Collect_Stone",
                "global_expert_idx": 1,
                "total_frames": 40_000_000,
            }
        }
        with open(expert_1_dir / "params.pkl", 'wb') as f:
            pickle.dump(expert_1_data, f)

    def test_prepare_simple_training_run(self):
        """Test preparing a training run with contiguous dependencies."""
        skill_name = "Make_Pickaxe"
        skill_data = {
            "code": "def task_0_reward(state):\n    return 1.0\n",
            "skill_with_consumption": {
                "gain": {"pickaxe": 1},
                "requirements": {"wood": 1, "stone": 1}
            }
        }
        global_expert_idx = 2
        dependencies = ["Collect_Wood", "Collect_Stone"]
        completed_skills = {
            "Collect_Wood": {"expert_idx": 0},
            "Collect_Stone": {"expert_idx": 1},
        }

        # Execute
        run_folder, module_path, policies_folder = prepare_training_run(
            skill_name,
            skill_data,
            global_expert_idx,
            dependencies,
            completed_skills,
            self.base_dir,
            initialize_expert_fn=lambda: {"actor": "new_expert_init"}
        )

        # Verify folder structure
        assert run_folder.exists()
        assert module_path.exists()
        assert policies_folder.exists()

        # Verify module content
        with open(module_path, 'r') as f:
            module_content = f.read()
        assert "GLOBAL_TO_LOCAL = {0: 0, 1: 1, 2: 2}" in module_content
        assert "LOCAL_TO_GLOBAL = {0: 0, 1: 1, 2: 2}" in module_content
        assert "def task_0_reward(state):" in module_content

        # Verify checkpoint
        checkpoint_path = policies_folder / "checkpoint_0.pkl"
        assert checkpoint_path.exists()

        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        # Check expert params
        assert "expert_0" in checkpoint["expert_params"]
        assert "expert_1" in checkpoint["expert_params"]
        assert "expert_2" in checkpoint["expert_params"]

        # Check remapping metadata
        metadata = checkpoint["remapping_metadata"]
        assert metadata["global_to_local"] == {0: 0, 1: 1, 2: 2}
        assert metadata["local_to_global"] == {0: 0, 1: 1, 2: 2}
        assert metadata["initial_frame_counts"] == {0: 50_000_000, 1: 40_000_000, 2: 0}
        assert metadata["skill_name"] == "Make_Pickaxe"
        assert metadata["global_expert_idx"] == 2

    def test_prepare_sparse_training_run(self):
        """Test preparing a training run with sparse expert indices."""
        # Create expert 5 (skipping 2-4)
        skills_dir = self.base_dir / "skills"
        skill_5_dir = skills_dir / "5_Collect_Iron"
        expert_5_dir = skill_5_dir / "expert_5_policy"
        expert_5_dir.mkdir(parents=True, exist_ok=True)

        expert_5_data = {
            "params": {"actor": "dummy_iron_params"},
            "metadata": {
                "skill_name": "Collect_Iron",
                "global_expert_idx": 5,
                "total_frames": 60_000_000,
            }
        }
        with open(expert_5_dir / "params.pkl", 'wb') as f:
            pickle.dump(expert_5_data, f)

        # Prepare training for expert 7 with dependencies on 0 and 5
        skill_name = "Smelt_Iron"
        skill_data = {
            "code": "def task_0_reward(state):\n    return 1.0\n",
            "skill_with_consumption": {}
        }
        global_expert_idx = 7
        dependencies = ["Collect_Wood", "Collect_Iron"]
        completed_skills = {
            "Collect_Wood": {"expert_idx": 0},
            "Collect_Iron": {"expert_idx": 5},
        }

        # Execute
        run_folder, module_path, policies_folder = prepare_training_run(
            skill_name,
            skill_data,
            global_expert_idx,
            dependencies,
            completed_skills,
            self.base_dir,
            initialize_expert_fn=lambda: {"actor": "new_expert_init"}
        )

        # Verify remapping is contiguous
        with open(module_path, 'r') as f:
            module_content = f.read()

        # Global indices 0, 5, 7 should map to local 0, 1, 2
        assert "GLOBAL_TO_LOCAL = {0: 0, 5: 1, 7: 2}" in module_content
        assert "LOCAL_TO_GLOBAL = {0: 0, 1: 5, 2: 7}" in module_content

        # Verify checkpoint has 3 experts (not 8)
        checkpoint_path = policies_folder / "checkpoint_0.pkl"
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        assert len(checkpoint["expert_params"]) == 3
        assert "expert_0" in checkpoint["expert_params"]
        assert "expert_1" in checkpoint["expert_params"]
        assert "expert_2" in checkpoint["expert_params"]

        # Frame counts should use global indices
        metadata = checkpoint["remapping_metadata"]
        assert metadata["initial_frame_counts"] == {0: 50_000_000, 5: 60_000_000, 7: 0}

    def test_no_dependencies_first_skill(self):
        """Test preparing first skill with no dependencies."""
        skill_name = "Explore"
        skill_data = {
            "code": "def task_0_reward(state):\n    return 1.0\n",
            "skill_with_consumption": {}
        }
        global_expert_idx = 0
        dependencies = []
        completed_skills = {}

        run_folder, module_path, policies_folder = prepare_training_run(
            skill_name,
            skill_data,
            global_expert_idx,
            dependencies,
            completed_skills,
            self.base_dir,
            initialize_expert_fn=lambda: {"actor": "new_expert_init"}
        )

        # Should have only 1 expert
        with open(module_path, 'r') as f:
            module_content = f.read()

        assert "GLOBAL_TO_LOCAL = {0: 0}" in module_content
        assert "LOCAL_TO_GLOBAL = {0: 0}" in module_content

        checkpoint_path = policies_folder / "checkpoint_0.pkl"
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        assert len(checkpoint["expert_params"]) == 1
        assert "expert_0" in checkpoint["expert_params"]


class TestExpertLoading:
    """Test expert loading utilities."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.expert_dir = Path(self.temp_dir) / "expert_0_policy"
        self.expert_dir.mkdir(parents=True)

        # Create dummy expert
        expert_data = {
            "params": {"actor": "test_params", "critic": "test_critic"},
            "metadata": {
                "skill_name": "Test_Skill",
                "global_expert_idx": 0,
                "total_frames": 100_000_000,
            }
        }
        with open(self.expert_dir / "params.pkl", 'wb') as f:
            pickle.dump(expert_data, f)

    def test_load_expert_params(self):
        """Test loading expert parameters."""
        params = load_expert_params(self.expert_dir)

        assert params == {"actor": "test_params", "critic": "test_critic"}

    def test_load_expert_metadata(self):
        """Test loading expert metadata."""
        metadata = load_expert_metadata(self.expert_dir)

        assert metadata["skill_name"] == "Test_Skill"
        assert metadata["global_expert_idx"] == 0
        assert metadata["total_frames"] == 100_000_000

    def test_load_nonexistent_expert_raises_error(self):
        """Test that loading nonexistent expert raises FileNotFoundError."""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"

        with pytest.raises(FileNotFoundError):
            load_expert_params(nonexistent_dir)

        with pytest.raises(FileNotFoundError):
            load_expert_metadata(nonexistent_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
