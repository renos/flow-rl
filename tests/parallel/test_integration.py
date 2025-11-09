"""
Integration test demonstrating the full parallel training workflow.

This shows how all components work together:
1. training_setup: Prepare training runs with remapping
2. scheduler: Orchestrate parallel training
3. training_processor: Merge results with frame-count heuristic
"""

import pytest
import pickle
import tempfile
from pathlib import Path

from flowrl.parallel.training_setup import prepare_training_run, build_remapping
from flowrl.parallel.training_processor import process_completed_training
from flowrl.parallel.scheduler import SkillScheduler


class TestParallelWorkflow:
    """
    Integration test showing realistic parallel training scenario.

    Scenario:
    - 3 skills: Collect_Wood, Collect_Stone, Make_Pickaxe
    - Collect_Wood and Collect_Stone can run in parallel (no deps)
    - Make_Pickaxe depends on both (waits for them)
    """

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)

    def dummy_expert_init(self):
        """Initialize dummy expert params."""
        return {"actor": "dummy_init", "critic": "dummy_init"}

    def simulate_training(self, run_folder, expert_idx, total_timesteps):
        """
        Simulate training by creating a final checkpoint.

        In real training, ppo_flow.py would do this.
        """
        policies_folder = run_folder / f"{expert_idx}_policies"

        # Load initial checkpoint
        initial_ckpt_path = policies_folder / "checkpoint_0.pkl"
        with open(initial_ckpt_path, 'rb') as f:
            checkpoint = pickle.load(f)

        # "Train" - just update version numbers to show it changed
        expert_params = checkpoint["expert_params"]
        for key, params in expert_params.items():
            if params is not None:
                params["trained"] = True
                params["timesteps"] = total_timesteps

        # Save final checkpoint
        checkpoint["total_timesteps"] = total_timesteps
        with open(policies_folder / "checkpoint_final.pkl", 'wb') as f:
            pickle.dump(checkpoint, f)

        # Create dummy log
        (run_folder / "training.log").write_text(f"Trained for {total_timesteps} timesteps\n")

    def test_sequential_skills(self):
        """Test simple sequential execution (no parallelism)."""
        scheduler = SkillScheduler(base_dir=self.base_dir, max_parallel=1)
        completed_skills = {}

        # Skill 1: Collect_Wood (no dependencies)
        skill_1_data = {
            "code": "def task_0_reward(state): return 1.0",
            "skill_with_consumption": {"gain": {"wood": 1}}
        }

        print("\n=== Preparing Skill 1: Collect_Wood ===")
        run_folder_1, module_path_1, policies_folder_1 = prepare_training_run(
            skill_name="Collect_Wood",
            skill_data=skill_1_data,
            global_expert_idx=0,
            dependency_skill_names=[],
            completed_skills={},
            base_dir=self.base_dir,
            initialize_expert_fn=self.dummy_expert_init
        )

        # Simulate training
        print("Simulating training...")
        self.simulate_training(run_folder_1, 0, 50_000_000)

        # Process
        print("Processing results...")
        results_1 = process_completed_training(
            run_folder=run_folder_1,
            skill_name="Collect_Wood",
            global_expert_idx=0,
            base_dir=self.base_dir,
            use_file_lock=False
        )

        # Verify expert 0 was saved
        assert results_1["expert_updates"][0]["action"] == "new"
        assert results_1["expert_updates"][0]["frames"] == 50_000_000

        # Add to completed
        completed_skills["Collect_Wood"] = {
            "expert_idx": 0,
            "path": "skills/0_Collect_Wood/"
        }

        # Skill 2: Collect_Stone (no dependencies)
        skill_2_data = {
            "code": "def task_0_reward(state): return 1.0",
            "skill_with_consumption": {"gain": {"stone": 1}}
        }

        print("\n=== Preparing Skill 2: Collect_Stone ===")
        run_folder_2, module_path_2, policies_folder_2 = prepare_training_run(
            skill_name="Collect_Stone",
            skill_data=skill_2_data,
            global_expert_idx=1,
            dependency_skill_names=[],
            completed_skills=completed_skills,
            base_dir=self.base_dir,
            initialize_expert_fn=self.dummy_expert_init
        )

        # Simulate and process
        self.simulate_training(run_folder_2, 1, 40_000_000)
        results_2 = process_completed_training(
            run_folder=run_folder_2,
            skill_name="Collect_Stone",
            global_expert_idx=1,
            base_dir=self.base_dir,
            use_file_lock=False
        )

        assert results_2["expert_updates"][1]["action"] == "new"

        completed_skills["Collect_Stone"] = {
            "expert_idx": 1,
            "path": "skills/1_Collect_Stone/"
        }

        # Skill 3: Make_Pickaxe (depends on both)
        skill_3_data = {
            "code": "def task_0_reward(state): return 1.0",
            "skill_with_consumption": {
                "gain": {"pickaxe": 1},
                "requirements": {"wood": 1, "stone": 1}
            }
        }

        print("\n=== Preparing Skill 3: Make_Pickaxe ===")
        run_folder_3, module_path_3, policies_folder_3 = prepare_training_run(
            skill_name="Make_Pickaxe",
            skill_data=skill_3_data,
            global_expert_idx=2,
            dependency_skill_names=["Collect_Wood", "Collect_Stone"],
            completed_skills=completed_skills,
            base_dir=self.base_dir,
            initialize_expert_fn=self.dummy_expert_init
        )

        # Verify remapping
        with open(module_path_3, 'r') as f:
            module_content = f.read()
        assert "GLOBAL_TO_LOCAL = {0: 0, 1: 1, 2: 2}" in module_content

        # Verify checkpoint has all 3 experts
        with open(policies_folder_3 / "checkpoint_0.pkl", 'rb') as f:
            checkpoint = pickle.load(f)
        assert len(checkpoint["expert_params"]) == 3
        assert "expert_0" in checkpoint["expert_params"]
        assert "expert_1" in checkpoint["expert_params"]
        assert "expert_2" in checkpoint["expert_params"]

        # Simulate training (all experts get trained for 100M)
        self.simulate_training(run_folder_3, 2, 100_000_000)

        # Process
        results_3 = process_completed_training(
            run_folder=run_folder_3,
            skill_name="Make_Pickaxe",
            global_expert_idx=2,
            base_dir=self.base_dir,
            use_file_lock=False
        )

        # Expert 0: 50M + 100M = 150M (updated)
        assert results_3["expert_updates"][0]["action"] == "updated"
        assert results_3["expert_updates"][0]["frames"] == 150_000_000

        # Expert 1: 40M + 100M = 140M (updated)
        assert results_3["expert_updates"][1]["action"] == "updated"
        assert results_3["expert_updates"][1]["frames"] == 140_000_000

        # Expert 2: new
        assert results_3["expert_updates"][2]["action"] == "new"

        print("\n=== Workflow Complete ===")
        print("Final skills:")
        for skill_name, skill_info in completed_skills.items():
            print(f"  - {skill_name} (expert {skill_info['expert_idx']})")

    def test_sparse_expert_indices(self):
        """Test that sparse expert indices are remapped to contiguous."""
        completed_skills = {}

        # Create skills with non-contiguous expert indices
        # Expert 0, 5, 10 → should remap to 0, 1, 2

        # Skill with expert 0
        skill_0_data = {"code": "test", "skill_with_consumption": {}}
        run_0, _, _ = prepare_training_run(
            "Skill_0", skill_0_data, 0, [], {}, self.base_dir, self.dummy_expert_init
        )
        self.simulate_training(run_0, 0, 10_000_000)
        process_completed_training(run_0, "Skill_0", 0, self.base_dir, use_file_lock=False)
        completed_skills["Skill_0"] = {"expert_idx": 0}

        # Skill with expert 5 (skipping 1-4)
        skill_5_data = {"code": "test", "skill_with_consumption": {}}
        run_5, _, _ = prepare_training_run(
            "Skill_5", skill_5_data, 5, [], completed_skills, self.base_dir, self.dummy_expert_init
        )
        self.simulate_training(run_5, 5, 10_000_000)
        process_completed_training(run_5, "Skill_5", 5, self.base_dir, use_file_lock=False)
        completed_skills["Skill_5"] = {"expert_idx": 5}

        # Skill with expert 10 that depends on 0 and 5
        skill_10_data = {"code": "test", "skill_with_consumption": {}}
        run_10, module_10, policies_10 = prepare_training_run(
            "Skill_10", skill_10_data, 10, ["Skill_0", "Skill_5"],
            completed_skills, self.base_dir, self.dummy_expert_init
        )

        # Verify remapping: global {0, 5, 10} → local {0, 1, 2}
        with open(module_10, 'r') as f:
            content = f.read()
        assert "GLOBAL_TO_LOCAL = {0: 0, 5: 1, 10: 2}" in content
        assert "LOCAL_TO_GLOBAL = {0: 0, 1: 5, 2: 10}" in content

        # Verify checkpoint has exactly 3 experts (not 11)
        with open(policies_10 / "checkpoint_0.pkl", 'rb') as f:
            ckpt = pickle.load(f)
        assert len(ckpt["expert_params"]) == 3

    def test_parallel_conflict_resolution(self):
        """Test that frame-count heuristic resolves parallel training conflicts."""
        completed_skills = {}

        # Setup: Create 2 completed skills
        for i, (name, frames) in enumerate([("Skill_A", 50_000_000), ("Skill_B", 30_000_000)]):
            skill_data = {"code": "test", "skill_with_consumption": {}}
            run, _, _ = prepare_training_run(
                name, skill_data, i, [], completed_skills, self.base_dir, self.dummy_expert_init
            )
            self.simulate_training(run, i, frames)
            process_completed_training(run, name, i, self.base_dir, use_file_lock=False)
            completed_skills[name] = {"expert_idx": i}

        # Simulate parallel training of 2 skills that both use expert 0

        # Skill C: Depends on Skill_A, trains for 80M steps
        # Expert 0: 50M + 80M = 130M
        skill_c_data = {"code": "test", "skill_with_consumption": {}}
        run_c, _, _ = prepare_training_run(
            "Skill_C", skill_c_data, 2, ["Skill_A"], completed_skills,
            self.base_dir, self.dummy_expert_init
        )
        self.simulate_training(run_c, 2, 80_000_000)

        # Skill D: Depends on Skill_A, trains for 100M steps
        # Expert 0: 50M + 100M = 150M
        skill_d_data = {"code": "test", "skill_with_consumption": {}}
        run_d, _, _ = prepare_training_run(
            "Skill_D", skill_d_data, 3, ["Skill_A"], completed_skills,
            self.base_dir, self.dummy_expert_init
        )
        self.simulate_training(run_d, 3, 100_000_000)

        # Process C first (130M frames for expert 0)
        results_c = process_completed_training(
            run_c, "Skill_C", 2, self.base_dir, use_file_lock=False
        )
        assert results_c["expert_updates"][0]["action"] == "updated"
        assert results_c["expert_updates"][0]["frames"] == 130_000_000

        # Process D second (150M frames for expert 0)
        # Should overwrite C's version
        results_d = process_completed_training(
            run_d, "Skill_D", 3, self.base_dir, use_file_lock=False
        )
        assert results_d["expert_updates"][0]["action"] == "updated"
        assert results_d["expert_updates"][0]["frames"] == 150_000_000

        # Verify expert 0 has 150M frames (from D)
        expert_0_path = self.base_dir / "skills" / "0_Skill_A" / "expert_0_policy" / "params.pkl"
        with open(expert_0_path, 'rb') as f:
            expert_data = pickle.load(f)
        assert expert_data["metadata"]["total_frames"] == 150_000_000
        assert expert_data["params"]["timesteps"] == 100_000_000  # From skill D


class TestSchedulerIntegration:
    """Test scheduler with actual workflow components."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)

    def test_scheduler_state_tracking(self):
        """Test that scheduler correctly tracks skill states."""
        scheduler = SkillScheduler(base_dir=self.base_dir, max_parallel=2)

        # Add skills
        scheduler.add_skill("Skill_A", {"code": "test"}, [])
        scheduler.add_skill("Skill_B", {"code": "test"}, [])
        scheduler.add_skill("Skill_C", {"code": "test"}, ["Skill_A"])

        # Check initial state
        assert len(scheduler.state["skills"]) == 3
        assert all(s["status"] == "waiting" for s in scheduler.state["skills"].values())

        # Check dependencies
        completed_skills = {}
        assert scheduler._are_dependencies_satisfied("Skill_A", completed_skills)
        assert scheduler._are_dependencies_satisfied("Skill_B", completed_skills)
        assert not scheduler._are_dependencies_satisfied("Skill_C", completed_skills)

        # Simulate Skill_A completion
        completed_skills["Skill_A"] = {"expert_idx": 0}
        assert scheduler._are_dependencies_satisfied("Skill_C", completed_skills)

    def test_expert_assignment_on_launch(self):
        """Test that expert indices are assigned when skills start."""
        scheduler = SkillScheduler(base_dir=self.base_dir)

        # Add skill but don't launch yet
        skill_id = scheduler.add_skill("Test_Skill", {"code": "test"}, [])
        skill = scheduler.state["skills"][skill_id]

        # Expert not assigned yet
        assert skill["expert_idx"] is None
        assert scheduler.next_expert_idx == 0

        # Assign expert
        expert_idx = scheduler.assign_expert("Test_Skill")
        assert expert_idx == 0
        assert scheduler.next_expert_idx == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
