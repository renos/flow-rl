"""
Unit tests for scheduler.py

Tests skill scheduling, dependency resolution, and state management.
"""

import pytest
import tempfile
import json
from pathlib import Path

from flowrl.parallel.scheduler import SkillScheduler


class TestSkillScheduler:
    """Test the SkillScheduler class."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)

    def test_scheduler_initialization(self):
        """Test scheduler initializes correctly."""
        scheduler = SkillScheduler(
            base_dir=self.base_dir,
            max_parallel=3,
            poll_interval=10
        )

        assert scheduler.max_parallel == 3
        assert scheduler.poll_interval == 10
        assert scheduler.next_expert_idx == 0
        assert len(scheduler.state["skills"]) == 0

    def test_add_skill(self):
        """Test adding skills to scheduler."""
        scheduler = SkillScheduler(base_dir=self.base_dir)

        skill_data = {"code": "test", "skill_with_consumption": {}}

        skill_id = scheduler.add_skill(
            skill_name="Test Skill",
            skill_data=skill_data,
            dependency_skill_names=[]
        )

        assert skill_id == "Test_Skill"
        assert "Test_Skill" in scheduler.state["skills"]
        assert scheduler.state["skills"]["Test_Skill"]["status"] == "waiting"
        assert scheduler.state["skills"]["Test_Skill"]["dependencies"] == []

    def test_add_skill_with_dependencies(self):
        """Test adding skill with dependencies."""
        scheduler = SkillScheduler(base_dir=self.base_dir)

        skill_id = scheduler.add_skill(
            skill_name="Make Pickaxe",
            skill_data={},
            dependency_skill_names=["Collect Wood", "Collect Stone"]
        )

        skill = scheduler.state["skills"][skill_id]
        assert skill["dependencies"] == ["Collect Wood", "Collect Stone"]

    def test_expert_assignment(self):
        """Test expert index assignment."""
        scheduler = SkillScheduler(base_dir=self.base_dir)

        # First assignment
        expert_0 = scheduler.assign_expert("Skill A")
        assert expert_0 == 0
        assert scheduler.next_expert_idx == 1

        # Second assignment
        expert_1 = scheduler.assign_expert("Skill B")
        assert expert_1 == 1
        assert scheduler.next_expert_idx == 2

        # Reassignment returns same index
        expert_0_again = scheduler.assign_expert("Skill A")
        assert expert_0_again == 0
        assert scheduler.next_expert_idx == 2  # Doesn't increment

    def test_expert_assignment_increments(self):
        """Test that expert indices increment correctly."""
        scheduler = SkillScheduler(base_dir=self.base_dir)

        indices = []
        for i in range(5):
            idx = scheduler.assign_expert(f"Skill {i}")
            indices.append(idx)

        assert indices == [0, 1, 2, 3, 4]
        assert scheduler.next_expert_idx == 5

    def test_state_persistence(self):
        """Test that scheduler state persists to disk."""
        # Create scheduler and add skills
        scheduler = SkillScheduler(base_dir=self.base_dir)
        scheduler.add_skill("Skill A", {}, [])
        scheduler.add_skill("Skill B", {}, ["Skill A"])

        expert_idx = scheduler.assign_expert("Skill A")

        # Load state in new scheduler
        scheduler2 = SkillScheduler(base_dir=self.base_dir)

        assert len(scheduler2.state["skills"]) == 2
        assert "Skill_A" in scheduler2.state["skills"]
        assert "Skill_B" in scheduler2.state["skills"]
        assert scheduler2.next_expert_idx == 1  # Expert 0 was assigned
        assert scheduler2.expert_assignments["Skill A"] == expert_idx

    def test_dependency_satisfaction_check(self):
        """Test checking if dependencies are satisfied."""
        scheduler = SkillScheduler(base_dir=self.base_dir)

        # Add skills
        scheduler.add_skill("Skill A", {}, [])
        scheduler.add_skill("Skill B", {}, [])
        scheduler.add_skill("Skill C", {}, ["Skill A", "Skill B"])

        # No skills completed yet
        completed_skills = {}
        assert not scheduler._are_dependencies_satisfied("Skill_C", completed_skills)

        # Only Skill A completed
        completed_skills = {"Skill A": {"expert_idx": 0}}
        assert not scheduler._are_dependencies_satisfied("Skill_C", completed_skills)

        # Both dependencies completed
        completed_skills = {
            "Skill A": {"expert_idx": 0},
            "Skill B": {"expert_idx": 1}
        }
        assert scheduler._are_dependencies_satisfied("Skill_C", completed_skills)

    def test_get_completed_skill_names(self):
        """Test retrieving completed skill names."""
        scheduler = SkillScheduler(base_dir=self.base_dir)

        scheduler.add_skill("Skill A", {}, [])
        scheduler.add_skill("Skill B", {}, [])
        scheduler.add_skill("Skill C", {}, [])

        # Mark some as completed
        scheduler.state["skills"]["Skill_A"]["status"] = "completed"
        scheduler.state["skills"]["Skill_B"]["status"] = "running"
        scheduler.state["skills"]["Skill_C"]["status"] = "completed"

        completed = scheduler.get_completed_skill_names()
        assert set(completed) == {"Skill A", "Skill C"}

    def test_get_running_skill_names(self):
        """Test retrieving running skill names."""
        scheduler = SkillScheduler(base_dir=self.base_dir)

        scheduler.add_skill("Skill A", {}, [])
        scheduler.add_skill("Skill B", {}, [])

        scheduler.state["skills"]["Skill_A"]["status"] = "running"
        scheduler.state["skills"]["Skill_B"]["status"] = "waiting"
        scheduler.state["currently_running"] = ["Skill_A"]

        running = scheduler.get_running_skill_names()
        assert running == ["Skill A"]

    def test_is_complete(self):
        """Test checking if all skills are done."""
        scheduler = SkillScheduler(base_dir=self.base_dir)

        scheduler.add_skill("Skill A", {}, [])
        scheduler.add_skill("Skill B", {}, [])

        # Skills waiting
        assert not scheduler._is_complete()

        # One running
        scheduler.state["skills"]["Skill_A"]["status"] = "running"
        assert not scheduler._is_complete()

        # One completed, one waiting
        scheduler.state["skills"]["Skill_A"]["status"] = "completed"
        assert not scheduler._is_complete()

        # Both completed
        scheduler.state["skills"]["Skill_B"]["status"] = "completed"
        assert scheduler._is_complete()

        # One failed, one completed
        scheduler.state["skills"]["Skill_B"]["status"] = "failed"
        assert scheduler._is_complete()


class TestCallbacks:
    """Test callback functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)

    def test_set_callbacks(self):
        """Test setting callbacks."""
        scheduler = SkillScheduler(base_dir=self.base_dir)

        def on_complete(skill_name):
            pass

        def prepare_run(*args, **kwargs):
            pass

        def process_completed(*args, **kwargs):
            pass

        scheduler.set_callbacks(
            on_skill_complete=on_complete,
            prepare_training_run=prepare_run,
            process_completed_training=process_completed
        )

        assert scheduler.on_skill_complete_callback == on_complete
        assert scheduler.prepare_training_run_callback == prepare_run
        assert scheduler.process_completed_training_callback == process_completed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
