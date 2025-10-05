#!/usr/bin/env python3
"""
Test script to verify frontier-based prompt integration is working correctly.
This tests the basic integration without requiring full training infrastructure.
"""

import sys
import os
from pathlib import Path

# Add the flowrl package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_frontier_integration():
    """Test that frontier-based prompts are properly integrated"""
    print("Testing frontier-based prompt integration...")

    # Mock arguments for testing
    class MockArgs:
        def __init__(self):
            self.env_name = "Craftax-Symbolic-v1"  # Should enable frontier prompts
            self.graph_path = Path("/tmp/test_flow")
            self.current_i = 0
            self.previous_i = None

    args = MockArgs()

    try:
        from flowrl.llm.flow import Flow

        # Create Flow instance
        print(f"Creating Flow instance for environment: {args.env_name}")
        flow = Flow(args)

        # Verify frontier-based prompts are enabled
        print(f"Frontier-based prompts enabled: {flow.use_frontier_based_prompts}")
        assert flow.use_frontier_based_prompts == True, "Frontier prompts should be enabled for Craftax-Symbolic-v1"

        # Test frontier summary generation
        print("Testing frontier summary generation...")
        frontier_summary = flow.generate_frontier_summary_from_skills()
        print(f"Initial frontier summary:\n{frontier_summary}")

        # Verify frontier summary is in database
        assert "frontier_summary" in flow.db, "Frontier summary should be in database"
        print("✓ Frontier summary is in database")

        # Test adding a mock skill and verifying frontier update
        print("\nTesting skill addition and frontier update...")
        mock_skill_data = {
            "skill_name": "collect_wood",
            "skill_with_consumption": {
                "skill_name": "collect_wood",
                "requirements": {},
                "consumption": {},
                "gain": {"wood": "lambda n: n"},
                "ephemeral": False
            },
            "functions": ["mock_function_1", "mock_function_2", "mock_function_3"],
            "iteration": 0
        }

        # Add the skill
        flow.add_skill("collect_wood", mock_skill_data)

        # Verify skill was added
        assert "collect_wood" in flow.skills, "Skill should be added to skills dictionary"
        print("✓ Skill added successfully")

        # Verify frontier summary was updated
        updated_summary = flow.db["frontier_summary"]
        print(f"Updated frontier summary:\n{updated_summary}")
        assert "wood" in updated_summary.lower(), "Updated frontier should mention wood"
        print("✓ Frontier summary updated after skill addition")

        # Test symbolic state verification
        print("\nTesting symbolic state verification...")
        from flowrl.llm.craftax.symbolic_state import verify_skill

        # Test verifying the existing skill (should be not novel)
        is_novel, is_feasible = verify_skill("collect_wood", mock_skill_data["skill_with_consumption"], flow.skills, 99)
        print(f"Existing skill verification - Novel: {is_novel}, Feasible: {is_feasible}")
        assert not is_novel, "Existing skill should not be novel"

        # Test verifying a new skill (should be novel)
        new_skill_data = {
            "skill_name": "collect_stone",
            "requirements": {},
            "consumption": {},
            "gain": {"stone": "lambda n: n"},
            "ephemeral": False
        }
        is_novel_new, is_feasible_new = verify_skill("collect_stone", new_skill_data, flow.skills, 99)
        print(f"New skill verification - Novel: {is_novel_new}, Feasible: {is_feasible_new}")
        assert is_novel_new, "New skill should be novel"
        print("✓ Symbolic state verification working")

        # Test that graph was properly initialized
        assert flow.graph is not None, "Graph should be initialized"
        assert flow.inventory_graph is not None, "Inventory graph should be initialized"
        print("✓ Graph and inventory graph properly initialized")

        print("\n🎉 All frontier integration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backwards_compatibility():
    """Test that Craftax Classic still works without frontier prompts"""
    print("\nTesting backwards compatibility...")

    class MockArgs:
        def __init__(self):
            self.env_name = "Craftax-Classic-Symbolic-v1"  # Should disable frontier prompts
            self.graph_path = Path("/tmp/test_flow_classic")
            self.current_i = 0
            self.previous_i = None

    args = MockArgs()

    try:
        from flowrl.llm.flow import Flow

        # Create Flow instance for Classic
        print(f"Creating Flow instance for environment: {args.env_name}")
        flow = Flow(args)

        # Verify frontier-based prompts are disabled
        print(f"Frontier-based prompts enabled: {flow.use_frontier_based_prompts}")
        assert flow.use_frontier_based_prompts == False, "Frontier prompts should be disabled for Craftax-Classic"

        # Verify no frontier summary in database
        frontier_summary = flow.generate_frontier_summary_from_skills()
        print(f"Classic frontier summary: {frontier_summary}")
        assert "disabled" in frontier_summary, "Frontier summary should indicate disabled for Classic"

        print("✓ Backwards compatibility maintained")
        return True

    except Exception as e:
        print(f"❌ Backwards compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("FRONTIER-BASED PROMPT INTEGRATION TEST")
    print("=" * 60)

    # Run tests
    test1_passed = test_frontier_integration()
    test2_passed = test_backwards_compatibility()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Frontier Integration Test: {'✓ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Backwards Compatibility Test: {'✓ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! Frontier integration is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)