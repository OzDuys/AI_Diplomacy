#!/usr/bin/env python3
"""
Test script for W&B LLM logging functionality.

This script tests the basic functionality of the W&B logging system
without running a full game.
"""

import os
import sys
import time
from ai_diplomacy.wandb_llm_logger import (
    initialize_llm_logging,
    log_order_generation,
    log_negotiation,
    log_planning,
    log_grpo_interaction,
    get_llm_logger
)

def test_basic_logging():
    """Test basic W&B logging functionality."""
    print("Testing W&B LLM logging...")
    
    # Initialize logging
    logger = initialize_llm_logging(
        project_name="diplomacy-llm-test",
        enabled=True
    )
    
    if not logger.enabled:
        print("W&B not available - skipping test")
        return
    
    # Start a test game session
    game_id = "test_game_001"
    game_config = {
        'test': True,
        'max_year': 1902,
        'models': {'AUSTRIA': 'test-model', 'ENGLAND': 'test-model'}
    }
    
    logger.start_game_session(
        game_id=game_id,
        game_config=game_config,
        is_grpo_training=False
    )
    
    print(f"Started game session: {game_id}")
    
    # Test order generation logging
    log_order_generation(
        game_id=game_id,
        model_name="test-model",
        power_name="AUSTRIA",
        phase="S1901M",
        prompt="Generate orders for Austria in Spring 1901...",
        response='{"orders": ["A VIE H", "A BUD H", "F TRI H"]}',
        orders=["A VIE H", "A BUD H", "F TRI H"],
        success=True,
        response_time_ms=1250.5,
        supply_centers=3,
        units=3
    )
    print("Logged order generation")
    
    # Test negotiation logging
    log_negotiation(
        game_id=game_id,
        model_name="test-model",
        power_name="ENGLAND",
        phase="S1901M",
        prompt="Generate negotiation messages...",
        response='[{"message_type": "private", "recipient": "FRANCE", "content": "Hello France"}]',
        messages=[{"message_type": "private", "recipient": "FRANCE", "content": "Hello France"}],
        success=True,
        response_time_ms=890.2
    )
    print("Logged negotiation")
    
    # Test planning logging
    log_planning(
        game_id=game_id,
        model_name="test-model",
        power_name="FRANCE",
        phase="S1901M",
        prompt="Generate strategic plan...",
        response="Focus on securing the Channel and Mediterranean...",
        success=True,
        response_time_ms=2100.7
    )
    print("Logged planning")
    
    # Test GRPO logging
    log_grpo_interaction(
        game_id="grpo_test_game",
        model_name="test-grpo-model",
        episode=1,
        step=5,
        prompt="GRPO training prompt...",
        response="GRPO response...",
        reward=0.75,
        power_name="GERMANY",
        success=True,
        response_time_ms=500.0
    )
    print("Logged GRPO interaction")
    
    # Wait a moment for logs to process
    time.sleep(2)
    
    # Get session stats
    stats = logger.get_session_stats(game_id)
    if stats:
        print(f"Session stats: {stats}")
    
    # End the session
    logger.end_game_session(game_id)
    print(f"Ended game session: {game_id}")
    
    print("✓ W&B logging test completed successfully!")

def test_bulk_logging():
    """Test bulk logging functionality."""
    print("\nTesting bulk logging...")
    
    logger = get_llm_logger()
    if not logger.enabled:
        print("W&B not available - skipping bulk test")
        return
    
    # Create multiple interactions for bulk logging
    from ai_diplomacy.wandb_llm_logger import LLMInteraction
    
    interactions = []
    for i in range(5):
        interaction = LLMInteraction(
            model_name="bulk-test-model",
            power_name="RUSSIA",
            game_id="bulk_test_game",
            phase=f"S190{i+1}M",
            interaction_type="order_generation",
            timestamp=time.time(),
            response_time_ms=1000 + i * 100,
            raw_input_prompt=f"Test prompt {i+1}",
            raw_response=f"Test response {i+1}",
            processed_output=f'["A MOS H", "F SEV H"]',
            success=True,
            phase_numeric=1901 + i,
            season_numeric=0,
            decision_type="orders",
            supply_center_count=4,
            unit_count=2,
        )
        interactions.append(interaction)
    
    # Log them in bulk
    logger.log_bulk_interactions(interactions)
    print(f"✓ Bulk logged {len(interactions)} interactions")

def test_error_handling():
    """Test error handling in logging."""
    print("\nTesting error handling...")
    
    logger = get_llm_logger()
    if not logger.enabled:
        print("W&B not available - skipping error test")
        return
    
    # Test logging with errors
    log_order_generation(
        game_id="error_test_game",
        model_name="error-test-model",
        power_name="ITALY",
        phase="F1901M",
        prompt="Generate orders...",
        response="",  # Empty response
        success=False,
        error_message="API timeout",
        response_time_ms=30000.0
    )
    print("✓ Logged error case")

if __name__ == "__main__":
    print("W&B LLM Logging Test Suite")
    print("=" * 40)
    
    try:
        test_basic_logging()
        test_bulk_logging()
        test_error_handling()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)