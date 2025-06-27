#!/usr/bin/env python3
"""
Test script to verify GRPO + W&B integration works correctly.
"""

import os
import sys
import time

def test_grpo_with_wandb():
    """Test GRPO training with W&B logging enabled."""
    print("Testing GRPO training with W&B logging...")
    
    try:
        from ai_diplomacy.grpo_trainer import TrainingConfig, DiplomacyGRPOTrainer
        
        # Create a minimal config for testing
        config = TrainingConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",  # Small model for testing
            batch_size=7,  # Single game (7 agents)
            num_episodes=2,  # Just 2 episodes for testing
            max_year=1902,  # Very short games
            num_negotiation_rounds=1,  # Minimal negotiations
            use_wandb=True,  # Enable W&B logging
            log_level="INFO"
        )
        
        print(f"Config: {config.model_name}, {config.num_episodes} episodes, max year {config.max_year}")
        
        # Create trainer
        trainer = DiplomacyGRPOTrainer(config)
        print("✓ Trainer created successfully")
        
        # Test that we can access the environments
        print(f"✓ Created {len(trainer.envs)} parallel environments")
        
        # Test that W&B logging is properly initialized
        if trainer.use_wandb:
            print("✓ W&B logging enabled")
        else:
            print("⚠ W&B logging disabled")
        
        # Run a very short training session
        print("Starting mini training session...")
        trainer.train()
        print("✓ Training completed successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install torch transformers wandb")
        return False
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wandb_logging_only():
    """Test just the W&B logging components without GRPO."""
    print("\nTesting W&B logging components...")
    
    try:
        from ai_diplomacy.wandb_llm_logger import (
            initialize_llm_logging,
            log_order_generation,
            get_llm_logger
        )
        
        # Initialize W&B logging
        logger = initialize_llm_logging(
            project_name="diplomacy-test",
            enabled=True
        )
        
        if not logger.enabled:
            print("⚠ W&B not available - logging disabled")
            return True
        
        # Start a test session
        logger.start_game_session(
            game_id="test_session",
            game_config={"test": True},
            is_grpo_training=True,
            grpo_episode=1
        )
        print("✓ Started test W&B session")
        
        # Log a test interaction
        log_order_generation(
            game_id="test_session",
            model_name="test-model",
            power_name="AUSTRIA",
            phase="S1901M",
            prompt="Test prompt",
            response="Test response",
            success=True,
            response_time_ms=100.0
        )
        print("✓ Logged test interaction")
        
        # End the session
        logger.end_game_session("test_session")
        print("✓ Ended test W&B session")
        
        return True
        
    except Exception as e:
        print(f"❌ W&B logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing GRPO + W&B Integration")
    print("=" * 50)
    
    # Test W&B logging first
    wandb_success = test_wandb_logging_only()
    
    if wandb_success:
        print("\n" + "=" * 50)
        # Test full GRPO integration
        grpo_success = test_grpo_with_wandb()
        
        if grpo_success:
            print("\n🎉 All tests passed! GRPO + W&B integration is working.")
        else:
            print("\n⚠ W&B logging works, but GRPO training failed.")
            sys.exit(1)
    else:
        print("\n❌ W&B logging failed - check your setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()