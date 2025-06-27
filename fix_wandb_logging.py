#!/usr/bin/env python3
"""
Quick fix script for W&B logging integration issues.
Run this script to apply fixes for common compatibility issues.
"""

import os
import sys
from pathlib import Path

def fix_wandb_logger():
    """Fix W&B logger issues"""
    logger_path = Path(__file__).parent / "ai_diplomacy" / "wandb_llm_logger.py"
    
    if not logger_path.exists():
        print(f"Warning: {logger_path} not found")
        return
    
    with open(logger_path, 'r') as f:
        content = f.read()
    
    # Fix any remaining issues
    fixes = [
        # Ensure proper error handling
        ('wandb.define_metric("sessions/*/total_interactions"', '# wandb.define_metric("sessions/*/total_interactions"'),
        ('wandb.define_metric("interactions/*/response_time_ms"', '# wandb.define_metric("interactions/*/response_time_ms"'),
        ('wandb.define_metric("interactions/*/success_rate"', '# wandb.define_metric("interactions/*/success_rate"'),
    ]
    
    modified = False
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            modified = True
            print(f"Fixed: {old}")
    
    if modified:
        with open(logger_path, 'w') as f:
            f.write(content)
        print(f"Applied fixes to {logger_path}")
    else:
        print("No fixes needed for wandb_llm_logger.py")

def add_graceful_fallback():
    """Add graceful fallback for W&B initialization"""
    logger_path = Path(__file__).parent / "ai_diplomacy" / "wandb_llm_logger.py"
    
    with open(logger_path, 'r') as f:
        content = f.read()
    
    # Add better error handling for W&B initialization
    better_init = '''    def _init_wandb(self) -> None:
        """Initialize W&B run with enhanced configuration."""
        if not self.enabled:
            return
        
        try:
            config = {
                'logging_type': 'llm_interactions',
                'timestamp': datetime.now().isoformat(),
                'field_types': {
                    'numeric_fields': [
                        'timestamp', 'response_time_ms', 'phase_numeric', 'season_numeric',
                        'supply_center_count', 'unit_count', 'grpo_episode', 'grpo_step', 
                        'grpo_reward', 'token_usage_*', 'cost_estimate', 'success_rate',
                        'avg_response_time', 'total_interactions'
                    ],
                    'categorical_fields': [
                        'model_name', 'power_name', 'interaction_type', 'decision_type',
                        'phase', 'game_id', 'success', 'error_type'
                    ],
                    'text_fields': [
                        'raw_input_prompt', 'raw_response', 'processed_output', 'error_message'
                    ]
                }
            }
            
            wandb.init(
                project=self.project_name,
                entity=self.entity,
                config=config,
                name=f"llm-interactions-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                tags=["llm-logging", "diplomacy", "structured-logging"]
            )
            
            # Define basic metrics only (no glob patterns)
            wandb.define_metric("timestamp")
            wandb.define_metric("step")
            wandb.define_metric("episode_batch")
            
            logger.info("W&B run initialized for LLM interaction logging")
            
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}. Disabling W&B logging.")
            self.enabled = False'''
    
    # Replace the _init_wandb method
    import re
    pattern = r'def _init_wandb\(self\) -> None:.*?logger\.info\("W&B run initialized for LLM interaction logging"\)'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, better_init.strip(), content, flags=re.DOTALL)
        
        with open(logger_path, 'w') as f:
            f.write(content)
        print("Added graceful fallback for W&B initialization")
    else:
        print("Could not find _init_wandb method to replace")

def fix_missing_imports():
    """Fix missing imports in grpo_trainer.py"""
    trainer_path = Path(__file__).parent / "ai_diplomacy" / "grpo_trainer.py"
    
    if not trainer_path.exists():
        print(f"Warning: {trainer_path} not found")
        return
    
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Check if time import is missing
    if 'import time' not in content and 'start_time = time.time()' in content:
        # Add time import after numpy
        content = content.replace(
            'import numpy as np\n',
            'import numpy as np\nimport time\n'
        )
        
        with open(trainer_path, 'w') as f:
            f.write(content)
        print("✓ Added missing 'time' import to grpo_trainer.py")
    else:
        print("✓ grpo_trainer.py imports look correct")

def main():
    """Apply all fixes"""
    print("Applying W&B logging fixes...")
    
    try:
        fix_wandb_logger()
        add_graceful_fallback()
        fix_missing_imports()
        print("\n✓ All fixes applied successfully!")
        print("\nYou can now run GRPO training with:")
        print("python -c \"from ai_diplomacy.grpo_trainer import TrainingConfig, DiplomacyGRPOTrainer; trainer = DiplomacyGRPOTrainer(TrainingConfig(use_wandb=True)); trainer.train()\"")
        print("\nOr test the integration with:")
        print("python test_grpo_wandb.py")
        
    except Exception as e:
        print(f"\n❌ Fix failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()