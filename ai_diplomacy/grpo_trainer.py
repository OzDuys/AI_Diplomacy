# ai_diplomacy/grpo_trainer.py
"""
Main GRPO training integration for Diplomacy self-play

This module integrates the DiplomacyMultiTurnEnv with willccbb/verifiers GRPO trainer
for online reinforcement learning of Diplomacy agents.
"""

import logging
import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json

# Import willccbb/verifiers components (will be installed in Colab)
try:
    from verifiers.trainer import GRPOTrainer
    from verifiers.envs import MultiTurnEnv
    from verifiers.data_types import Episode, Step
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    # Graceful fallback for development/testing
    logging.warning("verifiers package not found. Install with: pip install git+https://github.com/willccbb/verifiers.git")
    GRPOTrainer = None
    MultiTurnEnv = None

from .grpo_env import DiplomacyMultiTurnEnv, DecisionType
from .grpo_rewards import analyze_alliance_patterns

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for GRPO training"""
    # Model settings
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_length: int = 2048
    
    # Training settings
    batch_size: int = 7  # One batch = one full game (7 agents)
    learning_rate: float = 1e-5
    num_episodes: int = 100
    max_year: int = 1910
    num_negotiation_rounds: int = 3
    
    # GRPO specific
    temperature: float = 0.8
    top_p: float = 0.9
    kl_coeff: float = 0.1
    num_generations: int = 1  # Generations per prompt
    gradient_accumulation_steps: int = 1
    
    # Checkpointing
    save_every: int = 10  # Save every N episodes
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_level: str = "INFO"
    log_alliance_analysis: bool = True
    
    # Random seeds
    random_seed: int = 42
    torch_seed: int = 42


class DiplomacyGRPOTrainer:
    """
    Main trainer class that orchestrates GRPO training for Diplomacy agents.
    
    This class:
    1. Manages the DiplomacyMultiTurnEnv for game simulation
    2. Interfaces with willccbb/verifiers GRPOTrainer
    3. Handles batched generation for 7-agent self-play
    4. Manages model checkpointing and evaluation
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._setup_logging()
        self._setup_random_seeds()
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize environment
        self.env = DiplomacyMultiTurnEnv(
            model_name=config.model_name,
            max_year=config.max_year,
            num_negotiation_rounds=config.num_negotiation_rounds,
            random_seed=config.random_seed
        )
        
        # Initialize GRPO trainer
        if GRPOTrainer is not None:
            self.grpo_trainer = GRPOTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                learning_rate=config.learning_rate,
                kl_coeff=config.kl_coeff,
                temperature=config.temperature,
                top_p=config.top_p,
                max_length=config.max_length
            )
        else:
            self.grpo_trainer = None
            logger.warning("GRPO trainer not available - running in simulation mode")
        
        # Training state
        self.episode_count = 0
        self.training_stats = {
            'episode_rewards': [],
            'alliance_stats': [],
            'game_lengths': [],
            'victory_distribution': []
        }
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        
        logger.info(f"Initialized DiplomacyGRPOTrainer with model {config.model_name}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _setup_random_seeds(self):
        """Setup random seeds for reproducibility"""
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.torch_seed)
    
    def generate_batch_responses(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for all 7 agents using batched inference.
        
        Args:
            prompts: List of 7 prompts (one per agent)
            
        Returns:
            List of 7 responses
        """
        # Set consistent random seed for reproducible batching
        torch.manual_seed(self.config.torch_seed)
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate responses
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,  # Limit response length
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode responses (remove input prompt)
        responses = []
        for i, output in enumerate(outputs):
            input_length = inputs['input_ids'][i].shape[0]
            response_tokens = output[input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses
    
    def run_episode(self) -> Dict[str, Any]:
        """
        Run a single game episode and collect training data.
        
        Returns:
            Episode statistics and data for GRPO update
        """
        logger.info(f"Starting episode {self.episode_count + 1}")
        
        # Reset environment
        self.env.reset()
        episode_data = []
        step_count = 0
        
        # Run game until completion
        while not self.env.is_completed():
            # Get prompts for current decision
            prompts = self.env.get_batch_prompts()
            
            # Generate responses
            responses = self.generate_batch_responses(prompts)
            
            # Process responses and get rewards
            step_rewards, step_info = self.env.process_batch_responses(responses)
            
            # Store episode data
            episode_data.append({
                'step': step_count,
                'prompts': prompts,
                'responses': responses,
                'rewards': step_rewards,
                'info': step_info
            })
            
            step_count += 1
            
            # Log progress periodically
            if step_count % 10 == 0:
                logger.debug(f"Episode {self.episode_count + 1}, Step {step_count}")
        
        # Get final rewards
        final_rewards = self.env.get_final_rewards()
        
        # Add final rewards to last step
        if episode_data:
            episode_data[-1]['final_rewards'] = final_rewards
        
        # Calculate episode statistics
        episode_stats = self._calculate_episode_stats(episode_data, final_rewards)
        
        logger.info(f"Episode {self.episode_count + 1} completed: {episode_stats['summary']}")
        
        return {
            'episode_data': episode_data,
            'stats': episode_stats,
            'alliance_analysis': analyze_alliance_patterns(self.env.alliance_tracker)
        }
    
    def _calculate_episode_stats(self, episode_data: List[Dict], final_rewards: List[float]) -> Dict[str, Any]:
        """Calculate statistics for completed episode"""
        total_steps = len(episode_data)
        total_rewards = [sum(step['rewards']) for step in episode_data]
        
        # Find winner (highest final reward)
        winner_idx = np.argmax(final_rewards)
        winner_power = sorted(list(self.env.agents.keys()))[winner_idx]
        
        # Game length (phases)
        game_phases = total_steps // 2  # Approximate (orders + negotiations)
        
        stats = {
            'total_steps': total_steps,
            'game_length_phases': game_phases,
            'winner': winner_power,
            'final_rewards': final_rewards,
            'avg_step_reward': np.mean(total_rewards),
            'summary': f"Winner: {winner_power}, Steps: {total_steps}, Avg Reward: {np.mean(total_rewards):.2f}"
        }
        
        return stats
    
    def update_model(self, episode_result: Dict[str, Any]):
        """
        Update model using GRPO with episode data.
        
        Args:
            episode_result: Result from run_episode()
        """
        if self.grpo_trainer is None:
            logger.warning("GRPO trainer not available - skipping model update")
            return
        
        episode_data = episode_result['episode_data']
        
        # Convert to verifiers format
        episodes = []
        for step_data in episode_data:
            # Create episodes for each agent's decision
            for i, (prompt, response, reward) in enumerate(zip(
                step_data['prompts'], 
                step_data['responses'], 
                step_data['rewards']
            )):
                episode = Episode(
                    prompt=prompt,
                    response=response,
                    reward=reward
                )
                episodes.append(episode)
        
        # Perform GRPO update
        try:
            self.grpo_trainer.step(episodes)
            logger.info(f"Model updated with {len(episodes)} training examples")
        except Exception as e:
            logger.error(f"GRPO update failed: {e}")
    
    def train(self):
        """Run full training loop"""
        logger.info(f"Starting GRPO training for {self.config.num_episodes} episodes")
        
        for episode in range(self.config.num_episodes):
            self.episode_count = episode
            
            # Run episode
            episode_result = self.run_episode()
            
            # Update model
            self.update_model(episode_result)
            
            # Track statistics
            self.training_stats['episode_rewards'].append(episode_result['stats']['final_rewards'])
            self.training_stats['alliance_stats'].append(episode_result['alliance_analysis'])
            self.training_stats['game_lengths'].append(episode_result['stats']['game_length_phases'])
            self.training_stats['victory_distribution'].append(episode_result['stats']['winner'])
            
            # Save checkpoint
            if (episode + 1) % self.config.save_every == 0:
                self.save_checkpoint(episode + 1)
            
            # Log progress
            if (episode + 1) % 5 == 0:
                self._log_training_progress()
        
        logger.info("Training completed!")
        self.save_final_results()
    
    def save_checkpoint(self, episode: int):
        """Save model checkpoint and training stats"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_episode_{episode}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model
        if self.grpo_trainer is not None:
            self.model.save_pretrained(checkpoint_path / "model")
            self.tokenizer.save_pretrained(checkpoint_path / "tokenizer")
        
        # Save training stats
        with open(checkpoint_path / "training_stats.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            stats_to_save = {}
            for key, value in self.training_stats.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], np.ndarray):
                        stats_to_save[key] = [v.tolist() for v in value]
                    else:
                        stats_to_save[key] = value
                else:
                    stats_to_save[key] = value
            json.dump(stats_to_save, f, indent=2)
        
        # Save config
        with open(checkpoint_path / "config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Checkpoint saved at episode {episode}")
    
    def _log_training_progress(self):
        """Log training progress statistics"""
        recent_rewards = self.training_stats['episode_rewards'][-5:]  # Last 5 episodes
        recent_lengths = self.training_stats['game_lengths'][-5:]
        
        avg_reward = np.mean([np.mean(rewards) for rewards in recent_rewards])
        avg_length = np.mean(recent_lengths)
        
        victory_counts = {}
        for winner in self.training_stats['victory_distribution'][-10:]:  # Last 10 episodes
            victory_counts[winner] = victory_counts.get(winner, 0) + 1
        
        logger.info(f"Training Progress - Episode {self.episode_count + 1}:")
        logger.info(f"  Avg Reward (last 5): {avg_reward:.2f}")
        logger.info(f"  Avg Game Length (last 5): {avg_length:.1f} phases")
        logger.info(f"  Victory Distribution (last 10): {victory_counts}")
    
    def save_final_results(self):
        """Save final training results and analysis"""
        results_path = Path(self.config.checkpoint_dir) / "final_results"
        results_path.mkdir(exist_ok=True)
        
        # Save final model
        if self.grpo_trainer is not None:
            self.model.save_pretrained(results_path / "final_model")
            self.tokenizer.save_pretrained(results_path / "final_tokenizer")
        
        # Save complete training stats
        with open(results_path / "complete_training_stats.json", 'w') as f:
            stats_to_save = {
                'config': asdict(self.config),
                'training_stats': self.training_stats,
                'total_episodes': self.episode_count + 1
            }
            # Handle numpy arrays
            for key, value in stats_to_save['training_stats'].items():
                if isinstance(value, list) and len(value) > 0:
                    if hasattr(value[0], 'tolist'):
                        stats_to_save['training_stats'][key] = [v.tolist() if hasattr(v, 'tolist') else v for v in value]
            
            json.dump(stats_to_save, f, indent=2, default=str)
        
        logger.info("Final results saved!")


def create_trainer_from_config(config_dict: Dict[str, Any]) -> DiplomacyGRPOTrainer:
    """Create trainer from configuration dictionary"""
    config = TrainingConfig(**config_dict)
    return DiplomacyGRPOTrainer(config)


if __name__ == "__main__":
    # Example usage
    config = TrainingConfig(
        num_episodes=10,  # Small test run
        max_year=1905,    # Shorter games for testing
        save_every=5
    )
    
    trainer = DiplomacyGRPOTrainer(config)
    trainer.train()