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

# W&B logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available - install with: pip install wandb")

# Core ML imports (required)
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import willccbb/verifiers components (will be installed in Colab)
try:
    from verifiers.trainer import GRPOTrainer
    from verifiers.envs import MultiTurnEnv
    from verifiers.data_types import Episode, Step
except ImportError:
    # Graceful fallback for development/testing
    logging.warning("verifiers package not found. Install with: pip install git+https://github.com/willccbb/verifiers.git")
    GRPOTrainer = None
    MultiTurnEnv = None
    Episode = None
    Step = None

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
    use_wandb: bool = True
    wandb_project: str = "diplomacy-grpo"
    wandb_entity: Optional[str] = None
    log_step_rewards: bool = True
    log_center_changes: bool = True
    log_model_weights: bool = False  # Expensive, set to True for detailed analysis
    
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
        
        # Initialize W&B if requested
        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            try:
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    config=asdict(config),
                    name=f"grpo-{config.model_name.replace('/', '-')}-{config.num_episodes}ep",
                    tags=["grpo", "diplomacy", "multi-agent"]
                )
                logger.info("W&B logging initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False
        else:
            logger.info("W&B logging disabled")
        
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
            
            # Log step-level metrics to W&B
            if self.use_wandb and self.config.log_step_rewards:
                step_metrics = {
                    f'step_reward/{power}': reward 
                    for power, reward in zip(sorted(self.env.agents.keys()), step_rewards)
                }
                step_metrics.update({
                    'step': step_count,
                    'phase': step_info['phase'],
                    'decision_type': step_info['decision_type'],
                    'episode': self.episode_count + 1,
                    'avg_step_reward': np.mean(step_rewards),
                    'max_step_reward': np.max(step_rewards),
                    'min_step_reward': np.min(step_rewards)
                })
                
                # Log center changes if available
                if self.config.log_center_changes:
                    current_state = self.env.get_current_state()
                    for power, centers in current_state.supply_centers.items():
                        step_metrics[f'centers/{power}'] = len(centers)
                
                wandb.log(step_metrics)
            
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
        alliance_analysis = analyze_alliance_patterns(self.env.alliance_tracker)
        
        # Log comprehensive episode metrics to W&B
        if self.use_wandb:
            episode_metrics = {
                'episode': self.episode_count + 1,
                'game_length_steps': episode_stats['total_steps'],
                'game_length_phases': episode_stats['game_length_phases'],
                'winner': episode_stats['winner'],
                'avg_final_reward': np.mean(final_rewards),
                'max_final_reward': np.max(final_rewards),
                'min_final_reward': np.min(final_rewards),
                'reward_variance': np.var(final_rewards),
                'total_alliances_formed': alliance_analysis['total_alliances_formed'],
                'alliances_broken': alliance_analysis['alliances_broken'],
                'betrayals_detected': alliance_analysis['betrayals_detected']
            }
            
            # Individual power final rewards
            for power, reward in zip(sorted(self.env.agents.keys()), final_rewards):
                episode_metrics[f'final_reward/{power}'] = reward
            
            # Final center counts and changes
            for power, power_stats in alliance_analysis['power_stats'].items():
                episode_metrics[f'final_centers/{power}'] = power_stats['final_centers']
                # Calculate center change from start
                if len(self.env.alliance_tracker.power_stats[power].supply_centers) > 1:
                    start_centers = self.env.alliance_tracker.power_stats[power].supply_centers[0]
                    center_change = power_stats['final_centers'] - start_centers
                    episode_metrics[f'center_change/{power}'] = center_change
                
                # Alliance and betrayal stats per power
                episode_metrics[f'alliances_formed/{power}'] = power_stats['alliances_formed']
                episode_metrics[f'alliances_broken/{power}'] = power_stats['alliances_broken']
                episode_metrics[f'betrayals_committed/{power}'] = power_stats['betrayals_committed']
                episode_metrics[f'betrayals_suffered/{power}'] = power_stats['betrayals_suffered']
                episode_metrics[f'eliminated/{power}'] = 1 if power_stats['eliminated'] else 0
            
            # Victory distribution (one-hot encoding)
            for power in sorted(self.env.agents.keys()):
                episode_metrics[f'victory/{power}'] = 1 if power == episode_stats['winner'] else 0
            
            wandb.log(episode_metrics)
        
        logger.info(f"Episode {self.episode_count + 1} completed: {episode_stats['summary']}")
        
        return {
            'episode_data': episode_data,
            'stats': episode_stats,
            'alliance_analysis': alliance_analysis
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
        
        if Episode is None:
            logger.warning("Episode class not available (verifiers not installed) - skipping model update")
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
            update_metrics = self.grpo_trainer.step(episodes)
            logger.info(f"Model updated with {len(episodes)} training examples")
            
            # Log GRPO update metrics if available
            if self.use_wandb and update_metrics:
                grpo_metrics = {
                    f'grpo/{k}': v for k, v in update_metrics.items()
                    if isinstance(v, (int, float))
                }
                grpo_metrics['grpo_update_step'] = self.episode_count + 1
                wandb.log(grpo_metrics)
                
        except Exception as e:
            logger.error(f"GRPO update failed: {e}")
            if self.use_wandb:
                wandb.log({'grpo_update_failed': 1, 'episode': self.episode_count + 1})
    
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
        
        # Log final training summary to W&B
        if self.use_wandb:
            self._log_final_summary()
            wandb.finish()
    
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
        
        # Log training progress to W&B
        if self.use_wandb:
            progress_metrics = {
                'training/avg_reward_last_5': avg_reward,
                'training/avg_length_last_5': avg_length,
                'training/total_episodes': self.episode_count + 1
            }
            
            # Victory distribution over last 10 episodes
            for power in sorted(self.env.agents.keys()):
                wins = victory_counts.get(power, 0)
                progress_metrics[f'training/victories_last_10/{power}'] = wins
                progress_metrics[f'training/win_rate_last_10/{power}'] = wins / min(10, self.episode_count + 1)
            
            # Calculate learning trends
            if len(recent_rewards) >= 3:
                early_avg = np.mean([np.mean(rewards) for rewards in recent_rewards[:2]])
                late_avg = np.mean([np.mean(rewards) for rewards in recent_rewards[-2:]])
                progress_metrics['training/reward_trend'] = late_avg - early_avg
            
            # Alliance formation trends
            if len(self.training_stats['alliance_stats']) >= 5:
                recent_alliances = [stats['total_alliances_formed'] for stats in self.training_stats['alliance_stats'][-5:]]
                progress_metrics['training/avg_alliances_last_5'] = np.mean(recent_alliances)
            
            wandb.log(progress_metrics)
        
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
    
    def _log_final_summary(self):
        """Log comprehensive final training summary to W&B"""
        if not self.use_wandb or not self.training_stats['episode_rewards']:
            return
            
        total_episodes = len(self.training_stats['episode_rewards'])
        all_rewards = [np.mean(rewards) for rewards in self.training_stats['episode_rewards']]
        
        # Overall performance metrics
        summary_metrics = {
            'summary/total_episodes': total_episodes,
            'summary/final_avg_reward': np.mean(all_rewards),
            'summary/best_avg_reward': np.max(all_rewards),
            'summary/worst_avg_reward': np.min(all_rewards),
            'summary/reward_std': np.std(all_rewards),
            'summary/avg_game_length': np.mean(self.training_stats['game_lengths']),
            'summary/total_training_steps': sum(len(ep) for ep in self.training_stats['episode_rewards'])
        }
        
        # Learning progress
        if total_episodes >= 10:
            early_performance = np.mean(all_rewards[:5])
            late_performance = np.mean(all_rewards[-5:])
            summary_metrics['summary/learning_improvement'] = late_performance - early_performance
            summary_metrics['summary/learning_improvement_pct'] = ((late_performance - early_performance) / early_performance) * 100
        
        # Victory distribution analysis
        victory_counts = {}
        for winner in self.training_stats['victory_distribution']:
            victory_counts[winner] = victory_counts.get(winner, 0) + 1
        
        for power in sorted(self.env.agents.keys()):
            wins = victory_counts.get(power, 0)
            summary_metrics[f'summary/total_victories/{power}'] = wins
            summary_metrics[f'summary/win_rate/{power}'] = wins / total_episodes
        
        # Alliance analysis
        if self.training_stats['alliance_stats']:
            total_alliances = [stats['total_alliances_formed'] for stats in self.training_stats['alliance_stats']]
            total_betrayals = [stats['betrayals_detected'] for stats in self.training_stats['alliance_stats']]
            
            summary_metrics['summary/avg_alliances_per_game'] = np.mean(total_alliances)
            summary_metrics['summary/avg_betrayals_per_game'] = np.mean(total_betrayals)
            summary_metrics['summary/alliance_stability'] = 1 - (np.mean(total_betrayals) / max(np.mean(total_alliances), 1))
        
        wandb.log(summary_metrics)
        logger.info("Final training summary logged to W&B")
    
    def log_model_weights(self):
        """Log model weights and gradients to W&B (expensive operation)"""
        if not self.use_wandb or not self.config.log_model_weights:
            return
            
        try:
            # Log model weights histogram
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    wandb.log({
                        f'weights/{name}': wandb.Histogram(param.data.cpu().numpy()),
                        f'gradients/{name}': wandb.Histogram(param.grad.data.cpu().numpy()),
                        f'grad_norm/{name}': param.grad.data.norm().item()
                    })
            
            # Log overall gradient norm
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
            wandb.log({'gradients/total_norm': total_norm})
            
        except Exception as e:
            logger.warning(f"Failed to log model weights: {e}")


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