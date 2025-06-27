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
from .wandb_llm_logger import log_grpo_interaction, get_llm_logger

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
    log_level: str = "INFO"  # Set to "WARNING" to reduce verbosity further
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
        
        # Initialize model and tokenizer with optimized settings
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Optimize model loading for large VRAM
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        
        if torch.cuda.is_available():
            # Use device_map for multi-GPU or optimize for single GPU
            if torch.cuda.device_count() > 1:
                model_kwargs["device_map"] = "auto"
            else:
                # For single GPU, load everything on GPU 0
                model_kwargs["device_map"] = {"": 0}
            
            # Use default attention (Flash Attention 2 removed for compatibility)
        
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
        
        # Enable gradient checkpointing for memory efficiency during training
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for memory efficiency")
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize environments for parallel training
        self.num_parallel_games = config.batch_size // 7  # Number of parallel games
        self.envs = []
        for i in range(self.num_parallel_games):
            env = DiplomacyMultiTurnEnv(
                model_name=config.model_name,
                max_year=config.max_year,
                num_negotiation_rounds=config.num_negotiation_rounds,
                random_seed=config.random_seed + i  # Different seed per game
            )
            self.envs.append(env)
        
        logger.info(f"Initialized {self.num_parallel_games} parallel game environments")
        
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
        
        # Initialize LLM logger for detailed interaction tracking
        self.llm_logger = get_llm_logger()
        
        if self.use_wandb:
            try:
                # Create enhanced config with metadata about field types
                wandb_config = asdict(config)
                wandb_config.update({
                    'field_types': {
                        'numeric_fields': [
                            'game_year', 'game_season', 'phase_type', 'decision_type_numeric',
                            'step', 'episode_batch', 'active_games', 'winner_id',
                            'avg_step_reward_all_games', 'centers_game_*'
                        ],
                        'categorical_fields': [
                            'victory_AUSTRIA', 'victory_ENGLAND', 'victory_FRANCE', 
                            'victory_GERMANY', 'victory_ITALY', 'victory_RUSSIA', 'victory_TURKEY'
                        ]
                    },
                    'power_mapping': {
                        'AUSTRIA': 0, 'ENGLAND': 1, 'FRANCE': 2, 'GERMANY': 3,
                        'ITALY': 4, 'RUSSIA': 5, 'TURKEY': 6
                    }
                })
                
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    config=wandb_config,
                    name=f"grpo-{config.model_name.replace('/', '-')}-{config.num_episodes}ep",
                    tags=["grpo", "diplomacy", "multi-agent", "numeric-optimized"]
                )
                
                # Define metric summaries to help W&B understand data types
                wandb.define_metric("step")
                wandb.define_metric("episode_batch")
                wandb.define_metric("game_year", step_metric="step")
                wandb.define_metric("game_season", step_metric="step")
                wandb.define_metric("phase_type", step_metric="step")
                wandb.define_metric("decision_type_numeric", step_metric="step")
                wandb.define_metric("avg_step_reward_all_games", step_metric="step")
                
                logger.info("W&B logging initialized with enhanced field type definitions")
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
    
    def generate_batch_responses(self, prompts: List[str], 
                                   episode: Optional[int] = None,
                                   step: Optional[int] = None,
                                   env_ids: Optional[List[int]] = None,
                                   power_names: Optional[List[str]] = None) -> List[str]:
        """
        Generate responses for all 7 agents using batched inference.
        
        Args:
            prompts: List of 7 prompts (one per agent)
            episode: Current GRPO episode for logging
            step: Current GRPO step for logging  
            env_ids: Environment IDs for each prompt
            power_names: Power names for each prompt
            
        Returns:
            List of 7 responses
        """
        start_time = time.time()
        
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
        
        # Generate responses with optimized settings
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,  # Increased for longer responses
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=self.config.num_generations,
                use_cache=True,  # Enable KV cache for efficiency
                num_beams=1,     # Disable beam search for speed
            )
        
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # Decode responses (remove input prompt)
        responses = []
        num_prompts = len(prompts)
        
        for i in range(num_prompts):
            # Handle multiple generations per prompt
            prompt_responses = []
            for gen in range(self.config.num_generations):
                output_idx = i * self.config.num_generations + gen
                if output_idx < len(outputs):
                    input_length = inputs['input_ids'][i].shape[0]
                    response_tokens = outputs[output_idx][input_length:]
                    response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                    prompt_responses.append(response.strip())
            
            # For now, take the first generation (could implement response selection here)
            if prompt_responses:
                responses.append(prompt_responses[0])
            else:
                responses.append("")  # Fallback
        
        # Log to W&B if enabled and we have the necessary information
        if (self.use_wandb and episode is not None and step is not None and 
            env_ids is not None and power_names is not None):
            try:
                for i, (prompt, response) in enumerate(zip(prompts, responses)):
                    env_id = env_ids[i] if i < len(env_ids) else 0
                    power_name = power_names[i] if i < len(power_names) else f'agent_{i}'
                    game_id = f"grpo_game_{env_id}_episode_{episode}"
                    
                    log_grpo_interaction(
                        game_id=game_id,
                        model_name=self.config.model_name,
                        episode=episode,
                        step=step,
                        prompt=prompt,
                        response=response,
                        power_name=power_name,
                        success=bool(response),  # Consider empty response as failure
                        response_time_ms=response_time_ms / len(prompts),  # Approximate per-response time
                    )
            except Exception as e:
                logger.warning(f"Failed to log GRPO interactions to W&B: {e}")
        
        return responses
    
    def run_episode(self) -> Dict[str, Any]:
        """
        Run parallel game episodes and collect training data.
        
        Returns:
            Combined episode statistics and data for GRPO update
        """
        logger.info(f"Starting episode batch {self.episode_count + 1} with {self.num_parallel_games} parallel games")
        
        # Start W&B game sessions for each environment
        if self.use_wandb:
            for i, env in enumerate(self.envs):
                game_id = f"grpo_game_{i}_episode_{self.episode_count}"
                game_config = {
                    'episode': self.episode_count,
                    'env_id': i,
                    'model_name': self.config.model_name,
                    'max_year': self.config.max_year,
                    'num_negotiation_rounds': self.config.num_negotiation_rounds,
                }
                self.llm_logger.start_game_session(
                    game_id=game_id,
                    game_config=game_config,
                    is_grpo_training=True,
                    grpo_episode=self.episode_count
                )
        
        # Reset all environments
        for env in self.envs:
            env.reset()
        
        all_episode_data = []
        step_count = 0
        
        # Run all games until completion
        while any(not env.is_completed() for env in self.envs):
            # Collect prompts from all active environments
            all_prompts = []
            active_env_indices = []
            
            for i, env in enumerate(self.envs):
                if not env.is_completed():
                    prompts = env.get_batch_prompts()
                    all_prompts.extend(prompts)
                    active_env_indices.extend([i] * len(prompts))
            
            if not all_prompts:
                break
            
            # Prepare additional data for logging
            env_ids_for_logging = []
            power_names_for_logging = []
            for i, env in enumerate(self.envs):
                if not env.is_completed():
                    num_agents = len(env.agents)
                    env_ids_for_logging.extend([i] * num_agents)
                    power_names_for_logging.extend(list(env.agents.keys()))
            
            # Generate responses for all prompts in one batch
            responses = self.generate_batch_responses(
                all_prompts,
                episode=self.episode_count,
                step=step_count,
                env_ids=env_ids_for_logging,
                power_names=power_names_for_logging
            )
            
            # Distribute responses back to environments
            response_idx = 0
            step_rewards_all = []
            step_info_all = []
            
            for i, env in enumerate(self.envs):
                if not env.is_completed():
                    num_agents = len(env.agents)
                    env_responses = responses[response_idx:response_idx + num_agents]
                    response_idx += num_agents
                    
                    # Process responses for this environment
                    step_rewards, step_info = env.process_batch_responses(env_responses)
                    step_rewards_all.extend(step_rewards)
                    step_info_all.append(step_info)
                    
                    # Store episode data for this environment
                    all_episode_data.append({
                        'env_id': i,
                        'step': step_count,
                        'prompts': env.get_batch_prompts(),
                        'responses': env_responses,
                        'rewards': step_rewards,
                        'info': step_info
                    })
            
            # Log aggregated step-level metrics to W&B
            if self.use_wandb and self.config.log_step_rewards and step_rewards_all:
                step_metrics = {
                    'step': step_count,
                    'episode_batch': self.episode_count + 1,
                    'active_games': len([env for env in self.envs if not env.is_completed()]),
                    'avg_step_reward_all_games': np.mean(step_rewards_all),
                    'max_step_reward_all_games': np.max(step_rewards_all),
                    'min_step_reward_all_games': np.min(step_rewards_all)
                }
                
                # Add phase and decision type info from first active environment (avoid string conflicts)
                for env in self.envs:
                    if not env.is_completed():
                        # Convert phase to numeric representation for better visualization
                        phase_str = env.current_phase
                        # Extract year and season for numeric tracking
                        if len(phase_str) >= 5 and phase_str[1:5].isdigit():
                            year = int(phase_str[1:5])
                            season_map = {'S': 0, 'F': 1, 'W': 2}  # Spring, Fall, Winter
                            season = season_map.get(phase_str[0], 0)
                            phase_type = 1 if 'M' in phase_str else 0  # Movement vs Adjustment
                            
                            step_metrics['game_year'] = year
                            step_metrics['game_season'] = season
                            step_metrics['phase_type'] = phase_type
                        
                        # Convert decision type to numeric
                        step_metrics['decision_type_numeric'] = 1 if env.current_decision_type.value == "orders" else 0
                        break
                
                # Log center changes across all games
                if self.config.log_center_changes:
                    for env_idx, env in enumerate(self.envs):
                        if not env.is_completed():
                            current_state = env.get_current_state()
                            for power, centers in current_state.supply_centers.items():
                                step_metrics[f'centers_game_{env_idx}/{power}'] = len(centers)
                
                wandb.log(step_metrics)
            
            step_count += 1
            
            # Log progress periodically
            if step_count % 10 == 0:
                active_games = len([env for env in self.envs if not env.is_completed()])
                logger.debug(f"Episode batch {self.episode_count + 1}, Step {step_count}, Active games: {active_games}")
        
        # Collect final results from all environments
        all_final_rewards = []
        all_episode_stats = []
        all_alliance_analyses = []
        
        for env_idx, env in enumerate(self.envs):
            final_rewards = env.get_final_rewards()
            all_final_rewards.extend(final_rewards)
            
            # Calculate stats for this environment
            env_episode_data = [data for data in all_episode_data if data['env_id'] == env_idx]
            episode_stats = self._calculate_episode_stats(env_episode_data, final_rewards)
            episode_stats['env_id'] = env_idx
            all_episode_stats.append(episode_stats)
            
            alliance_analysis = analyze_alliance_patterns(env.alliance_tracker)
            alliance_analysis['env_id'] = env_idx
            all_alliance_analyses.append(alliance_analysis)
        
        # Log comprehensive episode metrics to W&B
        if self.use_wandb:
            # Aggregate metrics across all games
            avg_final_reward = np.mean(all_final_rewards)
            total_alliances = sum(analysis['total_alliances_formed'] for analysis in all_alliance_analyses)
            total_betrayals = sum(analysis['betrayals_detected'] for analysis in all_alliance_analyses)
            
            batch_metrics = {
                'episode_batch': self.episode_count + 1,
                'parallel_games': self.num_parallel_games,
                'avg_final_reward_all_games': avg_final_reward,
                'max_final_reward_all_games': np.max(all_final_rewards),
                'min_final_reward_all_games': np.min(all_final_rewards),
                'total_alliances_all_games': total_alliances,
                'total_betrayals_all_games': total_betrayals,
                'avg_game_length': np.mean([stats['total_steps'] for stats in all_episode_stats])
            }
            
            # Per-game detailed metrics with numeric winner encoding
            power_to_id = {power: i for i, power in enumerate(sorted(['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']))}
            
            for env_idx, (stats, analysis) in enumerate(zip(all_episode_stats, all_alliance_analyses)):
                # Convert winner to numeric ID to avoid string conflicts
                winner_id = power_to_id.get(stats['winner'], -1)
                batch_metrics[f'game_{env_idx}/winner_id'] = winner_id
                batch_metrics[f'game_{env_idx}/game_length'] = stats['total_steps']
                batch_metrics[f'game_{env_idx}/alliances_formed'] = analysis['total_alliances_formed']
                batch_metrics[f'game_{env_idx}/betrayals'] = analysis['betrayals_detected']
                
                # Add per-power victory flags for this game
                for power, power_id in power_to_id.items():
                    batch_metrics[f'game_{env_idx}/victory_{power}'] = 1 if stats['winner'] == power else 0
            
            wandb.log(batch_metrics)
        
        # Create combined results
        combined_stats = {
            'parallel_games': self.num_parallel_games,
            'individual_game_stats': all_episode_stats,
            'avg_final_reward': np.mean(all_final_rewards),
            'total_steps': sum(stats['total_steps'] for stats in all_episode_stats),
            'summary': f"Batch {self.episode_count + 1}: {self.num_parallel_games} games, avg reward: {np.mean(all_final_rewards):.2f}"
        }
        
        # End W&B game sessions
        if self.use_wandb:
            for i in range(self.num_parallel_games):
                game_id = f"grpo_game_{i}_episode_{self.episode_count}"
                self.llm_logger.end_game_session(game_id)
        
        logger.info(f"Episode batch {self.episode_count + 1} completed: {combined_stats['summary']}")
        
        return {
            'episode_data': all_episode_data,
            'stats': combined_stats,
            'alliance_analysis': all_alliance_analyses
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