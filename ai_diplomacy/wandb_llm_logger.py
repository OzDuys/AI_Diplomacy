"""
W&B logging module for comprehensive LLM output tracking.

This module provides structured logging of all LLM interactions to Weights & Biases,
including thinking, negotiations, orders, and GRPO training responses.
Supports parallel game separation and detailed analytics.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LLMInteraction:
    """Structured representation of an LLM interaction for logging."""
    # Core identification
    model_name: str
    power_name: Optional[str]
    game_id: str  # Unique identifier for the game instance
    phase: str
    interaction_type: str  # 'order_generation', 'negotiation', 'planning', 'grpo_training'
    
    # Timing
    timestamp: float
    response_time_ms: Optional[float] = None
    
    # Content
    raw_input_prompt: str = ""
    raw_response: str = ""
    processed_output: Optional[str] = None  # Parsed orders, messages, etc.
    
    # Success/failure tracking
    success: bool = True
    error_message: Optional[str] = None
    
    # Context metadata
    phase_numeric: Optional[int] = None  # Extracted year from phase (e.g., 1901)
    season_numeric: Optional[int] = None  # 0=Spring, 1=Fall, 2=Winter
    decision_type: Optional[str] = None  # 'orders', 'negotiations', 'planning'
    
    # Game state context
    supply_center_count: Optional[int] = None
    unit_count: Optional[int] = None
    
    # Performance metrics
    token_usage: Optional[Dict[str, int]] = None
    cost_estimate: Optional[float] = None
    
    # GRPO specific fields
    grpo_episode: Optional[int] = None
    grpo_step: Optional[int] = None
    grpo_reward: Optional[float] = None


class WandBLLMLogger:
    """
    Comprehensive W&B logger for all LLM interactions in the Diplomacy game.
    
    Handles structured logging of:
    - Order generation responses
    - Negotiation messages  
    - Strategic planning outputs
    - GRPO training interactions
    
    Supports parallel game separation and detailed analytics.
    """
    
    def __init__(self, 
                 project_name: str = "diplomacy-llm-interactions",
                 entity: Optional[str] = None,
                 enabled: bool = True):
        self.enabled = enabled and WANDB_AVAILABLE
        self.project_name = project_name
        self.entity = entity
        self.game_sessions: Dict[str, Dict] = {}  # Track active game sessions
        
        if not WANDB_AVAILABLE and enabled:
            logger.warning("W&B not available - LLM logging disabled. Install with: pip install wandb")
            self.enabled = False
        
        if self.enabled:
            logger.info("W&B LLM Logger initialized")
    
    def start_game_session(self, 
                          game_id: str, 
                          game_config: Dict[str, Any],
                          is_grpo_training: bool = False,
                          grpo_episode: Optional[int] = None) -> None:
        """Start tracking a new game session."""
        if not self.enabled:
            return
            
        session_info = {
            'game_id': game_id,
            'start_time': time.time(),
            'is_grpo_training': is_grpo_training,
            'grpo_episode': grpo_episode,
            'config': game_config,
            'interaction_count': 0,
            'interactions_by_type': {},
            'interactions_by_power': {},
        }
        
        self.game_sessions[game_id] = session_info
        
        # Initialize W&B run if this is the first game session
        if len(self.game_sessions) == 1:
            self._init_wandb()
        
        # Log game session start
        wandb.log({
            f'game_sessions/{game_id}/start_time': session_info['start_time'],
            f'game_sessions/{game_id}/is_grpo_training': 1 if is_grpo_training else 0,
            f'game_sessions/{game_id}/grpo_episode': grpo_episode or -1,
        })
        
        logger.info(f"Started W&B logging for game session: {game_id}")
    
    def log_llm_interaction(self, interaction: LLMInteraction) -> None:
        """Log a single LLM interaction to W&B."""
        if not self.enabled:
            return
        
        game_id = interaction.game_id
        
        # Ensure game session exists
        if game_id not in self.game_sessions:
            logger.warning(f"Game session {game_id} not found, creating default session")
            self.start_game_session(game_id, {})
        
        session = self.game_sessions[game_id]
        session['interaction_count'] += 1
        
        # Update session counters
        interaction_type = interaction.interaction_type
        power_name = interaction.power_name or 'system'
        
        session['interactions_by_type'][interaction_type] = \
            session['interactions_by_type'].get(interaction_type, 0) + 1
        session['interactions_by_power'][power_name] = \
            session['interactions_by_power'].get(power_name, 0) + 1
        
        # Create base metrics
        metrics = self._create_base_metrics(interaction)
        
        # Add session-specific metrics
        metrics.update({
            f'sessions/{game_id}/total_interactions': session['interaction_count'],
            f'sessions/{game_id}/interactions_by_type/{interaction_type}': 
                session['interactions_by_type'][interaction_type],
        })
        
        if power_name != 'system':
            metrics[f'sessions/{game_id}/interactions_by_power/{power_name}'] = \
                session['interactions_by_power'][power_name]
        
        # Add detailed interaction data
        metrics.update(self._create_interaction_metrics(interaction))
        
        # Log to W&B
        wandb.log(metrics)
        
        logger.debug(f"Logged LLM interaction: {interaction_type} for {power_name} in {game_id}")
    
    def log_bulk_interactions(self, interactions: List[LLMInteraction]) -> None:
        """Log multiple interactions efficiently."""
        if not self.enabled or not interactions:
            return
        
        # Group by game_id for batch processing
        by_game: Dict[str, List[LLMInteraction]] = {}
        for interaction in interactions:
            game_id = interaction.game_id
            if game_id not in by_game:
                by_game[game_id] = []
            by_game[game_id].append(interaction)
        
        # Process each game's interactions
        for game_id, game_interactions in by_game.items():
            if game_id not in self.game_sessions:
                self.start_game_session(game_id, {})
            
            # Create batch metrics
            batch_metrics = {}
            session = self.game_sessions[game_id]
            
            for interaction in game_interactions:
                session['interaction_count'] += 1
                
                # Update counters
                interaction_type = interaction.interaction_type
                power_name = interaction.power_name or 'system'
                
                session['interactions_by_type'][interaction_type] = \
                    session['interactions_by_type'].get(interaction_type, 0) + 1
                session['interactions_by_power'][power_name] = \
                    session['interactions_by_power'].get(power_name, 0) + 1
                
                # Add individual interaction metrics
                individual_metrics = self._create_interaction_metrics(interaction)
                batch_metrics.update(individual_metrics)
            
            # Add session summary metrics
            batch_metrics.update({
                f'sessions/{game_id}/total_interactions': session['interaction_count'],
                f'sessions/{game_id}/batch_size': len(game_interactions),
            })
            
            # Log batch
            wandb.log(batch_metrics)
        
        logger.info(f"Logged {len(interactions)} LLM interactions across {len(by_game)} games")
    
    def end_game_session(self, game_id: str) -> None:
        """End tracking for a game session and log summary."""
        if not self.enabled or game_id not in self.game_sessions:
            return
        
        session = self.game_sessions[game_id]
        end_time = time.time()
        duration = end_time - session['start_time']
        
        # Create final session summary
        summary_metrics = {
            f'sessions/{game_id}/end_time': end_time,
            f'sessions/{game_id}/duration_seconds': duration,
            f'sessions/{game_id}/final_interaction_count': session['interaction_count'],
        }
        
        # Add breakdown by type and power
        for interaction_type, count in session['interactions_by_type'].items():
            summary_metrics[f'sessions/{game_id}/final_by_type/{interaction_type}'] = count
        
        for power, count in session['interactions_by_power'].items():
            summary_metrics[f'sessions/{game_id}/final_by_power/{power}'] = count
        
        wandb.log(summary_metrics)
        
        # Clean up session
        del self.game_sessions[game_id]
        
        logger.info(f"Ended W&B logging for game session: {game_id} (duration: {duration:.1f}s)")
    
    def _init_wandb(self) -> None:
        """Initialize W&B run with enhanced configuration."""
        if not self.enabled:
            return
        
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
        
        # Define metrics for better visualization
        wandb.define_metric("timestamp")
        wandb.define_metric("sessions/*/total_interactions", step_metric="timestamp")
        wandb.define_metric("interactions/*/response_time_ms", step_metric="timestamp")
        wandb.define_metric("interactions/*/success_rate", step_metric="timestamp")
        
        logger.info("W&B run initialized for LLM interaction logging")
    
    def _create_base_metrics(self, interaction: LLMInteraction) -> Dict[str, Any]:
        """Create base metrics common to all interactions."""
        return {
            'timestamp': interaction.timestamp,
            'model_name_hash': self._hash_string(interaction.model_name),
            'interaction_type_numeric': self._encode_interaction_type(interaction.interaction_type),
            'success': 1 if interaction.success else 0,
            'response_time_ms': interaction.response_time_ms or 0,
        }
    
    def _create_interaction_metrics(self, interaction: LLMInteraction) -> Dict[str, Any]:
        """Create detailed metrics for a specific interaction."""
        game_id = interaction.game_id
        power_name = interaction.power_name or 'system'
        interaction_type = interaction.interaction_type
        
        # Create unique interaction ID for detailed logging
        interaction_id = self._generate_interaction_id(interaction)
        
        metrics = {
            # Detailed interaction data
            f'interactions/{interaction_id}/timestamp': interaction.timestamp,
            f'interactions/{interaction_id}/model_name': interaction.model_name,
            f'interactions/{interaction_id}/power_name': power_name,
            f'interactions/{interaction_id}/interaction_type': interaction_type,
            f'interactions/{interaction_id}/success': 1 if interaction.success else 0,
            f'interactions/{interaction_id}/response_time_ms': interaction.response_time_ms or 0,
            
            # Game context
            f'interactions/{interaction_id}/game_id': game_id,
            f'interactions/{interaction_id}/phase': interaction.phase,
            f'interactions/{interaction_id}/phase_numeric': interaction.phase_numeric or 0,
            f'interactions/{interaction_id}/season_numeric': interaction.season_numeric or 0,
            
            # Content metrics (length-based to avoid storing full text)
            f'interactions/{interaction_id}/input_length': len(interaction.raw_input_prompt),
            f'interactions/{interaction_id}/response_length': len(interaction.raw_response),
            f'interactions/{interaction_id}/output_length': len(interaction.processed_output or ""),
        }
        
        # Add optional fields if present
        if interaction.supply_center_count is not None:
            metrics[f'interactions/{interaction_id}/supply_center_count'] = interaction.supply_center_count
        
        if interaction.unit_count is not None:
            metrics[f'interactions/{interaction_id}/unit_count'] = interaction.unit_count
        
        if interaction.error_message:
            metrics[f'interactions/{interaction_id}/error_type'] = self._categorize_error(interaction.error_message)
        
        # GRPO specific metrics
        if interaction.grpo_episode is not None:
            metrics[f'interactions/{interaction_id}/grpo_episode'] = interaction.grpo_episode
        
        if interaction.grpo_step is not None:
            metrics[f'interactions/{interaction_id}/grpo_step'] = interaction.grpo_step
        
        if interaction.grpo_reward is not None:
            metrics[f'interactions/{interaction_id}/grpo_reward'] = interaction.grpo_reward
        
        # Token usage and cost
        if interaction.token_usage:
            for key, value in interaction.token_usage.items():
                metrics[f'interactions/{interaction_id}/tokens_{key}'] = value
        
        if interaction.cost_estimate:
            metrics[f'interactions/{interaction_id}/cost_estimate'] = interaction.cost_estimate
        
        return metrics
    
    def _generate_interaction_id(self, interaction: LLMInteraction) -> str:
        """Generate a unique but deterministic ID for an interaction."""
        # Use timestamp + game_id + power + type for uniqueness
        id_string = f"{interaction.timestamp}_{interaction.game_id}_{interaction.power_name}_{interaction.interaction_type}"
        return hashlib.md5(id_string.encode()).hexdigest()[:8]
    
    def _hash_string(self, s: str) -> int:
        """Convert string to numeric hash for W&B."""
        return hash(s) % (2**31)  # Keep within int32 range
    
    def _encode_interaction_type(self, interaction_type: str) -> int:
        """Encode interaction type as numeric for W&B visualization."""
        encoding = {
            'order_generation': 1,
            'negotiation': 2,
            'planning': 3,
            'grpo_training': 4,
            'system': 0
        }
        return encoding.get(interaction_type, 0)
    
    def _extract_phase_info(self, phase: str) -> tuple[Optional[int], Optional[int]]:
        """Extract numeric year and season from phase string."""
        try:
            if len(phase) >= 5 and phase[1:5].isdigit():
                year = int(phase[1:5])
                season_map = {'S': 0, 'F': 1, 'W': 2}  # Spring, Fall, Winter
                season = season_map.get(phase[0], 0)
                return year, season
        except (ValueError, IndexError):
            pass
        return None, None
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error messages for analytics."""
        error_lower = error_message.lower()
        
        if 'json' in error_lower or 'parse' in error_lower:
            return 'parsing_error'
        elif 'timeout' in error_lower or 'rate' in error_lower:
            return 'api_error'
        elif 'invalid' in error_lower or 'validation' in error_lower:
            return 'validation_error'
        elif 'connection' in error_lower or 'network' in error_lower:
            return 'network_error'
        else:
            return 'unknown_error'
    
    def get_session_stats(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get current statistics for a game session."""
        if not self.enabled or game_id not in self.game_sessions:
            return None
        
        session = self.game_sessions[game_id]
        current_time = time.time()
        
        return {
            'game_id': game_id,
            'duration': current_time - session['start_time'],
            'total_interactions': session['interaction_count'],
            'interactions_by_type': dict(session['interactions_by_type']),
            'interactions_by_power': dict(session['interactions_by_power']),
            'is_grpo_training': session.get('is_grpo_training', False),
            'grpo_episode': session.get('grpo_episode'),
        }


# Global logger instance
_global_logger: Optional[WandBLLMLogger] = None


def get_llm_logger() -> WandBLLMLogger:
    """Get the global LLM logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = WandBLLMLogger()
    return _global_logger


def initialize_llm_logging(project_name: str = "diplomacy-llm-interactions",
                          entity: Optional[str] = None,
                          enabled: bool = True) -> WandBLLMLogger:
    """Initialize the global LLM logger."""
    global _global_logger
    _global_logger = WandBLLMLogger(project_name, entity, enabled)
    return _global_logger


# Convenience functions for common use cases
def log_order_generation(game_id: str, 
                        model_name: str, 
                        power_name: str, 
                        phase: str,
                        prompt: str,
                        response: str,
                        orders: Optional[List[str]] = None,
                        success: bool = True,
                        error_message: Optional[str] = None,
                        response_time_ms: Optional[float] = None,
                        supply_centers: Optional[int] = None,
                        units: Optional[int] = None) -> None:
    """Log an order generation interaction."""
    logger = get_llm_logger()
    
    phase_numeric, season_numeric = logger._extract_phase_info(phase)
    
    interaction = LLMInteraction(
        model_name=model_name,
        power_name=power_name,
        game_id=game_id,
        phase=phase,
        interaction_type='order_generation',
        timestamp=time.time(),
        response_time_ms=response_time_ms,
        raw_input_prompt=prompt,
        raw_response=response,
        processed_output=json.dumps(orders) if orders else None,
        success=success,
        error_message=error_message,
        phase_numeric=phase_numeric,
        season_numeric=season_numeric,
        decision_type='orders',
        supply_center_count=supply_centers,
        unit_count=units,
    )
    
    logger.log_llm_interaction(interaction)


def log_negotiation(game_id: str,
                   model_name: str,
                   power_name: str,
                   phase: str,
                   prompt: str,
                   response: str,
                   messages: Optional[List[Dict]] = None,
                   success: bool = True,
                   error_message: Optional[str] = None,
                   response_time_ms: Optional[float] = None) -> None:
    """Log a negotiation interaction."""
    logger = get_llm_logger()
    
    phase_numeric, season_numeric = logger._extract_phase_info(phase)
    
    interaction = LLMInteraction(
        model_name=model_name,
        power_name=power_name,
        game_id=game_id,
        phase=phase,
        interaction_type='negotiation',
        timestamp=time.time(),
        response_time_ms=response_time_ms,
        raw_input_prompt=prompt,
        raw_response=response,
        processed_output=json.dumps(messages) if messages else None,
        success=success,
        error_message=error_message,
        phase_numeric=phase_numeric,
        season_numeric=season_numeric,
        decision_type='negotiations',
    )
    
    logger.log_llm_interaction(interaction)


def log_planning(game_id: str,
                model_name: str,
                power_name: str,
                phase: str,
                prompt: str,
                response: str,
                success: bool = True,
                error_message: Optional[str] = None,
                response_time_ms: Optional[float] = None) -> None:
    """Log a planning interaction."""
    logger = get_llm_logger()
    
    phase_numeric, season_numeric = logger._extract_phase_info(phase)
    
    interaction = LLMInteraction(
        model_name=model_name,
        power_name=power_name,
        game_id=game_id,
        phase=phase,
        interaction_type='planning',
        timestamp=time.time(),
        response_time_ms=response_time_ms,
        raw_input_prompt=prompt,
        raw_response=response,
        processed_output=response,  # For planning, the response is the output
        success=success,
        error_message=error_message,
        phase_numeric=phase_numeric,
        season_numeric=season_numeric,
        decision_type='planning',
    )
    
    logger.log_llm_interaction(interaction)


def log_grpo_interaction(game_id: str,
                        model_name: str,
                        episode: int,
                        step: int,
                        prompt: str,
                        response: str,
                        reward: Optional[float] = None,
                        power_name: Optional[str] = None,
                        success: bool = True,
                        error_message: Optional[str] = None,
                        response_time_ms: Optional[float] = None) -> None:
    """Log a GRPO training interaction."""
    logger = get_llm_logger()
    
    interaction = LLMInteraction(
        model_name=model_name,
        power_name=power_name,
        game_id=game_id,
        phase=f"grpo_e{episode}_s{step}",
        interaction_type='grpo_training',
        timestamp=time.time(),
        response_time_ms=response_time_ms,
        raw_input_prompt=prompt,
        raw_response=response,
        success=success,
        error_message=error_message,
        decision_type='grpo_training',
        grpo_episode=episode,
        grpo_step=step,
        grpo_reward=reward,
    )
    
    logger.log_llm_interaction(interaction)