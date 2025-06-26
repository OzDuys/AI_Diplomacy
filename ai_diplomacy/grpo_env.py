# ai_diplomacy/grpo_env.py
"""
DiplomacyMultiTurnEnv - Environment wrapper for online GRPO training

This module implements the multi-turn environment interface required by willccbb/verifiers
GRPO trainer, adapting the existing Diplomacy game for reinforcement learning.
"""

import logging
import json
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from diplomacy import Game
from .agent import DiplomacyAgent, ALL_POWERS
from .clients import load_model_client
from .game_history import GameHistory
from .prompt_constructor import construct_order_generation_prompt, build_context_prompt
from .grpo_rewards import AllianceTracker, calculate_step_rewards, calculate_final_rewards

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of decisions agents make in Diplomacy"""
    ORDER_GENERATION = "orders"
    NEGOTIATION = "negotiation"


@dataclass
class GameState:
    """Encapsulates current game state for GRPO training"""
    phase: str
    board_state: Dict[str, Any]
    agent_relationships: Dict[str, Dict[str, str]]
    negotiations: List[Dict[str, Any]]
    supply_centers: Dict[str, List[str]]
    units: Dict[str, List[str]]


class DiplomacyMultiTurnEnv:
    """
    Multi-turn environment for GRPO training on Diplomacy game.
    
    Extends the willccbb/verifiers MultiTurnEnv interface to handle:
    - 7-agent self-play with batched generation
    - Order generation and negotiation training
    - Alliance formation reward tracking
    - Full game episodes (1901-1910)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_year: int = 1910,
        num_negotiation_rounds: int = 3,
        prompts_dir: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize the Diplomacy GRPO environment.
        
        Args:
            model_name: Base model for all agents (same model for self-play)
            max_year: Maximum year to play until
            num_negotiation_rounds: Number of negotiation rounds per phase
            prompts_dir: Directory containing prompt templates
            random_seed: Fixed seed for reproducible batched generation
        """
        self.model_name = model_name
        self.max_year = max_year
        self.num_negotiation_rounds = num_negotiation_rounds
        self.prompts_dir = prompts_dir
        self.random_seed = random_seed
        
        # Game components
        self.game = Game()
        self.game_history = GameHistory()
        self.alliance_tracker = AllianceTracker()
        
        # Agents (7 powers, all using same base model)
        self.agents: Dict[str, DiplomacyAgent] = {}
        self._initialize_agents()
        
        # Training state
        self.current_decision_type = DecisionType.ORDER_GENERATION
        self.current_phase = "S1901M"
        self.episode_data = []  # Store all decisions for end-game GRPO update
        self.step_counter = 0
        
        logger.info(f"Initialized DiplomacyMultiTurnEnv with {len(self.agents)} agents")
    
    def _initialize_agents(self):
        """Initialize 7 agents, one for each power, using the same base model"""
        for power in ALL_POWERS:
            client = load_model_client(self.model_name, prompts_dir=self.prompts_dir)
            agent = DiplomacyAgent(
                power_name=power,
                client=client,
                prompts_dir=self.prompts_dir
            )
            # Initialize agent state (goals, relationships)
            agent.initialize_agent_state_ext(self.game, self.game_history)
            self.agents[power] = agent
            
    def is_completed(self) -> bool:
        """
        Check if the current game episode is complete.
        
        Returns:
            True if game should end (victory condition or max year reached)
        """
        # Check victory condition
        if self.game.is_game_done:
            logger.info(f"Game completed with victory: {self.game.get_current_phase()}")
            return True
            
        # Check max year
        current_year = int(self.current_phase[1:5])
        if current_year > self.max_year:
            logger.info(f"Game completed at max year {self.max_year}")
            return True
            
        return False
    
    def get_current_state(self) -> GameState:
        """Get current game state for prompt construction"""
        return GameState(
            phase=self.current_phase,
            board_state=self.game.get_state(),
            agent_relationships={
                power: agent.relationships 
                for power, agent in self.agents.items()
            },
            negotiations=self.game_history.get_negotiations_for_phase(self.current_phase),
            supply_centers={
                power: list(self.game.get_power(power).centers)
                for power in ALL_POWERS
            },
            units={
                power: [str(unit) for unit in self.game.get_power(power).units]
                for power in ALL_POWERS
            }
        )
    
    def get_batch_prompts(self) -> List[str]:
        """
        Generate prompts for all 7 agents for current decision type.
        
        Returns:
            List of 7 prompts (one per agent) for batched generation
        """
        prompts = []
        current_state = self.get_current_state()
        
        for power in sorted(ALL_POWERS):  # Consistent ordering
            agent = self.agents[power]
            
            if self.current_decision_type == DecisionType.ORDER_GENERATION:
                # Generate military orders prompt
                possible_orders = self.game.get_all_possible_orders()[power]
                prompt = construct_order_generation_prompt(
                    agent=agent,
                    game=self.game,
                    game_history=self.game_history,
                    possible_orders=possible_orders,
                    current_phase=self.current_phase
                )
            else:  # NEGOTIATION
                # Generate negotiation response prompt
                conversation_context = self._get_negotiation_context(power)
                prompt = build_context_prompt(
                    agent=agent,
                    game=self.game,
                    conversation_context=conversation_context,
                    phase=self.current_phase
                )
            
            prompts.append(prompt)
            
        return prompts
    
    def _get_negotiation_context(self, power: str) -> str:
        """Get negotiation context for a specific power"""
        # Get recent messages involving this power
        negotiations = self.game_history.get_negotiations_for_phase(self.current_phase)
        relevant_messages = [
            msg for msg in negotiations 
            if msg.get('sender') == power or msg.get('recipient') == power
        ]
        
        # Format as conversation context
        context_lines = []
        for msg in relevant_messages[-5:]:  # Last 5 messages
            sender = msg.get('sender', 'Unknown')
            recipient = msg.get('recipient', 'All')
            content = msg.get('content', '')
            context_lines.append(f"{sender} to {recipient}: {content}")
            
        return "\n".join(context_lines)
    
    def process_batch_responses(self, responses: List[str]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Process batch of responses from all 7 agents and advance game state.
        
        Args:
            responses: List of 7 responses (one per agent)
            
        Returns:
            Tuple of (rewards, step_info)
        """
        assert len(responses) == 7, f"Expected 7 responses, got {len(responses)}"
        
        step_info = {
            'phase': self.current_phase,
            'decision_type': self.current_decision_type.value,
            'step': self.step_counter
        }
        
        if self.current_decision_type == DecisionType.ORDER_GENERATION:
            rewards = self._process_orders(responses, step_info)
        else:  # NEGOTIATION
            rewards = self._process_negotiations(responses, step_info)
            
        self.step_counter += 1
        return rewards, step_info
    
    def _process_orders(self, order_responses: List[str], step_info: Dict) -> List[float]:
        """Process military orders from all agents"""
        # Parse orders from responses
        power_orders = {}
        for i, power in enumerate(sorted(ALL_POWERS)):
            try:
                # Extract orders from LLM response
                orders = self._parse_orders_from_response(order_responses[i])
                power_orders[power] = orders
            except Exception as e:
                logger.error(f"Failed to parse orders for {power}: {e}")
                power_orders[power] = []  # Invalid orders
        
        # Submit orders to game
        for power, orders in power_orders.items():
            for order in orders:
                self.game.set_orders(power, [order])
        
        # Process game phase
        self.game.process()
        
        # Calculate step rewards
        rewards = calculate_step_rewards(
            game=self.game,
            agents=self.agents,
            alliance_tracker=self.alliance_tracker,
            phase=self.current_phase
        )
        
        # Store episode data
        prompts = self.get_batch_prompts()  # Get the prompts that were used
        self.episode_data.append({
            'prompts': prompts,
            'responses': order_responses,
            'rewards': rewards,
            'step_info': step_info
        })
        
        # Advance to next phase or negotiation
        self._advance_phase()
        
        return rewards
    
    def _process_negotiations(self, negotiation_responses: List[str], step_info: Dict) -> List[float]:
        """Process negotiation messages from all agents"""
        # Store negotiation messages
        for i, power in enumerate(sorted(ALL_POWERS)):
            message = negotiation_responses[i].strip()
            if message and message != "NONE":
                # Add to game history
                self.game_history.add_negotiation_message(
                    phase=self.current_phase,
                    sender=power,
                    recipient="ALL",  # Public diplomacy for now
                    content=message
                )
        
        # Calculate negotiation rewards (alliance formation, etc.)
        rewards = calculate_step_rewards(
            game=self.game,
            agents=self.agents,
            alliance_tracker=self.alliance_tracker,
            phase=self.current_phase,
            decision_type="negotiation"
        )
        
        # Store episode data
        prompts = self.get_batch_prompts()
        self.episode_data.append({
            'prompts': prompts,
            'responses': negotiation_responses,
            'rewards': rewards,
            'step_info': step_info
        })
        
        # Switch back to order generation
        self.current_decision_type = DecisionType.ORDER_GENERATION
        
        return rewards
    
    def _parse_orders_from_response(self, response: str) -> List[str]:
        """Extract valid orders from LLM response"""
        # Simple parsing - look for lines that look like orders
        orders = []
        for line in response.split('\n'):
            line = line.strip()
            if line and ('A ' in line or 'F ' in line):
                orders.append(line)
        return orders
    
    def _advance_phase(self):
        """Advance to next phase or negotiation round"""
        if self.current_decision_type == DecisionType.ORDER_GENERATION:
            # Check if we should do negotiations
            if self.num_negotiation_rounds > 0:
                self.current_decision_type = DecisionType.NEGOTIATION
            else:
                self._advance_game_phase()
        else:
            # Negotiation completed, advance game phase
            self._advance_game_phase()
    
    def _advance_game_phase(self):
        """Advance to next game phase"""
        self.current_phase = self.game.get_current_phase()
        self.current_decision_type = DecisionType.ORDER_GENERATION
    
    def get_final_rewards(self) -> List[float]:
        """Calculate final rewards when game is complete"""
        return calculate_final_rewards(
            game=self.game,
            agents=self.agents,
            alliance_tracker=self.alliance_tracker
        )
    
    def get_episode_data(self) -> List[Dict[str, Any]]:
        """Get all episode data for GRPO training"""
        return self.episode_data
    
    def reset(self):
        """Reset environment for new game episode"""
        self.game = Game()
        self.game_history = GameHistory()
        self.alliance_tracker = AllianceTracker()
        self.current_phase = "S1901M"
        self.current_decision_type = DecisionType.ORDER_GENERATION
        self.episode_data = []
        self.step_counter = 0
        
        # Reinitialize agents
        self._initialize_agents()
        
        logger.info("Environment reset for new episode")