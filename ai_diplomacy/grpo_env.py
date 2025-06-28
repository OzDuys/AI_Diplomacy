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
from .initialization import initialize_agent_state_ext
from .enhanced_wandb_logger import get_enhanced_logger

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
        
        # Training state
        self.current_decision_type = DecisionType.ORDER_GENERATION
        self.current_phase = "S1901M"
        self.episode_data = []  # Store all decisions for end-game GRPO update
        self.step_counter = 0
        
        # Add initial phase to game history
        self.game_history.add_phase(self.current_phase)
        
        # Agents (7 powers, all using same base model)
        self.agents: Dict[str, DiplomacyAgent] = {}
        self._initialize_agents()
        
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
            # Simple initialization for GRPO training (no LLM calls)
            self._initialize_agent_simple(agent, power)
            self.agents[power] = agent
    
    def _initialize_agent_simple(self, agent: DiplomacyAgent, power: str):
        """Simple initialization without LLM calls for faster GRPO training"""
        # Set basic initial goals based on power
        initial_goals = {
            "AUSTRIA": ["Secure the Balkans", "Form alliance with Germany", "Contain Russia"],
            "ENGLAND": ["Control the seas", "Secure Scotland", "Build naval supremacy"],
            "FRANCE": ["Secure Spain", "Form alliance with Russia", "Contain Germany"],
            "GERMANY": ["Secure Scandinavia", "Form central alliance", "Control North Sea"],
            "ITALY": ["Secure the Mediterranean", "Form Lepanto alliance", "Build fleet"],
            "RUSSIA": ["Secure Scandinavia", "Form western alliance", "Control Black Sea"],
            "TURKEY": ["Secure the Black Sea", "Form Juggernaut alliance", "Build to Mediterranean"]
        }
        
        agent.goals = initial_goals.get(power, ["Survive", "Gain territory", "Form alliances"])
        
        # Initialize neutral relationships (will be updated during play)
        agent.relationships = {p: "Neutral" for p in ALL_POWERS if p != power}
            
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
            
        # Check max year - safely parse phase
        try:
            if len(self.current_phase) >= 5 and self.current_phase[1:5].isdigit():
                current_year = int(self.current_phase[1:5])
                if current_year > self.max_year:
                    logger.info(f"Game completed at max year {self.max_year}")
                    return True
            else:
                logger.warning(f"Invalid phase format for year parsing: {self.current_phase}")
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing year from phase {self.current_phase}: {e}")
            
        return False
    
    def get_current_state(self) -> GameState:
        """Get current game state for prompt construction"""
        # Safely get board state
        board_state = self.game.get_state()
        
        # Validate board state has required keys
        required_keys = ['units', 'centers', 'phase']
        for key in required_keys:
            if key not in board_state:
                logger.error(f"Missing required key in board_state: {key}")
                board_state[key] = {}  # Provide default empty dict
        
        # Safely get supply centers and units
        supply_centers = {}
        units = {}
        
        for power in ALL_POWERS:
            try:
                if power in self.game.powers:
                    power_obj = self.game.get_power(power)
                    supply_centers[power] = list(power_obj.centers) if power_obj.centers else []
                    units[power] = [str(unit) for unit in power_obj.units] if power_obj.units else []
                else:
                    logger.warning(f"Power {power} not found in game")
                    supply_centers[power] = []
                    units[power] = []
            except Exception as e:
                logger.error(f"Error getting state for power {power}: {e}")
                supply_centers[power] = []
                units[power] = []
        
        return GameState(
            phase=self.current_phase,
            board_state=board_state,
            agent_relationships={
                power: agent.relationships 
                for power, agent in self.agents.items()
            },
            negotiations=self._get_negotiations_for_phase(self.current_phase),
            supply_centers=supply_centers,
            units=units
        )
    
    def _get_negotiations_for_phase(self, phase: str) -> List[Dict[str, Any]]:
        """Get negotiations/messages for a specific phase"""
        negotiations = []
        
        # Find the phase in game history
        for game_phase in self.game_history.phases:
            if game_phase.name == phase:
                # Convert messages to dictionaries
                for message in game_phase.messages:
                    negotiations.append({
                        'sender': message.sender,
                        'recipient': message.recipient,
                        'content': message.content
                    })
                break
        
        return negotiations
    
    def _get_possible_orders_for_power(self, power: str) -> List[str]:
        """Get all possible orders for a specific power"""
        all_orders = self.game.get_all_possible_orders()
        
        # Validate power exists
        if power not in self.game.powers:
            logger.error(f"Power {power} not found in game")
            return []
            
        power_obj = self.game.get_power(power)
        
        possible_orders = []
        for unit in power_obj.units:
            # Safely extract location from unit (e.g., 'A VIE' -> 'VIE')
            unit_str = str(unit)
            parts = unit_str.split()
            
            if len(parts) >= 2:
                unit_location = parts[1]
                if unit_location in all_orders:
                    possible_orders.extend(all_orders[unit_location])
                else:
                    logger.debug(f"No orders found for unit location: {unit_location}")
            else:
                logger.error(f"Invalid unit format for {power}: '{unit_str}' - expected format like 'A VIE'")
        
        return possible_orders
    
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
                possible_orders = self._get_possible_orders_for_power(power)
                
                # Debug: Log possible orders for troubleshooting (reduced verbosity)
                if not possible_orders:
                    logger.warning(f"No possible orders found for {power}!")
                elif len(possible_orders) == 0:
                    logger.debug(f"Empty order set for {power}")
                # Removed verbose order listing to reduce log clutter
                
                prompt = construct_order_generation_prompt(
                    system_prompt=agent.client.system_prompt,
                    game=self.game,
                    board_state=self.game.get_state(),
                    power_name=power,
                    possible_orders={power: possible_orders},  # Format as expected
                    game_history=self.game_history,
                    agent_goals=agent.goals,
                    agent_relationships=agent.relationships,
                    agent_private_diary_str="\n".join(agent.private_diary),
                    prompts_dir=self.prompts_dir
                )
            else:  # NEGOTIATION
                # Generate negotiation response prompt
                conversation_context = self._get_negotiation_context(power)
                prompt = build_context_prompt(
                    game=self.game,
                    board_state=self.game.get_state(),
                    power_name=power,
                    possible_orders={power: []},  # No orders needed for negotiation
                    game_history=self.game_history,
                    agent_goals=agent.goals,
                    agent_relationships=agent.relationships,
                    agent_private_diary="\n".join(agent.private_diary),
                    prompts_dir=self.prompts_dir
                )
            
            prompts.append(prompt)
            
        # Debug: Only log basic info, no prompt content
        if prompts:
            logger.debug(f"Generated {len(prompts)} prompts for decision type: {self.current_decision_type.value}")
        else:
            logger.warning("No prompts generated!")
            
        return prompts
    
    def _get_negotiation_context(self, power: str) -> str:
        """Get negotiation context for a specific power"""
        # Get recent messages involving this power
        negotiations = self._get_negotiations_for_phase(self.current_phase)
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
        # Validate input
        if len(order_responses) != 7:
            logger.error(f"Expected 7 order responses, got {len(order_responses)}")
            return [0.0] * 7
        
        # Parse orders from responses
        power_orders = {}
        for i, power in enumerate(sorted(ALL_POWERS)):
            try:
                # Validate power exists
                if power not in self.game.powers:
                    logger.error(f"Power {power} not found in game")
                    power_orders[power] = []
                    continue
                    
                # Extract orders from LLM response, passing the power name for better logging
                orders = self._parse_orders_from_response(order_responses[i], power)
                
                # If no orders found, provide default hold orders for testing
                if not orders:
                    hold_orders = self._get_default_hold_orders(power)
                    if hold_orders:
                        # Log default orders with clear formatting
                        logger.info(f"Using default hold orders for {power}:")
                        for order in hold_orders:
                            logger.info(f"  - {order}")
                        orders = hold_orders
                
                power_orders[power] = orders
            except Exception as e:
                logger.error(f"Failed to parse orders for {power}: {e}")
                power_orders[power] = []  # Invalid orders
        
        # Submit orders to game
        for power, orders in power_orders.items():
            if orders:  # Only set orders if there are any
                # Log orders with clear formatting
                logger.info(f"Setting orders for {power}:")
                for order in orders:
                    logger.info(f"  - {order}")
                try:
                    self.game.set_orders(power, orders)
                except Exception as e:
                    logger.error(f"Failed to set orders for {power} - Error: {e}")
                    # Log full order list for debugging without truncation
                    if len(str(orders)) > 100:  # Only use detailed logging for long order lists
                        logger.error(f"Invalid orders for {power}:")
                        for idx, order in enumerate(orders):
                            logger.error(f"  [{idx}]: {order}")
                    else:
                        logger.error(f"Invalid orders: {orders}")
            else:
                logger.warning(f"No orders parsed for {power} - will skip this power's turn")
        
        # Store supply centers before processing for comparison
        old_centers = {}
        for power in ALL_POWERS:
            if power in self.game.powers:
                old_centers[power] = list(self.game.get_power(power).centers)
        
        # Process game phase
        try:
            self.game.process()
            logger.info(f"Game phase {self.current_phase} processed successfully")
            
            # Log supply center changes to enhanced logger
            try:
                enhanced_logger = get_enhanced_logger()
                for power in ALL_POWERS:
                    if power in self.game.powers:
                        new_centers = list(self.game.get_power(power).centers)
                        enhanced_logger.log_supply_center_change(
                            phase=self.current_phase,
                            power=power,
                            new_centers=new_centers,
                            old_centers=old_centers.get(power, [])
                        )
            except Exception as e:
                logger.warning(f"Failed to log supply center changes: {e}")
                
        except Exception as e:
            logger.error(f"Failed to process game phase {self.current_phase}: {e}")
            # Continue with rewards calculation even if processing failed
            return [0.0] * 7
        
        # Calculate step rewards
        rewards = calculate_step_rewards(
            game=self.game,
            agents=self.agents,
            alliance_tracker=self.alliance_tracker,
            phase=self.current_phase
        )
        
        # Log game metrics to enhanced logger
        try:
            enhanced_logger = get_enhanced_logger()
            enhanced_logger.log_game_metrics(
                phase=self.current_phase,
                game_state=self.game.get_state(),
                step=self.step_counter
            )
        except Exception as e:
            logger.warning(f"Failed to log game metrics: {e}")
        
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
                self.game_history.add_message(
                    phase_name=self.current_phase,
                    sender=power,
                    recipient="ALL",  # Public diplomacy for now
                    message_content=message
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
    
    def _parse_orders_from_response(self, response: str, power: str = "Unknown") -> List[str]:
        """Extract valid orders from LLM response (handles both JSON and plain text)"""
        orders = []
        
        # Always log the full LLM output for debugging purposes
        logger.info(f"=== FULL LLM OUTPUT FROM {power} ===")
        self._log_full_response(power, response, "LLM OUTPUT")
        
        # Use the power-aware logging utility if power is provided
        # Log the raw response for debugging - full response will be logged if no orders are found
        logger.debug(f"Raw LLM response from {power} (first 200 chars): {response[:200]}...")
        
        # First, try to parse as JSON (most common format from LLMs)
        try:
            # Look for JSON in the response
            import json
            import re
            
            # Try multiple approaches to find JSON with orders
            
            # Approach 1: Find complete JSON objects with "orders" key
            json_patterns = [
                r'\{[^{}]*"orders"[^{}]*\}',  # Simple JSON
                r'\{(?:[^{}]*\{[^{}]*\})*[^{}]*"orders"[^{}]*\}',  # Nested JSON
                r'"orders"\s*:\s*\[[^\]]*\]',  # Just the orders array
            ]
            
            for pattern in json_patterns:
                json_matches = re.findall(pattern, response, re.DOTALL)
                
                for json_str in json_matches:
                    try:
                        # If it's just the orders array, wrap it in an object
                        if json_str.startswith('"orders"'):
                            json_str = '{' + json_str + '}'
                            
                        parsed = json.loads(json_str)
                        if "orders" in parsed and isinstance(parsed["orders"], list):
                            for order in parsed["orders"]:
                                if isinstance(order, str) and order.strip():
                                    # Clean up the order
                                    cleaned_order = ' '.join(order.strip().split())
                                    if cleaned_order.startswith('A ') or cleaned_order.startswith('F '):
                                        orders.append(cleaned_order)
                            if orders:  # If we found orders, stop looking
                                break
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse JSON '{json_str[:50]}...': {e}")
                        continue
                
                if orders:  # If we found orders, stop trying patterns
                    break
        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}")
        
        # If JSON parsing didn't work, try plain text parsing
        if not orders:
            for line in response.split('\n'):
                line = line.strip()
                # Look for lines that start with unit notation (A or F followed by space)
                if line and (line.startswith('A ') or line.startswith('F ')):
                    # Clean up the order - remove extra whitespace
                    cleaned_order = ' '.join(line.split())
                    orders.append(cleaned_order)
        
        # If still no orders found, log the full response using our utility method
        if not orders:
            logger.warning(f"No valid orders found in response from {power} - logging full response:")
            self._log_full_response(power, response, "FAILED ORDER PARSING")
                
        # Log parsed orders for debugging
        logger.debug(f"Parsed orders from {power}: {orders}")
        return orders
    
    def _get_default_hold_orders(self, power: str) -> List[str]:
        """Generate default hold orders for a power (for testing when LLM fails)"""
        try:
            if power not in self.game.powers:
                return []
                
            power_obj = self.game.get_power(power)
            hold_orders = []
            
            for unit in power_obj.units:
                unit_str = str(unit)
                parts = unit_str.split()
                if len(parts) >= 2:
                    unit_type = parts[0]  # 'A' or 'F'
                    location = parts[1]   # 'VIE', 'TRI', etc.
                    hold_order = f"{unit_type} {location} H"
                    hold_orders.append(hold_order)
            
            return hold_orders
        except Exception as e:
            logger.error(f"Failed to generate default hold orders for {power}: {e}")
            return []
    
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
        
        # Add initial phase to game history
        self.game_history.add_phase(self.current_phase)
        
        # Reinitialize agents
        self._initialize_agents()
        
        logger.info("Environment reset for new episode")
    
    def _log_full_response(self, power: str, response: str, context: str = "LLM Response") -> None:
        """
        Log the full LLM response without truncation to aid in debugging.
        
        Args:
            power: The power (country) the response is for
            response: The full LLM response text
            context: Context information about what the response is for
        """
        # Log header with clear separation
        logger.info(f"===== FULL {context} FOR {power} =====")
        
        # Break the response into chunks of 1000 characters for better logging
        chunk_size = 5000
        num_chunks = (len(response) + chunk_size - 1) // chunk_size
        
        # Only warn about chunking for large responses
        if num_chunks > 1:
            logger.info(f"Response is {len(response)} characters, logging in {num_chunks} chunks")
        
        # Log each chunk with a clear prefix
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i+chunk_size]
            chunk_num = i // chunk_size + 1
            if num_chunks > 1:
                logger.info(f"CHUNK {chunk_num}/{num_chunks}: {chunk}")
            else:
                logger.info(f"{chunk}")
                
        # Log footer
        logger.info(f"===== END {context} FOR {power} =====")