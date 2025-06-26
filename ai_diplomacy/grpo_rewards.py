# ai_diplomacy/grpo_rewards.py
"""
Alliance detection and reward calculation for GRPO training

This module implements reward functions for Diplomacy GRPO training, focusing on:
- Alliance formation and maintenance rewards
- Victory condition rewards  
- Diplomatic success metrics
- Supply center progression rewards
"""

import logging
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from diplomacy import Game
from .agent import DiplomacyAgent, ALL_POWERS

logger = logging.getLogger(__name__)


@dataclass
class AllianceState:
    """Tracks alliance between two powers"""
    power1: str
    power2: str
    formed_phase: str
    strength: float = 0.0  # 0.0 = neutral, 1.0 = strong alliance
    last_updated: str = ""
    broken: bool = False
    betrayal_detected: bool = False


@dataclass
class PowerStats:
    """Track statistics for a power across the game"""
    supply_centers: List[int] = field(default_factory=list)
    units: List[int] = field(default_factory=list)
    alliances_formed: int = 0
    alliances_broken: int = 0
    successful_negotiations: int = 0
    betrayals_committed: int = 0
    betrayals_suffered: int = 0
    eliminated_phase: Optional[str] = None


class AllianceTracker:
    """
    Tracks alliance formation, maintenance, and betrayal throughout the game.
    Used to calculate diplomatic rewards for GRPO training.
    """
    
    def __init__(self):
        self.alliances: Dict[Tuple[str, str], AllianceState] = {}
        self.power_stats: Dict[str, PowerStats] = {
            power: PowerStats() for power in ALL_POWERS
        }
        self.phase_history: List[str] = []
        
    def _alliance_key(self, power1: str, power2: str) -> Tuple[str, str]:
        """Create consistent key for alliance lookup"""
        return tuple(sorted([power1, power2]))
    
    def update_relationships(self, agents: Dict[str, DiplomacyAgent], phase: str):
        """
        Update alliance tracking based on agent relationship changes.
        
        Args:
            agents: Dictionary of power -> DiplomacyAgent
            phase: Current game phase
        """
        self.phase_history.append(phase)
        
        # Check for new alliances and relationship changes
        for power1, agent1 in agents.items():
            for power2, relationship in agent1.relationships.items():
                if power1 >= power2:  # Avoid duplicate checking
                    continue
                    
                alliance_key = self._alliance_key(power1, power2)
                current_alliance = self.alliances.get(alliance_key)
                
                # Detect alliance formation
                if relationship in ["Ally", "Friendly"] and not current_alliance:
                    # Check if both sides consider each other allies/friendly
                    reciprocal_relationship = agents[power2].relationships.get(power1, "Neutral")
                    if reciprocal_relationship in ["Ally", "Friendly"]:
                        self._form_alliance(power1, power2, phase, relationship)
                
                # Detect alliance strengthening/weakening
                elif current_alliance and not current_alliance.broken:
                    self._update_alliance_strength(alliance_key, relationship, agents[power2].relationships.get(power1, "Neutral"))
                
                # Detect betrayal
                elif current_alliance and not current_alliance.broken:
                    if relationship in ["Enemy", "Unfriendly"]:
                        self._detect_betrayal(alliance_key, power1, power2, phase)
    
    def _form_alliance(self, power1: str, power2: str, phase: str, relationship_type: str):
        """Record new alliance formation"""
        alliance_key = self._alliance_key(power1, power2)
        strength = 1.0 if relationship_type == "Ally" else 0.7  # Friendly = weaker alliance
        
        self.alliances[alliance_key] = AllianceState(
            power1=power1,
            power2=power2,
            formed_phase=phase,
            strength=strength,
            last_updated=phase
        )
        
        # Update stats
        self.power_stats[power1].alliances_formed += 1
        self.power_stats[power2].alliances_formed += 1
        
        logger.debug(f"Alliance formed between {power1} and {power2} in {phase}")
    
    def _update_alliance_strength(self, alliance_key: Tuple[str, str], rel1: str, rel2: str):
        """Update alliance strength based on mutual relationship levels"""
        alliance = self.alliances[alliance_key]
        
        # Calculate strength based on mutual relationship
        strength_map = {"Ally": 1.0, "Friendly": 0.7, "Neutral": 0.3, "Unfriendly": 0.1, "Enemy": 0.0}
        avg_strength = (strength_map.get(rel1, 0.3) + strength_map.get(rel2, 0.3)) / 2
        
        alliance.strength = avg_strength
    
    def _detect_betrayal(self, alliance_key: Tuple[str, str], betrayer: str, victim: str, phase: str):
        """Detect and record betrayal"""
        alliance = self.alliances[alliance_key]
        alliance.broken = True
        alliance.betrayal_detected = True
        alliance.last_updated = phase
        
        # Update stats
        self.power_stats[betrayer].betrayals_committed += 1
        self.power_stats[victim].betrayals_suffered += 1
        self.power_stats[betrayer].alliances_broken += 1
        
        logger.debug(f"Betrayal detected: {betrayer} betrayed {victim} in {phase}")
    
    def get_alliance_rewards(self, phase: str) -> Dict[str, float]:
        """
        Calculate alliance-related rewards for all powers.
        
        Returns:
            Dictionary of power -> reward value
        """
        rewards = {power: 0.0 for power in ALL_POWERS}
        
        for alliance in self.alliances.values():
            if alliance.last_updated == phase:
                if not alliance.broken and not alliance.betrayal_detected:
                    # Reward alliance formation/maintenance
                    alliance_reward = 5.0 * alliance.strength
                    rewards[alliance.power1] += alliance_reward
                    rewards[alliance.power2] += alliance_reward
                elif alliance.betrayal_detected:
                    # Penalize betrayal, reward victim
                    betrayer = alliance.power1 if alliance.last_updated == phase else alliance.power2
                    victim = alliance.power2 if betrayer == alliance.power1 else alliance.power1
                    rewards[betrayer] -= 10.0
                    rewards[victim] += 2.0  # Sympathy bonus
        
        return rewards
    
    def get_active_alliances(self, power: str) -> List[str]:
        """Get list of powers that have active alliances with given power"""
        allies = []
        for alliance in self.alliances.values():
            if not alliance.broken and alliance.strength > 0.5:
                if alliance.power1 == power:
                    allies.append(alliance.power2)
                elif alliance.power2 == power:
                    allies.append(alliance.power1)
        return allies


def calculate_step_rewards(
    game: Game,
    agents: Dict[str, DiplomacyAgent],
    alliance_tracker: AllianceTracker,
    phase: str,
    decision_type: str = "orders"
) -> List[float]:
    """
    Calculate step rewards for all agents after a decision.
    
    Args:
        game: Current game state
        agents: Dictionary of power -> agent
        alliance_tracker: Alliance tracking system
        phase: Current game phase
        decision_type: Type of decision ("orders" or "negotiation")
        
    Returns:
        List of 7 rewards (one per power, in alphabetical order)
    """
    rewards = []
    center_changes = {}
    
    # Update alliance tracker with current relationships
    alliance_tracker.update_relationships(agents, phase)
    
    # Get alliance rewards
    alliance_rewards = alliance_tracker.get_alliance_rewards(phase)
    
    for power in sorted(ALL_POWERS):
        reward = 0.0
        game_power = game.get_power(power)
        stats = alliance_tracker.power_stats[power]
        
        if decision_type == "orders":
            # Territory/military rewards
            current_centers = len(game_power.centers)
            
            # Calculate change from start of game (first entry in stats)
            if stats.supply_centers:
                # Change from previous phase
                phase_change = current_centers - stats.supply_centers[-1]
                # Change from start of game
                game_change = current_centers - stats.supply_centers[0]
                reward += phase_change * 2.0  # +2 per center gained, -2 per center lost
            else:
                # First turn - initialize with starting centers
                game_change = 0  # No change from start
                phase_change = 0
            
            center_changes[power] = game_change
            
            # Update stats
            stats.supply_centers.append(current_centers)
            stats.units.append(len(game_power.units))
            
            # Elimination penalty
            if current_centers == 0 and not stats.eliminated_phase:
                reward -= 50.0
                stats.eliminated_phase = phase
                
        elif decision_type == "negotiation":
            # Diplomatic rewards (alliance formation, etc.)
            reward += alliance_rewards.get(power, 0.0)
            # For negotiation, show previous center changes
            if stats.supply_centers:
                game_change = stats.supply_centers[-1] - stats.supply_centers[0] if len(stats.supply_centers) > 1 else 0
                center_changes[power] = game_change
            
        rewards.append(reward)
    
    # Log center changes instead of alliance information
    if decision_type == "orders" and center_changes:
        center_summary = ", ".join([f"{power}: {change:+d}" for power, change in center_changes.items() if change != 0])
        if center_summary:
            logger.info(f"Center changes since game start ({phase}): {center_summary}")
        else:
            logger.info(f"No center changes this phase ({phase})")
    
    logger.debug(f"Step rewards for {phase} ({decision_type}): {dict(zip(sorted(ALL_POWERS), rewards))}")
    return rewards


def calculate_final_rewards(
    game: Game,
    agents: Dict[str, DiplomacyAgent],
    alliance_tracker: AllianceTracker
) -> List[float]:
    """
    Calculate final rewards when game is complete.
    
    Args:
        game: Final game state
        agents: Dictionary of power -> agent
        alliance_tracker: Alliance tracking system
        
    Returns:
        List of 7 final rewards (one per power, in alphabetical order)
    """
    rewards = []
    
    # Determine game outcome
    winner = None
    max_centers = 0
    for power in ALL_POWERS:
        centers = len(game.get_power(power).centers)
        if centers >= 18:  # Solo victory
            winner = power
            break
        if centers > max_centers:
            max_centers = centers
    
    for power in sorted(ALL_POWERS):
        reward = 0.0
        game_power = game.get_power(power)
        stats = alliance_tracker.power_stats[power]
        
        final_centers = len(game_power.centers)
        
        if winner == power:
            # Solo victory bonus
            reward += 100.0
            logger.info(f"Solo victory bonus for {power}: +100.0")
        elif final_centers > 0:
            # Survival bonus based on final position
            survival_bonus = 20.0 + (final_centers * 2.0)  # Base survival + centers
            reward += survival_bonus
        else:
            # Elimination penalty (if not already applied)
            if not stats.eliminated_phase:
                reward -= 50.0
        
        # Diplomatic success bonuses
        if stats.alliances_formed > stats.alliances_broken:
            reward += 5.0 * (stats.alliances_formed - stats.alliances_broken)
        
        # Betrayal penalties
        reward -= 3.0 * stats.betrayals_committed
        
        rewards.append(reward)
    
    # Log final center changes instead of just rewards
    center_changes = {}
    for power in sorted(ALL_POWERS):
        stats = alliance_tracker.power_stats[power]
        if stats.supply_centers and len(stats.supply_centers) > 1:
            game_change = stats.supply_centers[-1] - stats.supply_centers[0]
            center_changes[power] = game_change
        else:
            center_changes[power] = 0
    
    center_summary = ", ".join([f"{power}: {change:+d}" for power, change in center_changes.items()])
    logger.info(f"Final center changes: {center_summary}")
    logger.debug(f"Final rewards: {dict(zip(sorted(ALL_POWERS), rewards))}")
    return rewards


def analyze_alliance_patterns(alliance_tracker: AllianceTracker) -> Dict[str, Any]:
    """
    Analyze alliance patterns for training insights.
    
    Args:
        alliance_tracker: Completed game alliance tracker
        
    Returns:
        Dictionary of alliance analysis metrics
    """
    analysis = {
        'total_alliances_formed': len(alliance_tracker.alliances),
        'alliances_broken': sum(1 for a in alliance_tracker.alliances.values() if a.broken),
        'betrayals_detected': sum(1 for a in alliance_tracker.alliances.values() if a.betrayal_detected),
        'power_stats': {}
    }
    
    # Per-power analysis
    for power, stats in alliance_tracker.power_stats.items():
        analysis['power_stats'][power] = {
            'alliances_formed': stats.alliances_formed,
            'alliances_broken': stats.alliances_broken,
            'betrayals_committed': stats.betrayals_committed,
            'betrayals_suffered': stats.betrayals_suffered,
            'final_centers': stats.supply_centers[-1] if stats.supply_centers else 0,
            'eliminated': stats.eliminated_phase is not None
        }
    
    return analysis