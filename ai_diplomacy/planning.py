from dotenv import load_dotenv
import logging
import asyncio
from typing import Dict

from .clients import load_model_client
from .game_history import GameHistory
from .agent import DiplomacyAgent
from .utils import gather_possible_orders

logger = logging.getLogger(__name__)

async def planning_phase(
    game, 
    agents: Dict[str, DiplomacyAgent], 
    game_history: GameHistory, 
    model_error_stats,
    log_file_path: str,
    game_id: str = None,
):
    """
    Lets each power generate a strategic plan using their DiplomacyAgent.
    """
    logger.info(f"Starting planning phase for {game.current_short_phase}...")
    active_powers = [
        p_name for p_name, p_obj in game.powers.items() if not p_obj.is_eliminated()
    ]
    eliminated_powers = [
        p_name for p_name, p_obj in game.powers.items() if p_obj.is_eliminated()
    ]
    
    logger.info(f"Active powers for planning: {active_powers}")
    if eliminated_powers:
        logger.info(f"Eliminated powers (skipped): {eliminated_powers}")
    else:
        logger.info("No eliminated powers yet.")
    
    board_state = game.get_state()

    # Create async tasks for all planning operations
    tasks = []
    power_names_for_tasks = []
    
    for power_name in active_powers:
        if power_name not in agents:
            logger.warning(f"Agent for {power_name} not found in planning phase. Skipping.")
            continue
        agent = agents[power_name]
        client = agent.client
        
        task = client.get_plan(
            game,
            board_state,
            power_name,
            game_history,
            log_file_path,
            agent_goals=agent.goals,
            agent_relationships=agent.relationships,
            agent_private_diary_str=agent.format_private_diary_for_prompt(),
            game_id=game_id,
        )
        tasks.append(task)
        power_names_for_tasks.append(power_name)
        logger.debug(f"Created get_plan task for {power_name}.")

    if tasks:
        logger.info(f"Waiting for {len(tasks)} planning results...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, plan_result in enumerate(results):
            power_name = power_names_for_tasks[i]
            agent = agents[power_name]
            
            if isinstance(plan_result, Exception):
                logger.error(f"Exception during planning for {power_name}: {plan_result}")
                if power_name in model_error_stats:
                    model_error_stats[power_name].setdefault('planning_execution_errors', 0)
                    model_error_stats[power_name]['planning_execution_errors'] += 1
                else:
                     model_error_stats.setdefault(f'{power_name}_planning_execution_errors', 0)
                     model_error_stats[f'{power_name}_planning_execution_errors'] += 1
            elif plan_result and plan_result.startswith("Error:"):
                logger.warning(f"Agent {power_name} reported an error during planning: {plan_result}")
                if power_name in model_error_stats:
                    model_error_stats[power_name].setdefault('planning_generation_errors', 0)
                    model_error_stats[power_name]['planning_generation_errors'] += 1
                else:
                    model_error_stats.setdefault(f'{power_name}_planning_generation_errors', 0)
                    model_error_stats[f'{power_name}_planning_generation_errors'] += 1
            elif plan_result:
                logger.info(f"Received planning result from {power_name}.")
                agent.add_journal_entry(f"Generated plan for {game.current_short_phase}: {plan_result[:100]}...")
                game_history.add_plan(
                    game.current_short_phase, power_name, plan_result
                )
                logger.debug(f"Added plan for {power_name} to history.")
            else:
                logger.warning(f"Agent {power_name} returned an empty plan.")
        
    logger.info("Planning phase processing complete.")
    return game_history