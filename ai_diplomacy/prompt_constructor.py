"""
Module for constructing prompts for LLM interactions in the Diplomacy game.
"""
import logging
from typing import Dict, List, Optional, Any # Added Any for game type placeholder

from .utils import load_prompt
from .possible_order_context import generate_rich_order_context
from .game_history import GameHistory # Assuming GameHistory is correctly importable

# placeholder for diplomacy.Game to avoid circular or direct dependency if not needed for typehinting only
# from diplomacy import Game # Uncomment if 'Game' type hint is crucial and available

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG) # Or inherit from parent logger

def build_context_prompt(
    game: Any, # diplomacy.Game object
    board_state: dict,
    power_name: str,
    possible_orders: Dict[str, List[str]],
    game_history: GameHistory,
    agent_goals: Optional[List[str]] = None,
    agent_relationships: Optional[Dict[str, str]] = None,
    agent_private_diary: Optional[str] = None,
    prompts_dir: Optional[str] = None,
) -> str:
    """Builds the detailed context part of the prompt.

    Args:
        game: The game object.
        board_state: Current state of the board.
        power_name: The name of the power for whom the context is being built.
        possible_orders: Dictionary of possible orders.
        game_history: History of the game (messages, etc.).
        agent_goals: Optional list of agent's goals.
        agent_relationships: Optional dictionary of agent's relationships with other powers.
        agent_private_diary: Optional string of agent's private diary.
        prompts_dir: Optional path to the prompts directory.

    Returns:
        A string containing the formatted context.
    """
    context_template = load_prompt("context_prompt.txt", prompts_dir=prompts_dir)

    # Get our units and centers (not directly used in template, but good for context understanding)
    # units_info = board_state["units"].get(power_name, [])
    # centers_info = board_state["centers"].get(power_name, [])

    # Get the current phase
    year_phase = board_state["phase"]  # e.g. 'S1901M'

    possible_orders_context_str = generate_rich_order_context(game, power_name, possible_orders)

    messages_this_round_text = game_history.get_messages_this_round(
        power_name=power_name,
        current_phase_name=year_phase
    )
    if not messages_this_round_text.strip():
        messages_this_round_text = "\n(No messages this round)\n"

    # Separate active and eliminated powers for clarity
    active_powers = [p for p in game.powers.keys() if not game.powers[p].is_eliminated()]
    eliminated_powers = [p for p in game.powers.keys() if game.powers[p].is_eliminated()]
    
    # Build units representation with power status
    units_lines = []
    for p, u in board_state["units"].items():
        if game.powers[p].is_eliminated():
            units_lines.append(f"  {p}: {u} [ELIMINATED]")
        else:
            units_lines.append(f"  {p}: {u}")
    units_repr = "\n".join(units_lines)
    
    # Build centers representation with power status  
    centers_lines = []
    for p, c in board_state["centers"].items():
        if game.powers[p].is_eliminated():
            centers_lines.append(f"  {p}: {c} [ELIMINATED]")
        else:
            centers_lines.append(f"  {p}: {c}")
    centers_repr = "\n".join(centers_lines)

    context = context_template.format(
        power_name=power_name,
        current_phase=year_phase,
        all_unit_locations=units_repr,
        all_supply_centers=centers_repr,
        messages_this_round=messages_this_round_text,
        possible_orders=possible_orders_context_str,
        agent_goals="\n".join(f"- {g}" for g in agent_goals) if agent_goals else "None specified",
        agent_relationships="\n".join(f"- {p}: {s}" for p, s in agent_relationships.items()) if agent_relationships else "None specified",
        agent_private_diary=agent_private_diary if agent_private_diary else "(No diary entries yet)",
    )

    # Debug logging for context building when debug mode is enabled
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"📊 CONTEXT BUILDING DEBUG FOR {power_name}")
        logger.debug(f"Phase: {year_phase}")
        logger.debug(f"Active powers: {active_powers}")
        logger.debug(f"Eliminated powers: {eliminated_powers}")
        logger.debug(f"Units representation length: {len(units_repr)} chars")
        logger.debug(f"Centers representation length: {len(centers_repr)} chars")
        logger.debug(f"Messages this round length: {len(messages_this_round_text)} chars")
        logger.debug(f"Possible orders context length: {len(possible_orders_context_str)} chars")
        logger.debug(f"Built context length: {len(context)} chars")

    return context

def construct_order_generation_prompt(
    system_prompt: str,
    game: Any, # diplomacy.Game object
    board_state: dict,
    power_name: str,
    possible_orders: Dict[str, List[str]],
    game_history: GameHistory,
    agent_goals: Optional[List[str]] = None,
    agent_relationships: Optional[Dict[str, str]] = None,
    agent_private_diary_str: Optional[str] = None,
    prompts_dir: Optional[str] = None,
) -> str:
    """Constructs the final prompt for order generation.

    Args:
        system_prompt: The base system prompt for the LLM.
        game: The game object.
        board_state: Current state of the board.
        power_name: The name of the power for whom the prompt is being built.
        possible_orders: Dictionary of possible orders.
        game_history: History of the game (messages, etc.).
        agent_goals: Optional list of agent's goals.
        agent_relationships: Optional dictionary of agent's relationships with other powers.
        agent_private_diary_str: Optional string of agent's private diary.
        prompts_dir: Optional path to the prompts directory.

    Returns:
        A string containing the complete prompt for the LLM.
    """
    # Load prompts
    _ = load_prompt("few_shot_example.txt", prompts_dir=prompts_dir) # Loaded but not used, as per original logic
    instructions = load_prompt("order_instructions.txt", prompts_dir=prompts_dir)

    # Build the context prompt
    context = build_context_prompt(
        game,
        board_state,
        power_name,
        possible_orders,
        game_history,
        agent_goals=agent_goals,
        agent_relationships=agent_relationships,
        agent_private_diary=agent_private_diary_str,
        prompts_dir=prompts_dir,
    )

    # Debug logging for prompt components when debug mode is enabled
    if logger.isEnabledFor(logging.DEBUG):
        log_prompt_components(
            power_name=power_name,
            system_prompt=system_prompt,
            context=context,
            instructions=instructions,
            phase=board_state.get("phase")
        )

    final_prompt = system_prompt + "\n\n" + context + "\n\n" + instructions
    
    # Debug logging for full prompt when debug mode is enabled
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"=== FULL PROMPT FOR {power_name} ===")
        logger.debug(f"Phase: {board_state.get('phase', 'Unknown')}")
        logger.debug(f"Prompt length: {len(final_prompt)} characters")
        logger.debug("--- PROMPT START ---")
        logger.debug(final_prompt)
        logger.debug("--- PROMPT END ---")
        logger.debug("=== END FULL PROMPT ===")
    
    return final_prompt

def log_llm_generation(
    power_name: str,
    prompt: str,
    raw_response: str,
    parsed_orders: Optional[List[str]] = None,
    parsing_error: Optional[str] = None,
    phase: Optional[str] = None,
    generation_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log the complete LLM generation process for debugging order parsing issues.
    
    This function provides comprehensive logging of:
    - The full prompt sent to the LLM
    - The raw response from the LLM
    - Parsed orders (if successful)
    - Any parsing errors
    - Generation metadata (temperature, tokens, etc.)
    
    Args:
        power_name: The power making the decision
        prompt: The full prompt sent to the LLM
        raw_response: The raw response from the LLM
        parsed_orders: Successfully parsed orders (if any)
        parsing_error: Any error that occurred during parsing
        phase: Current game phase
        generation_metadata: Additional metadata about the generation
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    
    logger.debug("=" * 80)
    logger.debug(f"🤖 LLM GENERATION DEBUG LOG FOR {power_name}")
    logger.debug("=" * 80)
    
    if phase:
        logger.debug(f"📅 Phase: {phase}")
    
    if generation_metadata:
        logger.debug("🔧 Generation Metadata:")
        for key, value in generation_metadata.items():
            logger.debug(f"   {key}: {value}")
    
    logger.debug("\n📝 FULL PROMPT SENT TO LLM:")
    logger.debug("-" * 40)
    logger.debug(f"Length: {len(prompt)} characters")
    logger.debug("Content:")
    logger.debug(prompt)
    logger.debug("-" * 40)
    
    logger.debug("\n🗣️ RAW LLM RESPONSE:")
    logger.debug("-" * 40)
    logger.debug(f"Length: {len(raw_response)} characters")
    logger.debug("Content:")
    logger.debug(repr(raw_response))  # Use repr to show escape characters
    logger.debug("\nFormatted content:")
    logger.debug(raw_response)
    logger.debug("-" * 40)
    
    if parsed_orders is not None:
        logger.debug("\n✅ SUCCESSFULLY PARSED ORDERS:")
        logger.debug("-" * 40)
        for i, order in enumerate(parsed_orders, 1):
            logger.debug(f"  {i}. {order}")
        logger.debug(f"Total orders: {len(parsed_orders)}")
        logger.debug("-" * 40)
    
    if parsing_error:
        logger.debug("\n❌ PARSING ERROR:")
        logger.debug("-" * 40)
        logger.debug(f"Error: {parsing_error}")
        logger.debug("-" * 40)
        
        # Try to highlight potential issues in the response
        logger.debug("\n🔍 RESPONSE ANALYSIS:")
        logger.debug("-" * 40)
        lines = raw_response.split('\n')
        logger.debug(f"Response has {len(lines)} lines")
        
        # Look for common order patterns
        potential_orders = []
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            if any(keyword in line_stripped.upper() for keyword in ['ARMY', 'FLEET', 'A ', 'F ', 'MOVE', 'HOLD', 'SUPPORT', 'CONVOY']):
                potential_orders.append(f"Line {line_num}: {line_stripped}")
        
        if potential_orders:
            logger.debug("Potential order lines found:")
            for potential_order in potential_orders:
                logger.debug(f"  {potential_order}")
        else:
            logger.debug("No obvious order patterns found in response")
        
        # Check for common formatting issues
        if '{' in raw_response or '}' in raw_response:
            logger.debug("⚠️ JSON formatting detected - may need JSON parsing")
        if '```' in raw_response:
            logger.debug("⚠️ Code block formatting detected - may need markdown parsing")
        if raw_response.count('\n') < 3:
            logger.debug("⚠️ Very few line breaks - response may be on single line")
        
        logger.debug("-" * 40)
    
    logger.debug("\n" + "=" * 80)
    logger.debug(f"END LLM GENERATION DEBUG LOG FOR {power_name}")
    logger.debug("=" * 80)


def log_prompt_components(
    power_name: str,
    system_prompt: str,
    context: str,
    instructions: str,
    phase: Optional[str] = None
) -> None:
    """
    Log individual prompt components for debugging prompt construction.
    
    Args:
        power_name: The power the prompt is for
        system_prompt: System prompt component
        context: Context component  
        instructions: Instructions component
        phase: Current game phase
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    
    logger.debug("=" * 60)
    logger.debug(f"📋 PROMPT COMPONENTS FOR {power_name}")
    if phase:
        logger.debug(f"📅 Phase: {phase}")
    logger.debug("=" * 60)
    
    logger.debug("\n🎯 SYSTEM PROMPT:")
    logger.debug(f"Length: {len(system_prompt)} characters")
    logger.debug(system_prompt)
    
    logger.debug("\n🌍 CONTEXT:")
    logger.debug(f"Length: {len(context)} characters")
    logger.debug(context)
    
    logger.debug("\n📜 INSTRUCTIONS:")
    logger.debug(f"Length: {len(instructions)} characters")
    logger.debug(instructions)
    
    logger.debug("\n" + "=" * 60)
    logger.debug(f"END PROMPT COMPONENTS FOR {power_name}")
    logger.debug("=" * 60)

def enable_debug_logging():
    """
    Convenience function to enable debug logging for this module.
    Call this to see detailed LLM generation logs.
    
    Example:
        from ai_diplomacy.prompt_constructor import enable_debug_logging
        enable_debug_logging()
        # Now all LLM generations will be logged in detail
    """
    logger.setLevel(logging.DEBUG)
    
    # Also set up a console handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.debug("🐛 Debug logging enabled for prompt_constructor module")


def disable_debug_logging():
    """
    Convenience function to disable debug logging for this module.
    """
    logger.setLevel(logging.INFO)
    logger.info("Debug logging disabled for prompt_constructor module")


# Example usage documentation
"""
To enable comprehensive LLM generation debugging:

1. Enable debug logging:
   ```python
   from ai_diplomacy.prompt_constructor import enable_debug_logging
   enable_debug_logging()
   ```

2. Use log_llm_generation() in your agent code after getting LLM response:
   ```python
   # In your agent's order generation code:
   from ai_diplomacy.prompt_constructor import log_llm_generation
   
   # After generating response from LLM:
   raw_response = llm.generate(prompt)
   
   # Try to parse orders
   try:
       parsed_orders = parse_orders(raw_response)
       parsing_error = None
   except Exception as e:
       parsed_orders = None
       parsing_error = str(e)
   
   # Log everything for debugging
   log_llm_generation(
       power_name="FRANCE",
       prompt=prompt,
       raw_response=raw_response,
       parsed_orders=parsed_orders,
       parsing_error=parsing_error,
       phase="S1901M",
       generation_metadata={
           "temperature": 0.7,
           "max_tokens": 500,
           "model": "gpt-4"
       }
   )
   ```

This will show you:
- The exact prompt sent to the LLM
- The raw response from the LLM (including escape characters)
- Successfully parsed orders (if any)
- Detailed parsing error information
- Analysis of potential issues in the response format
"""