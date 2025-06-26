# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains an AI-powered Diplomacy game implementation with sophisticated LLM agents, visualization, and analysis tools. The project extends the base Diplomacy engine with stateful AI agents that conduct negotiations, form relationships, and make strategic decisions.

## Architecture

### Core Components

1. **Main Game Orchestrator** (`lm_game.py`)
   - Manages agent lifecycle and game phases
   - Coordinates async LLM calls for performance
   - Handles error tracking and recovery
   - Saves game state with phase summaries and agent relationships

2. **AI Agent System** (`ai_diplomacy/` directory)
   - `agent.py` - Stateful `DiplomacyAgent` with goals, relationships, and memory
   - `clients.py` - LLM abstraction layer supporting OpenAI, Claude, Gemini, DeepSeek, OpenRouter
   - `possible_order_context.py` - Strategic analysis with BFS pathfinding
   - `prompt_constructor.py` - Centralized prompt building
   - `game_history.py` - Phase-by-phase game tracking

3. **Visualization System** (`ai_animation/` directory)
   - TypeScript/Three.js 3D game visualization
   - Real-time diplomatic message display
   - Unit movement animations
   - Victory and standings displays

4. **Analysis Tools**
   - `analyze_game_moments.py` - Strategic moment analysis including betrayal detection
   - `csv_to_rl_json.py` + `analyze_rl_json.py` - LLM performance analysis pipeline
   - `experiment_runner.py` - Batch experiment orchestration

### Agent Memory Architecture

Each AI agent maintains:
- **Dynamic Goals**: Strategic objectives that evolve based on game events
- **Relationship Tracking**: Enemy/Unfriendly/Neutral/Friendly/Ally relationships with other powers
- **Private Diary**: Structured, phase-prefixed entries for LLM context
  - Negotiation summaries with relationship updates
  - Order reasoning and strategic justifications
  - Phase result analysis with betrayal detection
- **Yearly Consolidation**: Automatic summarization to prevent context overflow

## Common Development Commands

### Python Environment Setup
```bash
# Install dependencies and create virtual environment using uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # Unix/macOS
.venv\Scripts\activate     # Windows
```

### Running AI Games
```bash
# Basic game with negotiations
python lm_game.py --max_year 1910 --num_negotiation_rounds 3

# With strategic planning phase
python lm_game.py --max_year 1910 --planning_phase --num_negotiation_rounds 2

# Resume from specific phase
python lm_game.py --run_dir results/game_run_001 --resume_from_phase S1902M

# Custom model assignment (AU,EN,FR,GE,IT,RU,TR order)
python lm_game.py --models "claude-3-5-sonnet-20241022,gpt-4o,claude-3-5-sonnet-20241022,gpt-4o,claude-3-5-sonnet-20241022,gpt-4o,claude-3-5-sonnet-20241022"
```

### Batch Experiments
```bash
# Run 10 parallel games
python experiment_runner.py --experiment_dir "results/exp001" --iterations 10 --parallel 10 --max_year 1905

# Critical-state analysis
python experiment_runner.py --experiment_dir "results/exp002" --iterations 10 --resume_from_phase W1901A --end_at_phase S1902M --critical_state_base_run "results/test1"
```

### Analysis Tools
```bash
# Strategic moment analysis with betrayal detection
python analyze_game_moments.py results/game_folder

# LLM performance analysis pipeline
python csv_to_rl_json.py --scan_dir results/
python analyze_rl_json.py results/json/

# Game results statistics
python analyze_game_results.py
```

### Visualization System
```bash
# Start 3D visualization server
cd ai_animation
npm install
npm run dev  # Development server at http://localhost:5173

# Testing
npm run test        # Unit tests with Vitest
npm run test:e2e    # End-to-end tests with Playwright
npm run lint        # TypeScript linting
```

### Base Game Engine
```bash
# Run original diplomacy engine tests
cd diplomacy
python -m pytest tests/

# Web interface (React)
cd diplomacy/web
npm install
npm start  # React server at http://localhost:3000

# In separate terminal - diplomacy server
python -m diplomacy.server.run  # Server at http://localhost:8432
```

## Key File Locations

### Configuration
- **Environment Variables**: `.env` file with API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- **Python Dependencies**: `pyproject.toml` (uses uv for dependency management)
- **Model Assignment**: `ai_diplomacy/utils.py:assign_models_to_powers()`

### Prompt Templates
- **Power-specific**: `ai_diplomacy/prompts/[power]_system_prompt.txt`
- **Task-specific**: `ai_diplomacy/prompts/[task]_instructions.txt`
- **Standardized vs Flavor**: Different personality styles in subdirectories

### Game Data
- **Results**: `results/` directory with timestamped game folders
- **Game Format**: `lmvsgame.json` with phase summaries and agent relationships
- **Visualization Data**: `public/games/` for animation system

## Development Workflow

1. **Game Development**: Work primarily in `ai_diplomacy/` directory for agent logic
2. **Visualization**: Use `ai_animation/` for 3D display and UI components
3. **Analysis**: Create new analysis scripts in root directory, following existing patterns
4. **Testing**: Run both Python tests and TypeScript e2e tests before major changes

## Agent Behavior Patterns

- **Initialization**: Sets starting personality and objectives via LLM
- **Negotiation Rounds**: Generate contextual messages based on relationships
- **Strategic Planning**: Create high-level directives (optional phase)
- **Order Generation**: Select moves with full strategic context
- **State Updates**: Adjust goals and relationships based on outcomes
- **Diary Consolidation**: Automatic yearly summarization via Gemini Flash

## Supported LLM Models

### OpenAI
- `gpt-4o`, `gpt-4.1`, `o3`, `o4-mini`

### Anthropic
- `claude-3-5-sonnet-20241022`, `claude-opus-4-20250514`, `claude-sonnet-4-20250514`

### Google
- `gemini-2.0-flash`, `gemini-2.5-pro-preview`, `gemini-2.5-flash`

### OpenRouter
- Various models including Llama, Qwen, DeepSeek (prefix with `openrouter-`)

## Output Formats

Games generate comprehensive analysis data:
- **Phase Summaries**: Categorized move results for each phase
- **Agent Relationships**: Diplomatic standings throughout game
- **LLM Interactions**: Complete log of all model calls and responses
- **Strategic Moments**: High-interest events with betrayal detection
- **Performance Metrics**: Invalid move rates and model characteristics

## Common Pitfalls

1. **Model Availability**: Always verify model names in `clients.py` before assignment
2. **Context Limits**: Use `--max_tokens` flags to prevent context overflow
3. **API Keys**: Ensure all required keys are in `.env` file
4. **Game Resumption**: Use exact phase names when resuming (e.g., `S1902M`, not `Spring 1902`)
5. **Directory Structure**: Game folders must contain `lmvsgame.json` for analysis tools

## Debug and Troubleshooting

- **Game Logs**: Check `general_game.log` in game directories
- **LLM Responses**: Review `llm_responses.csv` for model behavior
- **Visualization Debug**: Enable debug mode in animation system for detailed logging
- **Agent State**: Examine `final_agent_states` in JSON for relationship/goal evolution