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
   - `prompt_constructor.py` - Centralized prompt building (with reduced logging verbosity)
   - `game_history.py` - Phase-by-phase game tracking

3. **GRPO Training System** (`ai_diplomacy/grpo_*.py`)
   - `grpo_trainer.py` - Main GRPO training orchestrator with parallel game support
   - `grpo_env.py` - Multi-turn environment wrapper for RL training
   - `grpo_rewards.py` - Alliance formation rewards and center change tracking
   - Supports 7B+ models with optimized VRAM usage
   - Comprehensive W&B logging with numeric field optimization

4. **Visualization System** (`ai_animation/` directory)
   - TypeScript/Three.js 3D game visualization
   - Real-time diplomatic message display
   - Unit movement animations
   - Victory and standings displays

5. **Analysis Tools**
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

### GRPO Training (Online Reinforcement Learning)
```bash
# Run GRPO training with default settings
python -c "from ai_diplomacy.grpo_trainer import TrainingConfig, DiplomacyGRPOTrainer; trainer = DiplomacyGRPOTrainer(TrainingConfig()); trainer.train()"

# Custom GRPO training configuration
python -c "
from ai_diplomacy.grpo_trainer import TrainingConfig, DiplomacyGRPOTrainer
config = TrainingConfig(
    model_name='Qwen/Qwen2.5-7B-Instruct',
    num_episodes=100,
    batch_size=14,  # 2 parallel games
    max_year=1908,
    use_wandb=True,
    log_level='WARNING'
)
trainer = DiplomacyGRPOTrainer(config)
trainer.train()
"

# Google Colab training (see diplomacy_grpo_training.ipynb)
# - Enhanced W&B logging with step-by-step metrics
# - Optimized for 24GB VRAM with parallel games
# - Automatic center change tracking and alliance analysis
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
- **GRPO Training**: `ai_diplomacy/grpo_trainer.py:TrainingConfig` class for all training parameters
- **W&B Integration**: Enhanced logging with numeric field optimization for better visualization

### Prompt Templates
- **Power-specific**: `ai_diplomacy/prompts/[power]_system_prompt.txt`
- **Task-specific**: `ai_diplomacy/prompts/[task]_instructions.txt`
- **Standardized vs Flavor**: Different personality styles in subdirectories

### Game Data
- **Results**: `results/` directory with timestamped game folders
- **Game Format**: `lmvsgame.json` with phase summaries and agent relationships
- **Visualization Data**: `public/games/` for animation system
- **GRPO Training Data**: `checkpoints/` directory with model checkpoints and training stats
- **W&B Logs**: Comprehensive metrics in Weights & Biases dashboard

## Development Workflow

1. **Game Development**: Work primarily in `ai_diplomacy/` directory for agent logic
2. **GRPO Training**: Use `ai_diplomacy/grpo_*.py` for reinforcement learning experiments
3. **Visualization**: Use `ai_animation/` for 3D display and UI components
4. **Analysis**: Create new analysis scripts in root directory, following existing patterns
5. **Testing**: Run both Python tests and TypeScript e2e tests before major changes
6. **W&B Monitoring**: Use Weights & Biases dashboard for training progress and detailed analytics

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
- **GRPO Training Logs**: Step-by-step rewards, center changes, alliance formation
- **W&B Metrics**: Real-time training progress with numeric field optimization







## GRPO Training System

### Key Features
- **Multi-Turn Environment**: Full Diplomacy games (1901-1910) with negotiations and orders
- **Parallel Training**: Multiple simultaneous games for faster data collection
- **Alliance Rewards**: Sophisticated reward system tracking diplomatic relationships
- **Center Tracking**: Real-time supply center gains/losses with territory-based rewards
- **W&B Integration**: Comprehensive logging with optimized field types for visualization

### Training Configuration
```python
from ai_diplomacy.grpo_trainer import TrainingConfig

# Optimized for 24GB VRAM
config = TrainingConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # 7B model for better performance
    batch_size=14,                          # 2 parallel games (2 * 7 agents)
    max_length=4096,                        # Extended context length
    num_episodes=100,                       # Training episodes
    max_year=1908,                          # Game length
    num_negotiation_rounds=4,               # Diplomatic rounds per phase
    log_level="WARNING",                    # Reduced logging verbosity
    use_wandb=True,                         # W&B integration
    log_step_rewards=True,                  # Detailed step tracking
    log_center_changes=True,                # Territory progression
)
```

### W&B Logging Features
- **Step-level Metrics**: Individual agent rewards, center counts, phase progression
- **Episode Metrics**: Victory distribution, alliance formation, game length
- **Training Progress**: Learning curves, model performance trends
- **Numeric Optimization**: Phase/decision types converted to numbers for better visualization
- **Cross-game Analysis**: Aggregate statistics across parallel games

### Memory Optimizations
- **Gradient Checkpointing**: Reduces memory usage during training
- **Flash Attention 2**: Efficient attention computation when available
- **Mixed Precision**: FP16 for memory savings
- **Parallel Games**: 2-14 games simultaneously depending on VRAM

## Common Pitfalls

1. **Model Availability**: Always verify model names in `clients.py` before assignment
2. **Context Limits**: Use `--max_tokens` flags to prevent context overflow
3. **API Keys**: Ensure all required keys are in `.env` file
4. **Game Resumption**: Use exact phase names when resuming (e.g., `S1902M`, not `Spring 1902`)
5. **Directory Structure**: Game folders must contain `lmvsgame.json` for analysis tools
6. **GRPO Dependencies**: Ensure `willccbb/verifiers` package is installed for training
7. **VRAM Limits**: Monitor GPU memory usage; reduce batch_size if OOM errors occur
8. **W&B Authentication**: Set up W&B API key for logging integration

## Debug and Troubleshooting

- **Game Logs**: Check `general_game.log` in game directories
- **LLM Responses**: Review `llm_responses.csv` for model behavior
- **Visualization Debug**: Enable debug mode in animation system for detailed logging
- **Agent State**: Examine `final_agent_states` in JSON for relationship/goal evolution
- **GRPO Training**: Monitor W&B dashboard for training progress and convergence
- **Memory Issues**: Use `torch.cuda.memory_summary()` to debug VRAM usage
- **Verbose Logging**: Temporarily set `log_level="DEBUG"` for detailed debugging
- **W&B Field Types**: Check numeric vs string field issues in dashboard panels

## Recent Updates (2025)

### GRPO Training Implementation
- Added complete GRPO training system with `willccbb/verifiers` integration
- Implemented parallel game training for efficient data collection
- Enhanced W&B logging with numeric field optimization to avoid media type conflicts
- Added comprehensive center change tracking with gain/loss reporting

### Logging Improvements
- Reduced verbosity of goal and system prompt logging in `prompt_constructor.py`
- Converted W&B string fields to numeric for better visualization (phase→game_year, winner→winner_id)
- Added proper W&B metric definitions to prevent visualization issues
- Implemented clean console output with configurable log levels

### Memory and Performance Optimizations
- Added support for larger models (7B+) with optimized VRAM usage
- Implemented gradient checkpointing and Flash Attention 2 support
- Enhanced batched generation with multiple response sampling
- Added VRAM monitoring and efficiency reporting