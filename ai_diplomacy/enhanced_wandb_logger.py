# ai_diplomacy/enhanced_wandb_logger.py
"""
Enhanced W&B logging for comprehensive Diplomacy GRPO training metrics.

This module provides:
- System metrics monitoring
- Supply center history tracking and visualization
- LLM generation logging to JSON files
- Training metrics and game analytics
- Performance monitoring
"""

import logging
import json
import time
import psutil
import torch
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedWandBLogger:
    """
    Comprehensive W&B logger for Diplomacy GRPO training with enhanced metrics.
    """
    
    def __init__(self, 
                 project_name: str = "diplomacy-grpo-enhanced",
                 entity: Optional[str] = None,
                 enabled: bool = True):
        self.enabled = enabled and WANDB_AVAILABLE
        self.project_name = project_name
        self.entity = entity
        
        # Tracking data
        self.supply_center_history = []  # [{phase, power, centers}, ...]
        self.llm_generations = []  # Store all LLM outputs
        self.system_metrics = []
        self.game_metrics = []
        
        # Temp directory for JSON files
        self.temp_dir = Path(tempfile.mkdtemp(prefix="diplomacy_logs_"))
        
        if not WANDB_AVAILABLE and enabled:
            logger.warning("W&B not available - enhanced logging disabled. Install with: pip install wandb")
            self.enabled = False
        
        if self.enabled:
            logger.info(f"Enhanced W&B Logger initialized - logs will be saved to {self.temp_dir}")
    
    def initialize_wandb_session(self, config: Dict[str, Any]) -> None:
        """Initialize W&B session with comprehensive configuration."""
        if not self.enabled:
            return
        
        # Enhanced config with system info
        enhanced_config = config.copy()
        enhanced_config.update({
            'system_info': self._get_system_info(),
            'logging_features': {
                'system_metrics': True,
                'supply_center_tracking': True,
                'llm_generation_logging': True,
                'game_analytics': True
            },
            'timestamp': datetime.now().isoformat()
        })
        
        try:
            wandb.init(
                project=self.project_name,
                entity=self.entity,
                config=enhanced_config,
                name=f"diplomacy_grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=["diplomacy", "grpo", "llm", "enhanced-logging"]
            )
            
            # Define custom metrics
            wandb.define_metric("episode")
            wandb.define_metric("step")
            wandb.define_metric("system/*", step_metric="step")
            wandb.define_metric("game/*", step_metric="step")
            wandb.define_metric("training/*", step_metric="episode")
            wandb.define_metric("supply_centers/*", step_metric="step")
            
            logger.info("W&B session initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.enabled = False
    
    def log_system_metrics(self, step: int) -> None:
        """Log comprehensive system metrics."""
        if not self.enabled:
            return
        
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU metrics if available
            gpu_metrics = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_metrics[f'gpu_{i}_memory_used'] = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    gpu_metrics[f'gpu_{i}_memory_total'] = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                    gpu_metrics[f'gpu_{i}_utilization'] = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            metrics = {
                'system/cpu_percent': cpu_percent,
                'system/memory_used_gb': memory.used / 1024**3,
                'system/memory_total_gb': memory.total / 1024**3,
                'system/memory_percent': memory.percent,
                'system/disk_used_gb': disk.used / 1024**3,
                'system/disk_total_gb': disk.total / 1024**3,
                'system/disk_percent': (disk.used / disk.total) * 100,
                'step': step,
                **gpu_metrics
            }
            
            # Store for later analysis
            metrics['timestamp'] = time.time()
            self.system_metrics.append(metrics.copy())
            
            # Log to W&B
            wandb.log(metrics)
            
        except Exception as e:
            logger.warning(f"Failed to log system metrics: {e}")
    
    def log_supply_center_change(self, phase: str, power: str, new_centers: List[str], old_centers: List[str]) -> None:
        """Log supply center changes and update history."""
        if not self.enabled:
            return
        
        try:
            change = len(new_centers) - len(old_centers)
            
            # Add to history
            center_data = {
                'phase': phase,
                'power': power,
                'centers': len(new_centers),
                'center_list': new_centers.copy(),
                'change': change,
                'timestamp': time.time()
            }
            self.supply_center_history.append(center_data)
            
            # Log individual power metrics
            wandb.log({
                f'supply_centers/{power}_count': len(new_centers),
                f'supply_centers/{power}_change': change,
                f'game/phase': phase,
                'step': len(self.supply_center_history)
            })
            
            # Create and log supply center history graph every few phases
            if len(self.supply_center_history) % 7 == 0:  # Every turn (7 powers)
                self._create_supply_center_graph()
            
        except Exception as e:
            logger.warning(f"Failed to log supply center change: {e}")
    
    def log_llm_generation(self, power: str, phase: str, prompt: str, response: str, 
                          parsed_orders: Optional[List[str]] = None, 
                          success: bool = True, episode: Optional[int] = None,
                          step: Optional[int] = None) -> None:
        """Log LLM generation to JSON file and W&B."""
        if not self.enabled:
            return
        
        try:
            generation_data = {
                'timestamp': time.time(),
                'power': power,
                'phase': phase,
                'prompt': prompt,  # Full prompt saved
                'response': response,  # Full response saved
                'parsed_orders': parsed_orders,
                'success': success,
                'episode': episode,
                'step': step,
                'prompt_length': len(prompt),
                'response_length': len(response),
                'order_count': len(parsed_orders) if parsed_orders else 0
            }
            
            # Store in memory
            self.llm_generations.append(generation_data)
            
            # Save FULL JSON file with complete content (NO TRUNCATION)
            generation_id = len(self.llm_generations)
            json_file = self.temp_dir / f"llm_generation_FULL_{generation_id:04d}_{power}_{phase}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(generation_data, f, indent=2, ensure_ascii=False)
            
            # Also save a preview version for quick viewing (truncated to 1000 chars)
            preview_data = generation_data.copy()
            preview_data['prompt'] = prompt[:1000] + "...[TRUNCATED - see FULL file]" if len(prompt) > 1000 else prompt
            preview_data['response'] = response[:1000] + "...[TRUNCATED - see FULL file]" if len(response) > 1000 else response
            preview_data['_note'] = f"PREVIEW ONLY! For complete content, see: llm_generation_FULL_{generation_id:04d}_{power}_{phase}.json"
            
            preview_file = self.temp_dir / f"llm_generation_PREVIEW_{generation_id:04d}_{power}_{phase}.json"
            with open(preview_file, 'w', encoding='utf-8') as f:
                json.dump(preview_data, f, indent=2, ensure_ascii=False)
            
            # Upload both files to W&B with clear naming
            wandb.save(str(json_file), base_path=self.temp_dir, policy="now")
            wandb.save(str(preview_file), base_path=self.temp_dir, policy="now")
            
            # Also log the full content as a W&B artifact to ensure it's accessible
            artifact = wandb.Artifact(
                name=f"llm_generation_{generation_id:04d}_{power}_{phase}",
                type="llm_generation",
                description=f"Full LLM generation for {power} in {phase}"
            )
            artifact.add_file(str(json_file))
            wandb.log_artifact(artifact)
            
            # Log file information
            full_size = json_file.stat().st_size
            preview_size = preview_file.stat().st_size
            logger.info(f"✅ LLM generation saved for {power}:")
            logger.info(f"   📄 FULL file: {json_file.name} ({full_size:,} bytes)")
            logger.info(f"   👁️  PREVIEW file: {preview_file.name} ({preview_size:,} bytes)")
            logger.info(f"   🔗 W&B Artifact: llm_generation_{generation_id:04d}_{power}_{phase}")
            
            # Log summary metrics
            wandb.log({
                f'llm_generations/{power}_success_rate': self._calculate_success_rate(power),
                f'llm_generations/{power}_avg_response_length': self._calculate_avg_response_length(power),
                f'llm_generations/{power}_avg_order_count': self._calculate_avg_order_count(power),
                'llm_generations/total_generations': len(self.llm_generations),
                'step': step or len(self.llm_generations)
            })
            
            logger.debug(f"Logged LLM generation for {power} in {phase}")
            
        except Exception as e:
            logger.warning(f"Failed to log LLM generation: {e}")
    
    def log_game_metrics(self, phase: str, game_state: Dict[str, Any], step: int) -> None:
        """Log comprehensive game state metrics."""
        if not self.enabled:
            return
        
        try:
            # Count active powers
            active_powers = sum(1 for p, units in game_state.get('units', {}).items() if units)
            eliminated_powers = 7 - active_powers
            
            # Calculate total units and centers
            total_units = sum(len(units) for units in game_state.get('units', {}).values())
            total_centers = sum(len(centers) for centers in game_state.get('centers', {}).values())
            
            # Phase information
            year = int(phase[1:5]) if len(phase) >= 5 else 1901
            season = phase[0] if phase else 'S'
            
            metrics = {
                'game/phase': phase,
                'game/year': year,
                'game/season': season,
                'game/active_powers': active_powers,
                'game/eliminated_powers': eliminated_powers,
                'game/total_units': total_units,
                'game/total_centers': total_centers,
                'game/avg_centers_per_power': total_centers / max(active_powers, 1),
                'step': step
            }
            
            # Store for analysis
            metrics['timestamp'] = time.time()
            self.game_metrics.append(metrics.copy())
            
            wandb.log(metrics)
            
        except Exception as e:
            logger.warning(f"Failed to log game metrics: {e}")
    
    def log_training_metrics(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Log training-specific metrics."""
        if not self.enabled:
            return
        
        try:
            # Prefix training metrics
            training_metrics = {f'training/{k}': v for k, v in metrics.items()}
            training_metrics['episode'] = episode
            
            wandb.log(training_metrics)
            
        except Exception as e:
            logger.warning(f"Failed to log training metrics: {e}")
    
    def _create_supply_center_graph(self) -> None:
        """Create and log supply center history graph."""
        try:
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(self.supply_center_history)
            if df.empty:
                return
            
            plt.figure(figsize=(12, 8))
            
            # Plot each power's supply center count over time
            powers = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
            colors = ['red', 'blue', 'lightblue', 'black', 'green', 'purple', 'yellow']
            
            for power, color in zip(powers, colors):
                power_data = df[df['power'] == power].sort_values('timestamp')
                if not power_data.empty:
                    plt.plot(range(len(power_data)), power_data['centers'], 
                            label=power, color=color, linewidth=2, marker='o')
            
            plt.title('Supply Center History', fontsize=16, fontweight='bold')
            plt.xlabel('Game Phases')
            plt.ylabel('Supply Centers')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 18)
            
            # Save and log to W&B
            temp_plot = self.temp_dir / f"supply_center_history_{len(self.supply_center_history)}.png"
            plt.savefig(temp_plot, dpi=300, bbox_inches='tight')
            wandb.log({"supply_center_history": wandb.Image(str(temp_plot))})
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to create supply center graph: {e}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            info = {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'platform': os.name,
                'python_version': f"{psutil.version_info}",
            }
            
            if torch.cuda.is_available():
                info['cuda_available'] = True
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_version'] = torch.version.cuda
                for i in range(torch.cuda.device_count()):
                    info[f'gpu_{i}_name'] = torch.cuda.get_device_name(i)
                    info[f'gpu_{i}_memory_gb'] = torch.cuda.get_device_properties(i).total_memory / 1024**3
            else:
                info['cuda_available'] = False
                
            return info
        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")
            return {}
    
    def _calculate_success_rate(self, power: str) -> float:
        """Calculate LLM generation success rate for a power."""
        power_generations = [g for g in self.llm_generations if g['power'] == power]
        if not power_generations:
            return 0.0
        return sum(g['success'] for g in power_generations) / len(power_generations)
    
    def _calculate_avg_response_length(self, power: str) -> float:
        """Calculate average response length for a power."""
        power_generations = [g for g in self.llm_generations if g['power'] == power]
        if not power_generations:
            return 0.0
        return sum(g['response_length'] for g in power_generations) / len(power_generations)
    
    def _calculate_avg_order_count(self, power: str) -> float:
        """Calculate average order count for a power."""
        power_generations = [g for g in self.llm_generations if g['power'] == power]
        if not power_generations:
            return 0.0
        return sum(g['order_count'] for g in power_generations) / len(power_generations)
    
    def save_final_logs(self) -> None:
        """Save final comprehensive logs to W&B."""
        if not self.enabled:
            return
        
        try:
            # Save comprehensive JSON logs
            final_log = {
                'supply_center_history': self.supply_center_history,
                'llm_generations_summary': self._create_llm_summary(),
                'system_metrics_summary': self._create_system_summary(),
                'game_metrics_summary': self._create_game_summary(),
                'total_generations': len(self.llm_generations),
                'total_phases': len(set(g['phase'] for g in self.llm_generations)),
                'timestamp': time.time()
            }
            
            final_log_file = self.temp_dir / "comprehensive_game_log.json"
            with open(final_log_file, 'w') as f:
                json.dump(final_log, f, indent=2)
            
            wandb.save(str(final_log_file))
            
            # Create final supply center graph
            if self.supply_center_history:
                self._create_supply_center_graph()
            
            logger.info(f"Final logs saved to W&B. Total generations: {len(self.llm_generations)}")
            
        except Exception as e:
            logger.error(f"Failed to save final logs: {e}")
    
    def _create_llm_summary(self) -> Dict[str, Any]:
        """Create summary of LLM generation statistics."""
        if not self.llm_generations:
            return {}
        
        powers = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
        summary = {}
        
        for power in powers:
            power_gens = [g for g in self.llm_generations if g['power'] == power]
            if power_gens:
                summary[power] = {
                    'total_generations': len(power_gens),
                    'success_rate': sum(g['success'] for g in power_gens) / len(power_gens),
                    'avg_response_length': sum(g['response_length'] for g in power_gens) / len(power_gens),
                    'avg_order_count': sum(g['order_count'] for g in power_gens) / len(power_gens)
                }
        
        return summary
    
    def _create_system_summary(self) -> Dict[str, Any]:
        """Create summary of system metrics."""
        if not self.system_metrics:
            return {}
        
        return {
            'avg_cpu_percent': sum(m['system/cpu_percent'] for m in self.system_metrics) / len(self.system_metrics),
            'avg_memory_percent': sum(m['system/memory_percent'] for m in self.system_metrics) / len(self.system_metrics),
            'peak_memory_gb': max(m['system/memory_used_gb'] for m in self.system_metrics),
        }
    
    def _create_game_summary(self) -> Dict[str, Any]:
        """Create summary of game metrics."""
        if not self.game_metrics:
            return {}
        
        return {
            'total_phases': len(self.game_metrics),
            'avg_active_powers': sum(m['game/active_powers'] for m in self.game_metrics) / len(self.game_metrics),
            'final_year': max(m['game/year'] for m in self.game_metrics),
        }
    
    def finalize_training_logs(self) -> None:
        """Create final comprehensive artifacts and summaries."""
        if not self.enabled:
            return
        
        try:
            # Create a comprehensive LLM generations archive
            all_generations_file = self.temp_dir / "ALL_LLM_GENERATIONS_COMPLETE.json"
            with open(all_generations_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_generations': len(self.llm_generations),
                    'generation_summary': self._create_llm_summary(),
                    'all_generations': self.llm_generations  # Full content, no truncation
                }, f, indent=2, ensure_ascii=False)
            
            # Create final artifact
            final_artifact = wandb.Artifact(
                name="complete_training_logs",
                type="training_archive",
                description="Complete archive of all LLM generations and training data"
            )
            final_artifact.add_file(str(all_generations_file))
            
            # Add supply center history
            if self.supply_center_history:
                sc_file = self.temp_dir / "supply_center_complete_history.json"
                with open(sc_file, 'w', encoding='utf-8') as f:
                    json.dump(self.supply_center_history, f, indent=2)
                final_artifact.add_file(str(sc_file))
            
            wandb.log_artifact(final_artifact)
            
            # Log final summary
            total_size = sum(f.stat().st_size for f in self.temp_dir.iterdir() if f.is_file())
            logger.info(f"🎯 Training logs finalized:")
            logger.info(f"   📊 Total LLM generations: {len(self.llm_generations)}")
            logger.info(f"   💾 Total data size: {total_size:,} bytes")
            logger.info(f"   🔗 Final artifact: complete_training_logs")
            logger.info(f"   📂 All files saved to W&B with NO truncation")
            
        except Exception as e:
            logger.warning(f"Failed to finalize training logs: {e}")
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")


# Global logger instance
_enhanced_logger: Optional[EnhancedWandBLogger] = None


def get_enhanced_logger() -> EnhancedWandBLogger:
    """Get the global enhanced logger instance."""
    global _enhanced_logger
    if _enhanced_logger is None:
        _enhanced_logger = EnhancedWandBLogger()
    return _enhanced_logger


def initialize_enhanced_logging(project_name: str = "diplomacy-grpo-enhanced",
                               entity: Optional[str] = None,
                               enabled: bool = True) -> EnhancedWandBLogger:
    """Initialize the global enhanced logger."""
    global _enhanced_logger
    _enhanced_logger = EnhancedWandBLogger(project_name, entity, enabled)
    return _enhanced_logger
