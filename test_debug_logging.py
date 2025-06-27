#!/usr/bin/env python3
"""
Test script to demonstrate the new LLM debug logging features.

This script shows how to enable debug logging to see the complete LLM generation
process for debugging order parsing issues.
"""

import logging
import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_debug_logging():
    """Demonstrate how to enable and use debug logging for LLM generations."""
    
    print("🐛 LLM Debug Logging Demo")
    print("=" * 50)
    
    # Method 1: Enable debug logging using the convenience function
    print("\n1. Enabling debug logging using convenience function:")
    try:
        from ai_diplomacy.prompt_constructor import enable_debug_logging, log_llm_generation
        enable_debug_logging()
        print("   ✅ Debug logging enabled!")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   Make sure you're running this from the project root directory.")
        return
    
    # Method 2: Manual debug logging setup
    print("\n2. Alternative manual setup:")
    print("   You can also enable debug logging manually:")
    print("   ```python")
    print("   import logging")
    print("   logger = logging.getLogger('ai_diplomacy.prompt_constructor')")
    print("   logger.setLevel(logging.DEBUG)")
    print("   ```")
    
    # Method 3: Example usage in a real scenario
    print("\n3. Example usage in your agent code:")
    print("   ```python")
    print("   # After getting LLM response:")
    print("   raw_response = await client.generate_response(prompt)")
    print("   ")
    print("   # Try to parse orders")
    print("   try:")
    print("       parsed_orders = parse_orders(raw_response)")
    print("       parsing_error = None")
    print("   except Exception as e:")
    print("       parsed_orders = None")
    print("       parsing_error = str(e)")
    print("   ")
    print("   # Log everything for debugging")
    print("   log_llm_generation(")
    print("       power_name='FRANCE',")
    print("       prompt=prompt,")
    print("       raw_response=raw_response,")
    print("       parsed_orders=parsed_orders,")
    print("       parsing_error=parsing_error,")
    print("       phase='S1901M',")
    print("       generation_metadata={")
    print("           'temperature': 0.7,")
    print("           'max_tokens': 500,")
    print("           'model': 'gpt-4'")
    print("       }")
    print("   )")
    print("   ```")
    
    # Demo the logging function with sample data
    print("\n4. Demo with sample data:")
    print("   Running log_llm_generation with sample problematic response...")
    
    sample_prompt = """You are FRANCE in Diplomacy. Spring 1901.
Your units: A MAR, A PAR, F BRE
Choose your orders from: A MAR-SPA, A MAR-BUR, A MAR H, A PAR-BUR, A PAR-PIC, A PAR H, F BRE-MAO, F BRE-ENG, F BRE H

Respond in this format:
PARSABLE OUTPUT: {"orders": ["A MAR-SPA", "A PAR-BUR", "F BRE-MAO"]}"""

    sample_response = """I need to consider my opening strategy carefully. 

As France, I should focus on securing my home centers and positioning for expansion into either Spain or Germany.

My orders will be:
- Move A MAR to SPA to secure Spain
- Move A PAR to BUR to contest the center
- Move F BRE to MAO to support naval operations

PARSABLE OUTPUT: {"orders": ["A MAR-SPA", "A PAR-BUR", "F BRE-MAO"]}"""

    sample_problematic_response = """Looking at the board, I think France should take an aggressive stance.

I'll move my armies forward:
1. Army in Marseilles attacks Spain
2. Paris army goes to Burgundy  
3. Brest fleet sails to Mid-Atlantic

Here are my moves:
A MAR-SPA
A PAR-BUR
F BRE-MAO"""

    # Demo successful parsing
    print("\n   📋 Example 1: Successful parsing")
    log_llm_generation(
        power_name="FRANCE",
        prompt=sample_prompt,
        raw_response=sample_response,
        parsed_orders=["A MAR-SPA", "A PAR-BUR", "F BRE-MAO"],
        parsing_error=None,
        phase="S1901M",
        generation_metadata={
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 200
        }
    )
    
    # Demo parsing failure
    print("\n   📋 Example 2: Parsing failure (no JSON format)")
    log_llm_generation(
        power_name="FRANCE", 
        prompt=sample_prompt,
        raw_response=sample_problematic_response,
        parsed_orders=None,
        parsing_error="Could not find JSON format in response",
        phase="S1901M",
        generation_metadata={
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 200
        }
    )
    
    print("\n🎯 What the debug logging shows you:")
    print("   • The exact prompt sent to the LLM")
    print("   • The raw response (including escape characters)")
    print("   • Successfully parsed orders (if any)")
    print("   • Detailed parsing error information")
    print("   • Analysis of potential formatting issues")
    print("   • Suggestions for common problems (JSON, markdown, etc.)")
    
    print("\n📖 Usage in your code:")
    print("   1. Import: from ai_diplomacy.prompt_constructor import enable_debug_logging")
    print("   2. Enable: enable_debug_logging()")
    print("   3. Run your agent code - all LLM generations will be logged!")
    print("   4. Check the logs for detailed prompt and response information")
    
    print("\n✅ Debug logging demo complete!")

if __name__ == "__main__":
    demo_debug_logging()
