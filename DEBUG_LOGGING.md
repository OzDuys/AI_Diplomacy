# LLM Debug Logging Guide

## Overview

The AI Diplomacy system now includes comprehensive debug logging to help troubleshoot LLM generation and order parsing issues. When enabled, you can see the exact prompts sent to LLMs and their raw responses, making it much easier to understand why orders aren't being parsed correctly.

## Quick Start

### Enable Debug Logging

```python
from ai_diplomacy.prompt_constructor import enable_debug_logging

# Enable comprehensive debug logging
enable_debug_logging()

# Now run your agent code - all LLM generations will be logged!
```

### Alternative Manual Setup

```python
import logging

# Set up debug logging manually
logger = logging.getLogger('ai_diplomacy.prompt_constructor')
logger.setLevel(logging.DEBUG)

# Optional: add console handler if needed
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
```

## What Gets Logged

When debug logging is enabled, you'll see:

### 1. Full Prompt Construction
- **System Prompt**: The base instructions for the LLM
- **Context**: Complete game state, agent goals, relationships, diary
- **Instructions**: Task-specific instructions (orders, negotiation, etc.)
- **Final Prompt**: The complete assembled prompt sent to the LLM

### 2. LLM Generation Details
- **Prompt**: Exact text sent to the LLM (character count included)
- **Raw Response**: Complete LLM response with escape characters visible
- **Formatted Response**: Human-readable version of the response
- **Generation Metadata**: Model, temperature, token counts, etc.

### 3. Parsing Analysis
- **Successful Orders**: List of successfully parsed orders
- **Parsing Errors**: Detailed error messages with analysis
- **Response Analysis**: Automatic detection of formatting issues
- **Suggestions**: Hints about JSON, markdown, or other formatting problems

## Example Output

```
🤖 LLM GENERATION DEBUG LOG FOR FRANCE
================================================================================
📅 Phase: S1901M
🔧 Generation Metadata:
   model: gpt-4
   temperature: 0.7
   max_tokens: 500

📝 FULL PROMPT SENT TO LLM:
----------------------------------------
Length: 2156 characters
Content:
You are FRANCE in Diplomacy...
[Full prompt content]
----------------------------------------

🗣️ RAW LLM RESPONSE:
----------------------------------------
Length: 342 characters
Content:
'Looking at the board, I think France should...\n\nPARSABLE OUTPUT: {"orders": ["A MAR-SPA", "A PAR-BUR"]}'
Formatted content:
Looking at the board, I think France should...

PARSABLE OUTPUT: {"orders": ["A MAR-SPA", "A PAR-BUR"]}
----------------------------------------

✅ SUCCESSFULLY PARSED ORDERS:
----------------------------------------
  1. A MAR-SPA
  2. A PAR-BUR
Total orders: 2
----------------------------------------
```

## Parsing Error Example

When parsing fails, you get detailed analysis:

```
❌ PARSING ERROR:
----------------------------------------
Error: Could not find JSON format in response
----------------------------------------

🔍 RESPONSE ANALYSIS:
----------------------------------------
Response has 5 lines
Potential order lines found:
  Line 3: A MAR-SPA
  Line 4: A PAR-BUR
  Line 5: F BRE-MAO
⚠️ No JSON formatting detected - response may need format training
⚠️ Very few line breaks - response may be on single line
----------------------------------------
```

## Integration in Your Code

### Automatic Integration

The debug logging is automatically integrated into the order generation pipeline. Just enable it:

```python
from ai_diplomacy.prompt_constructor import enable_debug_logging

# Enable logging
enable_debug_logging()

# Run your game normally - all order generation will be logged
python lm_game.py --max_year 1905
```

### Manual Integration

For custom LLM calls, use the logging function directly:

```python
from ai_diplomacy.prompt_constructor import log_llm_generation

# After your LLM call
raw_response = await client.generate_response(prompt)

# Try to parse
try:
    parsed_orders = parse_orders(raw_response)
    parsing_error = None
except Exception as e:
    parsed_orders = None
    parsing_error = str(e)

# Log everything
log_llm_generation(
    power_name="FRANCE",
    prompt=prompt,
    raw_response=raw_response,
    parsed_orders=parsed_orders,
    parsing_error=parsing_error,
    phase="S1901M",
    generation_metadata={
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 500
    }
)
```

## Common Issues and Solutions

### Issue: No JSON in Response
**Symptoms**: LLM responds with plain text orders
**Debug Output**: "⚠️ No JSON formatting detected"
**Solution**: Update prompt templates to emphasize JSON format requirement

### Issue: Malformed JSON
**Symptoms**: JSON parsing errors
**Debug Output**: Shows exact JSON syntax errors
**Solution**: Review model training or add JSON repair logic

### Issue: Wrong Order Format
**Symptoms**: Orders parsed but validation fails
**Debug Output**: Shows "validation_failed" count
**Solution**: Check order format requirements vs. LLM output

### Issue: Context Too Long
**Symptoms**: Truncated responses
**Debug Output**: Shows prompt/response lengths
**Solution**: Reduce context length or increase max_tokens

## Performance Impact

Debug logging has minimal performance impact:
- **Enabled**: Logs are only written when debug level is active
- **Disabled**: Zero overhead (logging checks are optimized)
- **File I/O**: Async logging prevents blocking
- **Memory**: No additional storage of prompts/responses

## Configuration

### Disable Debug Logging

```python
from ai_diplomacy.prompt_constructor import disable_debug_logging
disable_debug_logging()
```

### Log Level Control

```python
import logging

# Only show parsing errors, not full prompts
logger = logging.getLogger('ai_diplomacy.prompt_constructor')
logger.setLevel(logging.WARNING)

# Show everything
logger.setLevel(logging.DEBUG)
```

## Best Practices

1. **Enable during development**: Always use debug logging when developing/testing agents
2. **Disable in production**: Turn off debug logging for performance in production runs
3. **Save logs**: Redirect output to files for later analysis
4. **Filter logs**: Use log levels to control verbosity
5. **Monitor trends**: Look for patterns in parsing failures across different models

## Troubleshooting Workflow

1. **Enable Debug Logging**: `enable_debug_logging()`
2. **Run Single Turn**: Test with one agent for one turn
3. **Check Prompt**: Verify the prompt looks correct
4. **Check Response**: Look at raw LLM response for issues
5. **Check Parsing**: See what orders were extracted
6. **Check Validation**: See what orders passed validation
7. **Iterate**: Adjust prompts/parsing based on findings

This debug logging system makes it much easier to understand and fix order parsing issues in your Diplomacy agents!
