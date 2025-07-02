# SGLang Function Calling Implementation

This document describes the function calling implementation for SGLang models in Xinference.

## Overview

The SGLang function calling implementation leverages SGLang's native `/parse_function_call` endpoint to parse function calls from generated text. This approach provides more accurate function call parsing compared to manual text parsing methods.

## Features

### ✅ Supported Features
- **Non-streaming function calls**: Full support for function calling in non-streaming mode
- **Multiple model families**: Support for Qwen, DeepSeek, and Llama3 model families
- **Automatic parser selection**: Automatically selects the appropriate tool call parser based on model family
- **Vision model support**: Function calling works with both regular chat and vision models
- **Native SGLang integration**: Uses SGLang's built-in function call parsing for accuracy

### ⚠️ Limitations
- **Streaming mode**: Function call parsing is not supported in streaming mode due to complexity of handling partial responses
- **SGLang dependency**: Requires SGLang's `/parse_function_call` endpoint to be available

## Supported Models

### Qwen Models (Parser: `qwen25`)
- qwen-chat
- qwen1.5-chat
- qwen2-instruct
- qwen2-moe-instruct
- qwen2.5-instruct ✨
- qwen2.5-coder-instruct ✨
- XiYanSQL-QwenCoder-2504
- QwQ-32B, QwQ-32B-Preview
- qwen3
- HuatuoGPT-o1-Qwen2.5
- DianJin-R1

### DeepSeek Models (Parser: `deepseek`)
- deepseek-v2.5
- deepseek-v2-chat
- deepseek-v2-chat-0628
- deepseek-v3
- deepseek-v3-0324
- deepseek-r1
- deepseek-r1-0528
- deepseek-r1-0528-qwen3
- deepseek-prover-v2
- deepseek-r1-distill-qwen
- deepseek-r1-distill-llama

### Llama3 Models (Parser: `llama3`)
- llama-3-instruct
- llama-3.1-instruct
- llama-3.3-instruct
- HuatuoGPT-o1-LLaMA-3.1

✨ = Commonly tested models

## Usage

### Basic Function Calling

```python
import openai

# Define your tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string", 
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature"
                    },
                },
                "required": ["location"],
            },
        },
    }
]

# Initialize client
client = openai.Client(
    api_key="not-empty",
    base_url="http://localhost:9997/v1"
)

# Make function calling request
response = client.chat.completions.create(
    model="qwen2.5-instruct",  # Any supported SGLang model
    messages=[
        {"role": "user", "content": "What's the weather like in Tokyo?"}
    ],
    tools=tools,
    stream=False  # Streaming not supported for function calls
)

# Check for function calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

### Vision Model Function Calling

```python
# Same API works for vision models like qwen2.5-vl-instruct
response = client.chat.completions.create(
    model="qwen2.5-vl-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image? Also, what's the weather like in the location shown?"},
                {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}}
            ]
        }
    ],
    tools=tools,
    stream=False
)
```

## Implementation Details

### Parser Selection

The implementation automatically selects the appropriate parser based on the model family:

```python
def _get_tool_call_parser(self, model_family: str) -> str:
    if model_family in QWEN_TOOL_CALL_FAMILY:
        return "qwen25" 
    elif model_family in DEEPSEEK_TOOL_CALL_FAMILY:
        return "deepseek"
    elif model_family in LLAMA3_TOOL_CALL_FAMILY:
        return "llama3"
    else:
        return "qwen25"  # Default fallback
```

### Function Call Parsing

The implementation uses SGLang's native parsing endpoint:

```python
async def _parse_function_calls(self, generated_text: str, tools: List[Dict], tool_call_parser: Optional[str] = None) -> Dict:
    parse_url = f"{self._engine.generate_url.replace('/generate', '/parse_function_call')}"
    
    function_call_input = {
        "text": generated_text,
        "tool_call_parser": tool_call_parser,
        "tools": tools,
    }
    
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.post(parse_url, json=function_call_input) as response:
            return await response.json()
```

### Response Format

The implementation returns standard OpenAI-compatible function call responses:

```json
{
    "id": "chatcmpl-123",
    "model": "qwen2.5-instruct",
    "object": "chat.completion",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"
                        }
                    }
                ]
            },
            "finish_reason": "tool_calls"
        }
    ]
}
```

## Testing

Use the provided test script to verify function calling works:

```bash
python test_sglang_function_calling.py
```

Make sure you have:
1. Xinference server running on localhost:9997
2. An SGLang model loaded (e.g., qwen2.5-instruct)
3. The model supports function calling

## Error Handling

The implementation includes proper error handling:

- If the `/parse_function_call` endpoint is not available, it logs a warning and returns the text without function calls
- If parsing fails, it gracefully falls back to treating the response as regular text
- Tool validation follows OpenAI's schema requirements

## Integration with Other Xinference Features

The function calling implementation integrates seamlessly with:
- ✅ Model management
- ✅ OpenAI-compatible API
- ✅ Vision models
- ✅ Multi-GPU support
- ✅ Distributed inference
- ⚠️ Streaming (not supported for function calls)

## Future Improvements

Potential enhancements:
1. **Streaming support**: Implement partial function call parsing for streaming responses
2. **Tool choice**: Add support for `tool_choice` parameter
3. **Parallel tools**: Enhanced support for multiple simultaneous function calls
4. **Custom parsers**: Allow custom tool call parsers for specific use cases 