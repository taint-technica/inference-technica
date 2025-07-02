#!/usr/bin/env python3
"""
Test script to demonstrate function calling with SGLang models in Xinference.

This script shows how to use function calling with SGLang models like qwen2.5-instruct
through Xinference's OpenAI-compatible API.

Prerequisites:
1. Xinference server running with an SGLang model loaded
2. Model should be one that supports function calling (e.g., qwen2.5-instruct)

Usage:
    python test_sglang_function_calling.py
"""

import openai

# Define a simple weather function for testing
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
                        "description": "The unit of temperature",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "Search for restaurants near a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city or area to search in",
                    },
                    "cuisine": {
                        "type": "string",
                        "description": "Type of cuisine (optional)",
                    },
                    "price_range": {
                        "type": "string",
                        "enum": ["$", "$$", "$$$", "$$$$"],
                        "description": "Price range for the restaurant",
                    },
                },
                "required": ["location"],
            },
        },
    },
]


def test_sglang_function_calling():
    """Test function calling with SGLang models"""

    # Initialize OpenAI client pointing to Xinference
    client = openai.Client(
        api_key="not-empty",  # Xinference doesn't validate API key
        base_url="http://localhost:9997/v1",
    )

    # Test messages that should trigger function calls
    test_cases = [
        {
            "name": "Weather Query",
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather like in Tokyo? Use celsius.",
                }
            ],
        },
        {
            "name": "Restaurant Search",
            "messages": [
                {
                    "role": "user",
                    "content": "Find me some good Italian restaurants in New York City, preferably in the $$ price range.",
                }
            ],
        },
        {
            "name": "Multi-step Query",
            "messages": [
                {
                    "role": "user",
                    "content": "I'm planning a trip to Paris. Can you check the weather there and also recommend some good French restaurants?",
                }
            ],
        },
    ]

    print("Testing SGLang Function Calling Implementation")
    print("=" * 50)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 30)

        try:
            # Make the function calling request
            response = client.chat.completions.create(
                model="qwen3",  # Use appropriate SGLang model
                messages=test_case["messages"],
                tools=tools,
                temperature=0.1,
            )

            print(f"Messages: {test_case['messages']}")
            print(f"Response: {response}")

            # Check if function calls were made
            if response.choices[0].message.tool_calls:
                print(
                    f"✓ Function calls detected: {len(response.choices[0].message.tool_calls)}"
                )
                for j, tool_call in enumerate(response.choices[0].message.tool_calls):
                    print(f"  Function {j + 1}: {tool_call.function.name}")
                    print(f"  Arguments: {tool_call.function.arguments}")
            else:
                print("✗ No function calls detected")
                print(f"Content: {response.choices[0].message.content}")

        except Exception as e:
            print(f"✗ Error: {e}")

    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    test_sglang_function_calling()
