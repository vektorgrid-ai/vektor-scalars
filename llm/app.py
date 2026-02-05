import time
import json
import ollama


async def infer_llm(data: dict):
    start = time.perf_counter()

    req_id = data.get("request_id")

    # Input
    user_prompt = data["input"].get("prompt", "")
    chat_context = data["input"].get("chat_context", {})
    tool_schemas = data["input"].get("tools", [])

    # Config
    model = data["config"].get("model", "llama3.2")
    temperature = data["config"].get("temperature", 0.2)
    max_tokens = data["config"].get("max_tokens", 4096)

    # Optional external context
    context = data.get("context", {})

    # Build messages
    messages = build_messages(user_prompt, chat_context, context)

    # Convert tools to Ollama tool schema
    tools = convert_tools(tool_schemas)

    # Run Ollama Chat
    if not check_model_exists(model):
        print(f"Model {model} not found. Pulling...")
        ollama.pull(model)
    response = ollama.chat(
        model=model,
        messages=messages,
        tools=tools,
        options={
            "temperature": temperature,
            "num_predict": max_tokens
        }
    )

    msg = response["message"]

    # If model returned tool calls, return them (DO NOT EXECUTE)
    if msg.get("tool_calls"):
        output = {
            "tool_calls": msg["tool_calls"]
        }
    else:
        output = {
            "text": msg.get("content", "")
        }

    end = time.perf_counter()
    latency_ms = int((end - start) * 1000)

    return {
        "request_id": req_id,
        "output": output,
        "usage": {
            "latency_ms": latency_ms,
            "model": model
        },
        "error": None
    }


# ---------------------------
# Build Chat Messages
# ---------------------------

def build_messages(user_prompt, chat_context, context):
    messages = []

    # External context → system message
    if context:
        messages.append({
            "role": "system",
            "content": f"Context: {json.dumps(context)}"
        })

    # Prior conversation (ignore timestamps/chat_id)
    for event in chat_context.get("events", []):
        role = "assistant" if event.get("$type") == "assistant" else "user"
        text = event.get("text")
        if text:
            messages.append({
                "role": role,
                "content": text
            })

    # New user input
    messages.append({
        "role": "user",
        "content": user_prompt
    })

    return messages

def convert_tools(tool_schemas):
    tools = []

    for tool in tool_schemas:
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": {
                        p["name"]: {
                            "type": p.get("type", "string"),
                            "description": p.get("description", "")
                        }
                        for p in tool.get("parameters", [])
                    },
                    "required": [
                        p["name"] for p in tool.get("parameters", [])
                        if p.get("required")
                    ]
                }
            }
        })

    return tools

def check_model_exists(model):
    res = ollama.list()
    return model in [m["model"] for m in res.models]