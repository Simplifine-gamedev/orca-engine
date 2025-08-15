"""
© 2025 Simplifine Corp. Original backend example for Orca Engine.
Free personal/non-commercial use only under Company Non‑Commercial License.
Commercial use requires a separate license from Simplifine. See LICENSES/COMPANY-NONCOMMERCIAL.md.
"""
# pip install litellm python-dateutil
import os, json, sys, time
from datetime import datetime
from dateutil import tz
import dotenv
from litellm import completion
dotenv.load_dotenv()

# --- model mapping ---
# swap values to the exact models you have access to.
MODEL_MAP = {
    "gemini-2.5":   os.getenv("GEMINI_MODEL",   "gemini/gemini-2.5-pro"),
    "claude-4":     os.getenv("CLAUDE_MODEL",   "anthropic/claude-sonnet-4-20250514"),
    "gpt-5":        os.getenv("OPENAI_MODEL",   "openai/gpt-5"),
}

# --- a tiny tool ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Return the current local time for a given IANA timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA tz, e.g. America/Los_Angeles"}
                },
                "required": ["timezone"]
            },
        },
    }
]

def get_time(timezone: str):
    try:
        tzinfo = tz.gettz(timezone)
        if tzinfo is None:
            raise ValueError("Unknown timezone")
        now = datetime.now(tzinfo).strftime("%Y-%m-%d %H:%M:%S %Z")
        return {"now": now}
    except Exception as e:
        return {"error": str(e)}

FUNCTIONS = {"get_time": get_time}

SYSTEM = {"role": "system", "content": "You are concise. If a tool is available and useful, call it."}
USER = {
    "role": "user",
    "content": "What time is it in America/Los_Angeles? Also say 'pong' at the end."
}
BASE_MESSAGES = [SYSTEM, USER]

def _parse_provider_and_model(real_model: str):
    # Expecting format like "provider/model_id"; if missing, return None provider
    if "/" in real_model:
        provider, model_id = real_model.split("/", 1)
        return provider.strip(), model_id.strip()
    return None, real_model


def stream_and_collect(label, model_id, messages, tools=None):
    print(f"\n=== {label} ===")
    tool_calls_accumulator = {}
    last_tool_call_key = None

    # 1) first turn: allow tool calls, stream tokens
    stream = completion(model=model_id, messages=messages, tools=tools, stream=True)
    for chunk in stream:
        delta = (chunk.choices[0].delta or {})
        content = delta.get("content")
        if content:
            # Show streaming by printing each character with a small delay
            for char in content:
                sys.stdout.write(char)
                sys.stdout.flush()
                # time.sleep(0.02)  # 20ms delay to make streaming visible



        # Accumulate streamed tool_call function args across deltas (Anthropic/OpenAI style)
        tool_calls_delta = delta.get("tool_calls") or []
        for tc in tool_calls_delta:
            # Handle both dict and pydantic object forms
            if hasattr(tc, 'id'):  # Pydantic object
                tc_id = tc.id
                tc_index = getattr(tc, 'index', 0)
                tc_type = getattr(tc, 'type', 'function')
                fn = tc.function
                fn_name = getattr(fn, 'name', None) if fn else None
                fn_args = getattr(fn, 'arguments', '') if fn else ''
            else:  # Dict form
                tc_id = tc.get("id")
                tc_index = tc.get('index', 0)
                tc_type = tc.get("type", "function")
                fn = tc.get("function") or {}
                fn_name = fn.get("name")
                fn_args = fn.get("arguments") or ''
            
            # Use index as the key since id becomes None in subsequent chunks
            key = f"tool_call_{tc_index}"
            last_tool_call_key = key

            # Initialize accumulator with proper fallback ID
            if key not in tool_calls_accumulator:
                tool_calls_accumulator[key] = {
                    "id": tc_id or key, 
                    "type": tc_type, 
                    "function": {"name": "", "arguments": ""}
                }
            
            acc = tool_calls_accumulator[key]
            # Preserve the original ID if we have it
            if tc_id and not acc["id"].startswith("tool_call_"):
                acc["id"] = tc_id
            if fn_name:
                acc["function"]["name"] = fn_name
            if fn_args:
                acc["function"]["arguments"] += fn_args

    print()

    # 2) if a tool was requested, execute it and do a follow-up streamed turn
    if last_tool_call_key and last_tool_call_key in tool_calls_accumulator:
        tool_call_full = tool_calls_accumulator[last_tool_call_key]
        func = tool_call_full.get("function", {})
        name = func.get("name")
        args_json = func.get("arguments") or "{}"
        
        # Only proceed if we have a valid function name
        if not name or name.strip() == "":
            print("(No valid tool call detected)")
            return
        try:
            args = json.loads(args_json) if isinstance(args_json, str) else args_json
        except json.JSONDecodeError:
            args = {}

        result = FUNCTIONS.get(name, lambda **_: {"error": "unknown function"})(**args)
        messages = messages + [
            {"role": "assistant", "tool_calls": [tool_call_full]},
            {"role": "tool", "tool_call_id": tool_call_full.get("id", "tool_call_0"), "content": json.dumps(result)}
        ]

        # final streamed answer after tool result
        stream2 = completion(model=model_id, messages=messages, tools=tools, stream=True)
        for chunk in stream2:
            delta = (chunk.choices[0].delta or {})
            content = delta.get("content")
            if content:
                # Show streaming by printing each character with a small delay
                for char in content:
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    time.sleep(0.02)  # 20ms delay to make streaming visible
        print()

def main():
    for friendly, real in MODEL_MAP.items():
        stream_and_collect(f"{friendly} -> {real}", real, BASE_MESSAGES, TOOLS)

if __name__ == "__main__":
    main()
