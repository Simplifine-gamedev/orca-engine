"""
Explorer Agent for Godot AI Chat
© 2025 Simplifine Corp.

A specialized sub-agent for deep codebase exploration. The Explorer is given
search/exploration tools and tasked with finding specific information in a
Godot project. It yields streaming results including tool calls, and concludes
with a submit_exploration_report call containing cited findings.
"""

import json
import copy
import time
import uuid
from typing import Generator, Dict, Any, List, Optional
from litellm import completion
from Godot_tools import godot_tools

# ============================================================================
# EXPLORATION TOOLS EXTRACTION
# ============================================================================

def get_exploration_tools() -> List[Dict]:
    """
    Extract the search/exploration subset of tools from godot_tools.
    These are the tools the Explorer agent can use to investigate the codebase.
    """
    exploration_tool_names = {
        "project_manager",   # For fs.read, fs.list, context.get, project.analyze_dir
        "search_manager",    # For project.search, docs.search, scene.composition_tree  
        "graph_manager",     # For graph.neighbors, graph.walk
        "scene_manager",     # For scene.analyze, scene.nodes.get_all, scene.nodes.find_by_type
        "script_manager",    # For script.get_for_node, classes.custom_list
    }
    
    # Deep copy to prevent mutation
    tools = []
    for tool in godot_tools:
        if tool.get("function", {}).get("name") in exploration_tool_names:
            tools.append(copy.deepcopy(tool))
    
    # Add the submit_exploration_report tool
    tools.append({
        "type": "function",
        "function": {
            "name": "submit_exploration_report",
            "description": "REQUIRED: Call this tool LAST after you have thoroughly explored the codebase and gathered all relevant information. This submits your final findings to the user. Your report MUST include file paths and line numbers as citations for all findings.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "report": {
                        "type": "string",
                        "maxLength": 3000,
                        "description": """BE BRIEF! Short, direct report. Format:

**Found:** 1-2 sentence answer.

**Key Files:**
- `res://path/file.gd` (lines X-Y): What it does

**How it works:** 2-3 sentences max explaining the mechanism.

NO long paragraphs. NO filler text. Just facts and citations."""
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "How confident you are in your findings. 'high' = found definitive answers with clear code references. 'medium' = found relevant code but may have missed some details. 'low' = limited information found, recommend further investigation."
                    },
                    "files_explored": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of all file paths you examined during exploration (for transparency)."
                    },
                    "search_queries_used": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search queries you used (for transparency and reproducibility)."
                    }
                },
                "required": ["report", "confidence", "files_explored", "search_queries_used"]
            }
        }
    })
    
    return tools


# ============================================================================
# EXPLORER SYSTEM PROMPT
# ============================================================================

EXPLORER_SYSTEM_PROMPT = """You are the **Explorer Agent**, a specialized AI investigator for Godot game projects.

## YOUR MISSION
Explore the codebase to answer the given question. Be BRIEF and DIRECT in your final report.

## YOUR TOOLS
- `search_manager` (op: project.search) - Semantic/keyword search across codebase
- `project_manager` (op: fs.read) - Read file contents
- `project_manager` (op: fs.list) - List directory contents
- `graph_manager` (op: graph.neighbors) - Find related files
- `scene_manager` (op: scene.analyze) - Analyze scene structure

## EXPLORATION STRATEGY
1. Search 2-3 times with different keywords
2. Read the most relevant files you find
3. Follow references to related files if needed
4. Stop when you have enough information

## CRITICAL RULES
- Call multiple search/read tools before concluding
- ALWAYS cite file paths and line numbers
- If nothing found, say so clearly
- End with submit_exploration_report

## REPORT FORMAT - BE BRIEF!
Keep your report SHORT and TO THE POINT. No fluff, no filler text.

Format:
```
**Found:** [1-2 sentence summary]

**Key Files:**
- `path/file.gd` (lines X-Y): Brief description

**How it works:** [2-3 sentences max]
```

DO NOT write long paragraphs. Use bullet points. Be concise."""


# ============================================================================
# EXPLORER AGENT IMPLEMENTATION  
# ============================================================================

class ExplorerAgent:
    """
    A specialized agent for deep codebase exploration.
    
    The Explorer uses a subset of Godot tools focused on search and file reading.
    It's designed to be thorough, calling multiple tools before reaching conclusions.
    """
    
    def __init__(self, model: str = "openai/gpt-4o"):
        self.model = model
        self.tools = get_exploration_tools()
        self.exploration_id = str(uuid.uuid4())[:8]
        
    def explore(
        self, 
        question: str,
        project_context: Optional[Dict] = None,
        max_turns: int = 15,
        on_tool_call: Optional[callable] = None,
        on_tool_result: Optional[callable] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run an exploration session to answer a question about the codebase.
        
        Args:
            question: The question or topic to explore
            project_context: Optional project context (project_root, etc.)
            max_turns: Maximum number of LLM turns before forcing conclusion
            on_tool_call: Callback when a tool call is made (for frontend execution)
            on_tool_result: Callback to get tool results (for frontend execution)
            
        Yields:
            Dict events with keys:
                - type: "started" | "thinking" | "tool_call" | "tool_result" | "text" | "report" | "completed" | "error"
                - data: Event-specific data
        """
        
        # Initialize exploration
        yield {
            "type": "started",
            "data": {
                "exploration_id": self.exploration_id,
                "question": question,
                "timestamp": time.time()
            }
        }
        
        # Build initial messages
        messages = [
            {"role": "system", "content": EXPLORER_SYSTEM_PROMPT}
        ]
        
        # Add project context if provided
        if project_context:
            context_str = json.dumps({
                "type": "project_context",
                "project_root": project_context.get("project_root"),
                "project_name": project_context.get("project_name"),
                "scenes_count": project_context.get("scenes_count"),
                "scripts_count": project_context.get("scripts_count")
            }, ensure_ascii=False)
            messages.append({"role": "system", "content": context_str})
        
        # Add the exploration question
        messages.append({
            "role": "user",
            "content": f"""Please explore this topic thoroughly:

**Question:** {question}

Remember:
1. Search multiple times with different queries
2. Read the actual files you find, don't just list them
3. Follow references to related files
4. Cite everything with file paths and line numbers
5. End by calling submit_exploration_report with your findings"""
        })
        
        turn = 0
        report_submitted = False
        files_explored = set()
        queries_used = []
        
        while turn < max_turns and not report_submitted:
            turn += 1
            
            yield {
                "type": "thinking", 
                "data": {"turn": turn, "max_turns": max_turns}
            }
            
            try:
                # Call LLM
                response = completion(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    stream=True,
                    timeout=120
                )
                
                # Process streaming response
                full_text = ""
                tool_call_aggregator = {}
                tool_ids = {}
                
                for chunk in response:
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta
                    
                    # Handle text content
                    content = getattr(delta, 'content', None)
                    if content:
                        full_text += content
                        yield {"type": "text", "data": {"delta": content}}
                    
                    # Handle tool calls
                    tool_calls = getattr(delta, 'tool_calls', None)
                    if tool_calls:
                        for tc in tool_calls:
                            index = tc.index
                            key = f"tc_{index}"
                            
                            if key not in tool_call_aggregator:
                                tool_call_aggregator[key] = {"name": "", "arguments": ""}
                                tool_ids[key] = tc.id or f"exp_{self.exploration_id}_{index}"
                            
                            if tc.function:
                                if tc.function.name:
                                    tool_call_aggregator[key]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_call_aggregator[key]["arguments"] += tc.function.arguments
                
                # Add assistant message to history
                assistant_message = {"role": "assistant", "content": full_text}
                
                if tool_call_aggregator:
                    # Process tool calls
                    tool_calls_list = []
                    
                    for key, func in tool_call_aggregator.items():
                        tool_id = tool_ids[key]
                        tool_name = func["name"]
                        tool_args_str = func["arguments"]
                        
                        try:
                            tool_args = json.loads(tool_args_str) if tool_args_str else {}
                        except json.JSONDecodeError:
                            tool_args = {}
                        
                        tool_calls_list.append({
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_args_str}
                        })
                        
                        # Track exploration metrics
                        if tool_name == "search_manager":
                            query = tool_args.get("query", "")
                            if query:
                                queries_used.append(query)
                        elif tool_name == "project_manager":
                            path = tool_args.get("path", "")
                            if path:
                                files_explored.add(path)
                        
                        # Yield tool call event
                        yield {
                            "type": "tool_call",
                            "data": {
                                "tool_id": tool_id,
                                "tool_name": tool_name,
                                "arguments": tool_args
                            }
                        }
                        
                        # Check for submit_exploration_report
                        if tool_name == "submit_exploration_report":
                            report_submitted = True
                            yield {
                                "type": "report",
                                "data": {
                                    "report": tool_args.get("report", ""),
                                    "confidence": tool_args.get("confidence", "medium"),
                                    "files_explored": tool_args.get("files_explored", list(files_explored)),
                                    "search_queries_used": tool_args.get("search_queries_used", queries_used)
                                }
                            }
                            
                            # Add synthetic tool result for report submission
                            messages.append(assistant_message)
                            messages[-1]["tool_calls"] = tool_calls_list
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": json.dumps({"success": True, "message": "Report submitted successfully"})
                            })
                            break
                        
                        # Execute tool and get result
                        if on_tool_call and on_tool_result:
                            # Frontend execution mode - yield and wait for result
                            tool_result = on_tool_result(tool_id, tool_name, tool_args)
                        else:
                            # Backend execution mode - execute directly
                            tool_result = self._execute_tool_backend(tool_name, tool_args)
                        
                        yield {
                            "type": "tool_result",
                            "data": {
                                "tool_id": tool_id,
                                "tool_name": tool_name,
                                "result": tool_result
                            }
                        }
                    
                    if not report_submitted:
                        # Add assistant message with tool calls to history
                        assistant_message["tool_calls"] = tool_calls_list
                        messages.append(assistant_message)
                        
                        # Add tool results to history
                        for key, func in tool_call_aggregator.items():
                            tool_id = tool_ids[key]
                            tool_name = func["name"]
                            tool_args_str = func["arguments"]
                            
                            try:
                                tool_args = json.loads(tool_args_str) if tool_args_str else {}
                            except:
                                tool_args = {}
                            
                            if on_tool_call and on_tool_result:
                                tool_result = on_tool_result(tool_id, tool_name, tool_args)
                            else:
                                tool_result = self._execute_tool_backend(tool_name, tool_args)
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)
                            })
                else:
                    # No tool calls - just text response
                    messages.append(assistant_message)
                    
                    # If we're past turn 10 and no report yet, remind the agent
                    if turn >= 10 and not report_submitted:
                        messages.append({
                            "role": "user",
                            "content": "You've done a lot of exploration. Please now call submit_exploration_report with your findings. Remember to cite all file paths and line numbers."
                        })
                        
            except Exception as e:
                yield {
                    "type": "error",
                    "data": {"error": str(e), "turn": turn}
                }
                break
        
        # Force report if max turns reached
        if not report_submitted:
            yield {
                "type": "report",
                "data": {
                    "report": "Exploration reached maximum turns without completing. Please try a more specific question.",
                    "confidence": "low",
                    "files_explored": list(files_explored),
                    "search_queries_used": queries_used,
                    "incomplete": True
                }
            }
        
        yield {
            "type": "completed",
            "data": {
                "exploration_id": self.exploration_id,
                "total_turns": turn,
                "files_explored_count": len(files_explored),
                "queries_used_count": len(queries_used)
            }
        }
    
    def _execute_tool_backend(self, tool_name: str, tool_args: Dict) -> Dict:
        """
        Execute a tool on the backend using the actual backend implementations.
        
        This calls the real internal functions from app.py for search and graph operations.
        For frontend-only operations (fs.read, fs.list), returns a placeholder.
        """
        op = tool_args.get("op", "")
        
        try:
            # Import the actual backend functions
            from app import (
                search_manager_internal,
                graph_manager_internal,
                project_manager_internal,
                scene_manager_internal,
                script_manager_internal
            )
            
            # Route to appropriate internal function
            if tool_name == "search_manager":
                # search_manager handles: docs.search, project.search (semantic/keyword/hybrid)
                result = search_manager_internal(tool_args, None)
                return result
            
            elif tool_name == "graph_manager":
                # graph_manager handles: graph.neighbors, graph.walk
                result = graph_manager_internal(tool_args)
                return result
            
            elif tool_name == "project_manager":
                # project_manager handles: context.get, project.analyze_dir
                # Note: fs.read and fs.list may require frontend for local filesystem
                if op in ["context.get", "project.analyze_dir"]:
                    result = project_manager_internal(tool_args)
                    return result
                else:
                    # fs.read, fs.list need actual filesystem access
                    return {
                        "success": False,
                        "requires_frontend": True,
                        "message": f"project_manager op '{op}' requires frontend execution for filesystem access"
                    }
            
            elif tool_name == "scene_manager":
                result = scene_manager_internal(tool_args)
                return result
            
            elif tool_name == "script_manager":
                result = script_manager_internal(tool_args)
                return result
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}"
                }
                
        except ImportError as e:
            print(f"EXPLORER: Failed to import backend functions: {e}")
            return {
                "success": False,
                "error": f"Backend import error: {e}"
            }
        except Exception as e:
            print(f"EXPLORER: Tool execution error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }


# ============================================================================
# STREAMING EXPLORATION ENDPOINT GENERATOR
# ============================================================================

def stream_exploration(
    question: str,
    project_context: Optional[Dict] = None,
    model: str = "openai/gpt-4o",
    tool_executor: Optional[callable] = None
) -> Generator[str, None, None]:
    """
    Stream an exploration session as NDJSON lines.
    
    This is the main entry point for the Flask endpoint.
    
    Args:
        question: The question to explore
        project_context: Optional project context
        model: LLM model to use
        tool_executor: Optional callback to execute tools on frontend
        
    Yields:
        NDJSON lines (JSON strings followed by newlines)
    """
    explorer = ExplorerAgent(model=model)
    
    for event in explorer.explore(
        question=question,
        project_context=project_context,
        on_tool_call=None,  # Will be set up for frontend mode
        on_tool_result=tool_executor
    ):
        # Convert event to NDJSON line
        yield json.dumps(event) + '\n'


# ============================================================================
# TOOL DEFINITION FOR MAIN AGENT
# ============================================================================

EXPLORE_CODEBASE_TOOL = {
    "type": "function",
    "function": {
        "name": "explore_codebase",
        "description": """Launch the Explorer Agent to deeply investigate a topic in the Godot project.

The Explorer will:
1. Perform multiple semantic and keyword searches
2. Read and analyze relevant files
3. Follow references between files using the project graph
4. Compile findings with full citations (file paths + line numbers)

Use this tool when you need thorough investigation of:
- How a feature is implemented
- Where specific functionality exists
- Connections between different parts of the codebase
- Understanding complex systems or patterns

The Explorer returns a comprehensive report with cited findings.""",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "question": {
                    "type": "string",
                    "maxLength": 1000,
                    "description": "The specific question or topic to explore. Be as specific as possible for best results. Examples: 'How does the player movement system work?', 'Where is the inventory system implemented?', 'What signals does the GameManager emit?'"
                },
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of directories or file patterns to focus the search on. Example: ['res://scripts/player/', 'res://systems/']"
                },
                "depth": {
                    "type": "string",
                    "enum": ["quick", "normal", "thorough"],
                    "default": "normal",
                    "description": "'quick' (3-5 searches), 'normal' (5-10 searches), 'thorough' (10-15 searches, follows more references)"
                }
            },
            "required": ["question"]
        }
    }
}


def get_explore_codebase_tool_def() -> Dict:
    """Get the explore_codebase tool definition for inclusion in Godot_tools."""
    return copy.deepcopy(EXPLORE_CODEBASE_TOOL)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Simple test of the explorer
    print("Testing Explorer Agent...")
    print(f"Loaded {len(get_exploration_tools())} exploration tools:")
    for tool in get_exploration_tools():
        print(f"  - {tool['function']['name']}")
    
    print("\nExplore codebase tool definition:")
    print(json.dumps(get_explore_codebase_tool_def(), indent=2))

