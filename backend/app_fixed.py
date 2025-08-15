from flask import Flask, request, Response, jsonify
import openai
import json
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io
import requests

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Get OpenAI API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = openai.OpenAI(api_key=api_key)

# Model configuration
MODEL = "gpt-4o"  # Using GPT-4o which supports function calling

# --- Asset Processing Function ---
def process_asset_internal(arguments: dict) -> dict:
    """Process assets using various AI and image processing techniques"""
    try:
        operation = arguments.get('operation', '')
        input_path = arguments.get('input_path', '')
        
        if not operation:
            return {"success": False, "error": "No operation specified"}
        
        # For now, return a placeholder for asset processing operations
        # This would be expanded to include actual asset processing logic
        operations = {
            'remove_background': 'Background removal using AI',
            'auto_crop': 'Intelligent sprite boundary detection',
            'generate_spritesheet': 'Automatic sprite sheet generation',
            'style_transfer': 'Apply consistent art style',
            'batch_process': 'Process multiple assets',
            'classify': 'Classify asset types',
            'create_variants': 'Generate asset variations'
        }
        
        if operation not in operations:
            return {"success": False, "error": f"Unknown operation: {operation}"}
        
        return {
            "success": True,
            "message": f"Asset processing '{operation}' completed",
            "operation": operation,
            "description": operations[operation],
            "input_path": input_path
        }
            
    except Exception as e:
        return {"success": False, "error": f"Asset processing failed: {str(e)}"}

# --- Image Generation Function ---
def generate_image_internal(arguments: dict) -> dict:
    """Generate an image using OpenAI's DALL-E API"""
    try:
        prompt = arguments.get('prompt', '')
        style = arguments.get('style', 'digital art')
        
        if not prompt:
            return {"success": False, "error": "No prompt provided for image generation"}
        
        # Enhance the prompt with style if specified
        enhanced_prompt = f"{prompt}, {style} style" if style and style != "digital art" else prompt
        
        print(f"DALL-E: Generating image with prompt: '{enhanced_prompt}'")
        
        # Call OpenAI's image generation API
        response = client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        
        image_url = response.data[0].url
        
        # Download the image and convert to base64
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            # Convert to base64
            image_data = base64.b64encode(image_response.content).decode('utf-8')
            
            print(f"DALL-E: Successfully generated image ({len(image_data)} chars base64)")
            
            return {
                "success": True,
                "image_data": image_data,
                "prompt": enhanced_prompt,
                "style": style,
                "format": "png"
            }
        else:
            return {"success": False, "error": f"Failed to download generated image: {image_response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": f"Image generation failed: {str(e)}"}

# --- Tool Execution Function ---
def execute_godot_tool(function_name: str, arguments: dict) -> dict:
    """Execute backend-specific tools"""
    if function_name == "generate_image":
        return generate_image_internal(arguments)
    elif function_name == "asset_processor":
        return process_asset_internal(arguments)
    else:
        # This shouldn't happen if we filter correctly
        print(f"WARNING: Unknown backend tool called: {function_name}")

    return {"success": False, "error": f"Unknown backend tool called: {function_name}"}

# --- Individual Tool Definitions (Original 22 Tools) ---
godot_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_scene_info",
            "description": "Get information about the current scene including root node and structure",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_nodes",
            "description": "Get all nodes in the current scene with their information",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_nodes_by_type",
            "description": "Search for nodes by their type (e.g., 'Node2D', 'Button', 'CharacterBody2D')",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "The node type to search for"
                    }
                },
                "required": ["type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_editor_selection",
            "description": "Get currently selected nodes in the editor",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_node_properties",
            "description": "Get properties of a specific node",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_scene",
            "description": "Save the current scene",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_node",
            "description": "Create a new node in the scene",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Type of node to create (e.g., 'Node2D', 'Button', 'CharacterBody2D')"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name for the new node"
                    },
                    "parent": {
                        "type": "string",
                        "description": "Parent node path (optional)"
                    }
                },
                "required": ["type", "name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_node",
            "description": "Delete a node from the scene",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node to delete"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_node_property",
            "description": "Set a property on a node",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node"
                    },
                    "property": {
                        "type": "string",
                        "description": "Property name to set"
                    },
                    "value": {
                        "description": "Value to set for the property"
                    }
                },
                "required": ["path", "property", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_node",
            "description": "Move a node to a different parent",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node to move"
                    },
                    "new_parent": {
                        "type": "string",
                        "description": "Path to the new parent node"
                    }
                },
                "required": ["path", "new_parent"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_node_method",
            "description": "Call a method on a node",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node"
                    },
                    "method": {
                        "type": "string",
                        "description": "Method name to call"
                    },
                    "method_args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments for the method call"
                    }
                },
                "required": ["path", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_classes",
            "description": "Get list of all available node classes in Godot",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_node_script",
            "description": "Get the script attached to a node",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "attach_script",
            "description": "Attach a script to a node",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node"
                    },
                    "script_path": {
                        "type": "string",
                        "description": "Path to the script file"
                    }
                },
                "required": ["path", "script_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_scene",
            "description": "Manage scene operations (open, create, save, instantiate)",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["open", "create_new", "save_as", "instantiate"],
                        "description": "Scene operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "Scene file path"
                    },
                    "parent_node": {
                        "type": "string",
                        "description": "Parent node for instantiate operations"
                    }
                },
                "required": ["operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_collision_shape",
            "description": "Add a collision shape to a physics body node",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_path": {
                        "type": "string",
                        "description": "Path to the physics body node"
                    },
                    "shape_type": {
                        "type": "string",
                        "enum": ["rectangle", "circle", "capsule"],
                        "description": "Type of collision shape to create"
                    }
                },
                "required": ["node_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_project_files",
            "description": "List files and directories in the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir": {
                        "type": "string",
                        "description": "Directory to list (default: res://)"
                    },
                    "filter": {
                        "type": "string",
                        "description": "File filter (e.g., '*.gd', '*.tscn')"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_content",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_advanced",
            "description": "Read specific lines from a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (1-indexed)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number (inclusive)"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_edit",
            "description": "Apply AI-powered edits to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to edit"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Description of the edit to apply"
                    }
                },
                "required": ["path", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_compilation_errors",
            "description": "Check for compilation errors in a script file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Script file path to check"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image using AI",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt for image generation"
                    },
                    "style": {
                        "type": "string",
                        "description": "Art style (e.g., 'pixel_art', 'cartoon', 'realistic')"
                    }
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_scene",
            "description": "Run/play the current scene or a specific scene to observe behavior",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_path": {
                        "type": "string",
                        "description": "Path to scene file to run (optional, uses current scene if not specified)"
                    },
                    "duration": {
                        "type": "integer",
                        "description": "How long to run the scene in seconds (default: 5)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_scene_tree_hierarchy",
            "description": "Get complete scene tree hierarchy with parent-child relationships",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_properties": {
                        "type": "boolean",
                        "description": "Include basic properties for each node (default: false)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_physics_body",
            "description": "Inspect physics properties of a RigidBody2D, CharacterBody2D, or other physics body",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the physics body node"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_camera_info",
            "description": "Get camera and viewport information for debugging visual issues",
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_path": {
                        "type": "string",
                        "description": "Path to specific camera (optional, finds main camera if not specified)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Take a screenshot of the current game view for visual debugging",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename for the screenshot (optional)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_node_in_scene_tree",
            "description": "Verify if a node is properly added to the scene tree and active",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node to check"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_animation_state",
            "description": "Check animation player state and current animations",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to AnimationPlayer or AnimatedSprite2D node"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_layers_and_zindex",
            "description": "Get layer and z-index information for visual debugging",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node (optional, gets all if not specified)"
                    }
                },
                "required": []
            }
        }
    }
]


@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint that handles the full conversation flow:
    1. Receives messages from Godot
    2. Calls OpenAI API 
    3. Executes any tool calls
    4. Streams the final response back to Godot
    """
    data = request.json
    messages = data.get('messages', [])

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    def generate_stream():
        try:
            conversation_messages = messages.copy()

            while True:  # Loop to handle tool calling and responses
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=conversation_messages,
                    tools=godot_tools,
                    tool_choice="auto",
                    stream=True
                )

                full_text_response = ""
                tool_call_aggregator = {}
                tool_ids = {}
                current_tool_index = None
                
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        
                        # Handle streaming text content
                        if delta.content:
                            full_text_response += delta.content
                            yield json.dumps({
                                "content_delta": delta.content,
                                "status": "streaming"
                            }) + '\n'
                        
                        # Handle tool calls
                        if delta.tool_calls:
                            for tool_call in delta.tool_calls:
                                index = tool_call.index
                                current_tool_index = index
                                
                                if index not in tool_call_aggregator:
                                    tool_call_aggregator[index] = {
                                        "name": "",
                                        "arguments": ""
                                    }
                                    tool_ids[index] = tool_call.id or f"call_{index}"
                                
                                if tool_call.function:
                                    if tool_call.function.name:
                                        tool_call_aggregator[index]["name"] = tool_call.function.name
                                    if tool_call.function.arguments:
                                        tool_call_aggregator[index]["arguments"] += tool_call.function.arguments

                # Handle image generation (backend-only tool)
                image_generations = []
                if tool_call_aggregator:
                    for i, func in tool_call_aggregator.items():
                        if func["name"] == "generate_image":
                            try:
                                arguments = json.loads(func["arguments"])
                            except json.JSONDecodeError:
                                arguments = {}
                            
                            image_result = generate_image_internal(arguments)
                            if image_result.get("success"):
                                image_generations.append(image_result)
                                
                                # Remove generate_image from tool calls so it doesn't get sent to frontend
                                del tool_call_aggregator[i]
                                del tool_ids[i]

                # Case 1: Only image generations (no other function calls)
                if image_generations and not tool_call_aggregator:
                    # Add assistant message with generated images to conversation history
                    assistant_message = {
                        "role": "assistant",
                        "content": full_text_response or "I've generated the requested image(s).",
                        "images": image_generations
                    }
                    conversation_messages.append(assistant_message)
                    yield json.dumps({"status": "completed"}) + '\n'
                    break  # Exit the conversation loop
                
                # Case 1.5: Only image generations (no function calls)
                if image_generations and not tool_call_aggregator:
                    # Add assistant message with generated images to conversation history
                    assistant_message = {
                        "role": "assistant",
                        "content": full_text_response or "I've generated the requested image(s).",
                        "images": image_generations
                    }
                    conversation_messages.append(assistant_message)
                    yield json.dumps({"status": "completed"}) + '\n'
                    break  # Exit the conversation loop
                
                # Case 2: Tool calls were made, so execute them.
                
                # Assemble the full tool calls for OpenAI history
                final_tool_calls = []
                for i, func in sorted(tool_call_aggregator.items()):
                    final_tool_calls.append({
                        "id": tool_ids[i],
                        "type": "function",
                        "function": func
                    })
                
                # Add the complete assistant message to history
                assistant_message = {
                    "role": "assistant",
                    "content": full_text_response or None,
                    "tool_calls": final_tool_calls
                }
                conversation_messages.append(assistant_message)
                
                # Filter out backend-only tools before sending to frontend
                frontend_tool_calls = []
                backend_only_tools = ["generate_image"]
                for tool_call in final_tool_calls:
                    function_name = tool_call["function"]["name"]
                    if function_name not in backend_only_tools:
                        frontend_tool_calls.append(tool_call)
                
                # Create a frontend-specific assistant message
                frontend_assistant_message = {
                    "role": "assistant",
                    "content": full_text_response or None,
                    "tool_calls": frontend_tool_calls
                }
                
                # Dispatch frontend tools to Godot for execution
                if frontend_tool_calls:
                    print(f"FRONTEND_TOOLS: Sending {len(frontend_tool_calls)} tools to Godot for execution")
                    yield json.dumps({"assistant_message": frontend_assistant_message, "status": "executing_tools"}) + '\n'
                    
                    # IMPORTANT: Stop here and let the frontend send a new request with tool results
                    # The frontend will call _send_chat_request() again with the complete conversation including tool results
                    print(f"FRONTEND_TOOLS: Stopping conversation loop - waiting for frontend to send new request with tool results")
                    yield json.dumps({"status": "waiting_for_tool_results"}) + '\n'
                    break  # Exit the conversation loop and wait for new request
                
                # Execute any backend-only tools
                backend_tool_results = []
                for tool_call in final_tool_calls:
                    function_name = tool_call["function"]["name"]
                    if function_name in backend_only_tools:
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}

                        tool_result = execute_godot_tool(function_name, arguments)

                        tool_result_message = {
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(tool_result)
                        }
                        conversation_messages.append(tool_result_message)
                        backend_tool_results.append(tool_result_message)
                        
                        yield json.dumps({
                            "tool_executed": function_name,
                            "tool_result": tool_result,
                            "status": "tool_completed"
                        }) + '\n'
                
                # If we only had backend tools, continue the conversation
                # If we had frontend tools, we already broke out of the loop above
                if not frontend_tool_calls and backend_tool_results:
                    print(f"BACKEND_ONLY: Completed {len(backend_tool_results)} backend tools, continuing conversation")
                    continue  # Continue the conversation loop to send tool results back to the AI
                elif not frontend_tool_calls and not backend_tool_results:
                    print(f"NO_TOOLS: No tools to execute, ending conversation")
                    yield json.dumps({"status": "completed"}) + '\n'
                    break
                
        except Exception as e:
            print(f"ERROR: Exception in stream generation: {e}")
            yield json.dumps({"error": str(e), "status": "error"}) + '\n'

    return Response(generate_stream(), mimetype='application/x-ndjson')

@app.route('/predict_code_edit', methods=['POST'])
def predict_code_edit():
    """
    Uses OpenAI's Predicted Outputs feature to edit code files.
    Provides the original file content as prediction and returns the complete edited file.
    """
    data = request.json
    file_content = data.get('file_content', '')
    prompt = data.get('prompt')
    
    print(f"APPLY_EDIT_REQUEST: Received request with prompt: '{prompt}' for file content length: {len(file_content)}")

    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    # Simple, clear prompt for file editing
    user_prompt = f"Edit this file according to the request: {prompt}\n\nReturn the complete edited file content."

    try:
        # Simple approach: provide file content in context and ask for complete edited file
        full_prompt = f"""Edit this file according to the request: {prompt}

Original file content:
```
{file_content}
```

Return the complete edited file content (no explanations, just the code):"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )

        edited_content = response.choices[0].message.content
        print(f"APPLY_EDIT_MODEL: Received edited file content (length: {len(edited_content)})")
        
        # Return the complete edited file content
        return jsonify({
            "edited_content": edited_content,
            "success": True
        })
        
    except Exception as e:
        print(f"APPLY_EDIT_ERROR: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)