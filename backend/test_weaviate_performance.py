#!/usr/bin/env python3
"""Test Weaviate vector manager performance"""
import os
import time
from dotenv import load_dotenv
import openai
import numpy as np

# Load environment variables
load_dotenv()

# Test configuration
WEAVIATE_URL = os.getenv('WEAVIATE_URL')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not all([WEAVIATE_URL, WEAVIATE_API_KEY, OPENAI_API_KEY]):
    print("Missing required environment variables")
    exit(1)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Import our Weaviate manager
from weaviate_vector_manager import WeaviateVectorManager

print("Testing Weaviate Vector Manager Performance")
print("=" * 50)

# Initialize manager
print("\n1. Initializing Weaviate connection...")
start = time.time()
manager = WeaviateVectorManager(WEAVIATE_URL, WEAVIATE_API_KEY, client)
print(f"   ✓ Initialized in {time.time() - start:.2f}s")

# Test data
test_files = [
    {
        'file_path': 'test/player.gd',
        'content': '''extends CharacterBody2D

signal health_changed(new_health)
signal died

@export var max_health: int = 100
@export var move_speed: float = 300.0
@export var jump_velocity: float = -400.0

var current_health: int
var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

func _ready():
    current_health = max_health
    
func _physics_process(delta):
    # Add gravity
    if not is_on_floor():
        velocity.y += gravity * delta
    
    # Handle jump
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = jump_velocity
    
    # Handle movement
    var direction = Input.get_axis("ui_left", "ui_right")
    if direction:
        velocity.x = direction * move_speed
    else:
        velocity.x = move_toward(velocity.x, 0, move_speed)
    
    move_and_slide()

func take_damage(amount: int):
    current_health -= amount
    health_changed.emit(current_health)
    if current_health <= 0:
        die()

func die():
    died.emit()
    queue_free()
'''
    },
    {
        'file_path': 'test/enemy.gd',
        'content': '''extends Area2D

@export var damage: int = 10
@export var patrol_speed: float = 100.0
@export var patrol_distance: float = 200.0

var start_position: Vector2
var direction: int = 1

func _ready():
    start_position = position
    body_entered.connect(_on_body_entered)

func _process(delta):
    # Simple patrol movement
    position.x += patrol_speed * direction * delta
    
    # Reverse direction at patrol limits
    if abs(position.x - start_position.x) > patrol_distance:
        direction *= -1
        scale.x *= -1  # Flip sprite

func _on_body_entered(body):
    if body.has_method("take_damage"):
        body.take_damage(damage)
'''
    },
    {
        'file_path': 'test/main_menu.gd',
        'content': '''extends Control

signal start_game
signal quit_game

@onready var start_button = $VBoxContainer/StartButton
@onready var options_button = $VBoxContainer/OptionsButton
@onready var quit_button = $VBoxContainer/QuitButton

func _ready():
    start_button.pressed.connect(_on_start_pressed)
    options_button.pressed.connect(_on_options_pressed)
    quit_button.pressed.connect(_on_quit_pressed)
    
    # Focus first button
    start_button.grab_focus()

func _on_start_pressed():
    start_game.emit()
    get_tree().change_scene_to_file("res://levels/level_1.tscn")

func _on_options_pressed():
    # TODO: Implement options menu
    print("Options not implemented yet")

func _on_quit_pressed():
    quit_game.emit()
    get_tree().quit()
'''
    }
]

# Generate more test files
for i in range(20):
    test_files.append({
        'file_path': f'test/generated/script_{i}.gd',
        'content': f'''# Generated test file {i}
extends Node

var test_var_{i} = {i * 10}

func test_function_{i}():
    print("This is test function {i}")
    return test_var_{i} * 2

func calculate_something_{i}(input: int) -> int:
    var result = input + test_var_{i}
    for j in range({i}):
        result += j
    return result
'''
    })

# Test indexing performance
print(f"\n2. Testing indexing performance with {len(test_files)} files...")
start = time.time()
indexing_stats = manager.index_files_with_content(
    test_files, 
    user_id="test_user",
    project_id="test_project",
    max_workers=10
)
indexing_elapsed = time.time() - start
print(f"   ✓ Indexed in {indexing_elapsed:.2f}s")
print(f"   - Files indexed: {indexing_stats['indexed']}")
print(f"   - Chunks created: {indexing_stats['chunks']}")
print(f"   - Rate: {indexing_stats['chunks']/indexing_elapsed:.1f} chunks/sec")

# Test search performance
test_queries = [
    "how to handle player movement",
    "damage system implementation",
    "menu button handling",
    "physics process delta",
    "signal emission pattern",
    "calculate something function"
]

print(f"\n3. Testing search performance with {len(test_queries)} queries...")
search_times = []
for query in test_queries:
    start = time.time()
    results = manager.search(
        query=query,
        user_id="test_user",
        project_id="test_project",
        max_results=5
    )
    elapsed = time.time() - start
    search_times.append(elapsed)
    print(f"   - '{query}': {elapsed:.3f}s ({len(results)} results)")

avg_search_time = np.mean(search_times)
print(f"\n   ✓ Average search time: {avg_search_time:.3f}s")

# Test stats
print("\n4. Getting project statistics...")
start = time.time()
stats = manager.get_stats("test_user", "test_project")
print(f"   ✓ Stats retrieved in {time.time() - start:.3f}s")
print(f"   - Total chunks: {stats['total_chunks']}")
print(f"   - Backend: {stats['backend']}")

# Clean up
print("\n5. Cleaning up test data...")
start = time.time()
manager.clear_project("test_user", "test_project")
print(f"   ✓ Cleaned up in {time.time() - start:.2f}s")

# Summary
print("\n" + "=" * 50)
print("Performance Summary:")
# Use indexing_stats from earlier indexing operation
print(f"- Indexed chunks: {indexing_stats['chunks']}")
print(f"- Indexing rate: {indexing_stats['chunks']/indexing_elapsed:.1f} chunks/sec")
print(f"- Average search latency: {avg_search_time*1000:.1f}ms")
print(f"- Total chunks in DB: {stats['total_chunks']}")
print(f"- Connection: {stats['status']}")

manager.close()
print("\nTest completed!")
