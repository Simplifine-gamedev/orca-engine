#!/usr/bin/env python3
"""
Demonstrate graph functionality with mock data
"""

from project_graph_manager import ProjectGraphManager
import json

def demo_graph_structure():
    """Demonstrate the graph structure with sample Godot project data"""
    print("🌐 Graph Structure Demo")
    print("=" * 50)
    
    # Create a sample project graph
    graph_manager = ProjectGraphManager("/mock/project", "demo_user")
    
    # Sample file contents representing a typical Godot project
    sample_files = {
        "/mock/project/main.tscn": '''[gd_scene load_steps=4 format=3 uid="uid://main"]

[ext_resource type="Script" path="res://scripts/player.gd" id="1"]
[ext_resource type="PackedScene" path="res://scenes/GameUI.tscn" id="2"]
[ext_resource type="Texture2D" path="res://textures/player_sprite.png" id="3"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

[node name="Player" type="CharacterBody2D" parent="."]
texture = ExtResource("3")

[node name="UI" parent="." instance=ExtResource("2")]''',

        "/mock/project/scripts/player.gd": '''extends CharacterBody2D

@export var speed: float = 300.0
@export var jump_velocity: float = -400.0

var ui_scene = preload("res://scenes/GameUI.tscn")
var item_class = preload("res://scripts/Item.gd")

signal health_changed(new_health)
signal player_died

func _ready():
    var ui = get_node("../UI")
    health_changed.connect(ui.update_health)

func _physics_process(delta):
    # Player movement logic
    pass

func change_scene():
    get_tree().change_scene_to_file("res://scenes/menu.tscn")''',

        "/mock/project/scenes/GameUI.tscn": '''[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://scripts/ui_controller.gd" id="1"]

[node name="GameUI" type="Control"]
script = ExtResource("1")

[node name="HealthBar" type="ProgressBar" parent="."]''',

        "/mock/project/scripts/ui_controller.gd": '''extends Control

func update_health(new_health):
    $HealthBar.value = new_health''',

        "/mock/project/scripts/Item.gd": '''extends Resource
class_name Item

@export var name: String
@export var description: String''',

        "/mock/project/scenes/menu.tscn": '''[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://scripts/menu_controller.gd" id="1"]

[node name="Menu" type="Control"]
script = ExtResource("1")''',

        "/mock/project/scripts/menu_controller.gd": '''extends Control

func _on_start_button_pressed():
    get_tree().change_scene_to_file("res://main.tscn")'''
    }
    
    # File metadata
    file_metadata = {}
    for file_path, content in sample_files.items():
        file_metadata[file_path] = {
            'last_modified': 1644000000,
            'size': len(content),
            'modality': 'text'
        }
    
    # Build the graph
    print("📊 Building sample project graph...")
    graph_manager.build_project_graph(sample_files, file_metadata)
    
    # Show graph summary
    summary = graph_manager.get_graph_summary()
    print(f"✅ Graph built successfully!")
    print(f"   📁 Nodes: {summary['nodes']}")
    print(f"   🔗 Edges: {summary['edges']}")
    print(f"   📊 Node types: {summary['node_types']}")
    print(f"   🔗 Relationship types: {summary['relationship_types']}")
    
    # Show central files
    print(f"\n⭐ Most Central Files:")
    central_files = summary.get('central_files', [])
    for i, file_info in enumerate(central_files):
        print(f"   {i+1}. {file_info['file']} (centrality: {file_info['centrality']:.3f})")
    
    # Show connections for main.tscn
    print(f"\n🔗 Connections for main.tscn:")
    connections = graph_manager.find_connected_files("/mock/project/main.tscn", max_depth=2)
    for relation_type, files in connections.items():
        if files:
            print(f"   {relation_type}:")
            for f in files:
                print(f"      • {f}")
    
    # Show connections for player.gd
    print(f"\n🔗 Connections for player.gd:")
    connections = graph_manager.find_connected_files("/mock/project/scripts/player.gd", max_depth=2)
    for relation_type, files in connections.items():
        if files:
            print(f"   {relation_type}:")
            for f in files:
                print(f"      • {f}")
    
    print(f"\n🎯 Graph Analysis Complete!")
    print(f"\n💡 The graph structure provides:")
    print(f"   🔍 Enhanced file search with relationship context")
    print(f"   📊 Project structure analysis and visualization")
    print(f"   🎯 Important file identification through centrality")
    print(f"   🔗 Dependency tracking and relationship mapping")
    print(f"   ⚡ Quick navigation through connected files")
    
    return graph_manager

if __name__ == "__main__":
    demo_graph_structure()