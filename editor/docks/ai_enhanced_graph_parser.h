/*
 * © 2025 Simplifine Corp.
 * Enhanced Godot Graph Parser for Frontend
 * Personal Non-Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#ifndef AI_ENHANCED_GRAPH_PARSER_H
#define AI_ENHANCED_GRAPH_PARSER_H

#include "core/object/ref_counted.h"
#include "core/variant/dictionary.h"
#include "core/variant/array.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"

/**
 * AIEnhancedGraphParser
 * 
 * Parses Godot project files to build a comprehensive relationship graph including:
 * - Signal flows and connections across files
 * - Scene-script relationships  
 * - Resource dependency chains
 * - Autoload/singleton tracking
 * - Export variable relationships
 * - Input action mappings
 * - Group memberships
 * 
 * This provides world-class context for AI agents working with Godot projects.
 */
class AIEnhancedGraphParser : public RefCounted {
	GDCLASS(AIEnhancedGraphParser, RefCounted);

private:
	String project_root;
	
	// Core graph data structures
	HashMap<String, Dictionary> nodes; // file_path -> node data
	Array connections; // Array of connection dictionaries
	HashMap<String, Array> signal_flows; // signal_name -> Array of flow data
	HashMap<String, String> autoloads; // name -> path
	HashMap<String, Dictionary> input_actions; // action_name -> action data
	HashMap<String, Array> scene_hierarchies; // scene_path -> node hierarchy
	HashMap<String, Array> resource_dependencies; // resource_path -> dependencies
	
	// Helper methods
	void _parse_gdscript_file(const String &p_file_path, const String &p_content);
	void _parse_scene_file(const String &p_file_path, const String &p_content);
	void _parse_project_settings(const String &p_content);
	Array _extract_signal_definitions(const String &p_content);
	Array _extract_signal_emissions(const String &p_content);
	Array _extract_function_definitions(const String &p_content);
	Array _extract_preloads(const String &p_content);
	Array _extract_extends(const String &p_content);
	Array _parse_scene_connections(const String &p_content);
	Dictionary _parse_scene_node(const String &p_content, int &p_offset);
	Array _extract_external_resources(const String &p_content);
	
	// WORLD-CLASS: Method call and node access tracking
	Array _extract_method_calls(const String &p_content);
	Array _extract_node_accesses(const String &p_content);
	Array _extract_property_modifications(const String &p_content);
	Array _extract_scene_instances(const String &p_content);
	Array _extract_export_variables(const String &p_content);
	Array _extract_group_memberships(const String &p_content);
	Array _extract_dynamic_scene_loads(const String &p_content);
	Array _extract_scene_groups(const String &p_content);
	Array _extract_export_assignments(const String &p_content, const Array &p_external_resources, const Array &p_script_attachments);
	
	// GAME-CHANGER: Method-level signal emission tracking for multi-hop tracing
	Array _extract_function_signal_mappings(const String &p_content, const Array &p_functions, const Array &p_signals_emitted);
	
	String _normalize_path(const String &p_path);
	
protected:
	static void _bind_methods();

public:
	AIEnhancedGraphParser();
	~AIEnhancedGraphParser();
	
	// Main parsing methods
	void set_project_root(const String &p_root);
	void parse_file(const String &p_file_path, const String &p_content);
	void parse_project_file(const String &p_content);
	
	// Graph retrieval methods
	Dictionary get_graph_data();
	Dictionary get_context_for_file(const String &p_file_path);
	Array get_signal_flows();
	Dictionary get_autoloads();
	Dictionary get_input_actions();
	
	// Context enrichment methods
	Dictionary enrich_file_context(const String &p_file_path);
	void clear();
};

#endif // AI_ENHANCED_GRAPH_PARSER_H

