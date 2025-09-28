/**************************************************************************/
/*  node_pattern_utils.cpp                                                */
/**************************************************************************/

#include "node_pattern_utils.h"
#include "scene/main/node.h"
#include "editor/editor_node.h"

Array NodePatternUtils::find_nodes_by_pattern(const String &p_pattern) {
    Array matching_nodes;
    Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
    if (!edited_scene) {
        return matching_nodes;
    }
    
    _find_nodes_recursive(edited_scene, p_pattern, "", matching_nodes);
    return matching_nodes;
}

bool NodePatternUtils::path_matches_pattern(const String &p_path, const String &p_pattern) {
    // Handle // recursive separator
    String pattern = p_pattern;
    String path = p_path;
    
    // If pattern has //, it means "anywhere in the hierarchy"
    if (pattern.find("//") != -1) {
        PackedStringArray pattern_parts = pattern.split("//");
        if (pattern_parts.size() == 2) {
            String prefix = pattern_parts[0];
            String suffix = pattern_parts[1];
            
            // Check if path starts with prefix and contains suffix pattern
            if (!prefix.is_empty() && !path.begins_with(prefix)) {
                return false;
            }
            
            // Check if the suffix pattern matches somewhere in the remaining path
            String remaining_path = prefix.is_empty() ? path : path.substr(prefix.length());
            return path_matches_pattern(remaining_path, suffix);
        }
    }
    
    // Handle * wildcards
    if (pattern.find("*") != -1) {
        PackedStringArray pattern_parts = pattern.split("*");
        int path_pos = 0;
        
        for (int i = 0; i < pattern_parts.size(); i++) {
            String part = pattern_parts[i];
            if (part.is_empty()) continue;
            
            int found_pos = path.find(part, path_pos);
            if (found_pos == -1) {
                return false;
            }
            
            // For first part, must match at current position
            if (i == 0 && found_pos != path_pos) {
                return false;
            }
            
            path_pos = found_pos + part.length();
        }
        
        // For last part, must match at end if pattern doesn't end with *
        if (!pattern.ends_with("*") && path_pos != path.length()) {
            String last_part = pattern_parts[pattern_parts.size() - 1];
            if (!last_part.is_empty() && !path.ends_with(last_part)) {
                return false;
            }
        }
        
        return true;
    }
    
    // Exact match
    return path == pattern;
}

void NodePatternUtils::_find_nodes_recursive(Node *p_node, const String &p_pattern, const String &p_current_path, Array &p_results) {
    if (!p_node) return;
    
    String node_name = p_node->get_name();
    String full_path = p_current_path.is_empty() ? node_name : p_current_path + "/" + node_name;
    
    // Check if this node matches the pattern
    if (path_matches_pattern(full_path, p_pattern)) {
        p_results.push_back(full_path);
    }
    
    // Recursively check children
    for (int i = 0; i < p_node->get_child_count(); i++) {
        Node *child = p_node->get_child(i);
        _find_nodes_recursive(child, p_pattern, full_path, p_results);
    }
}
