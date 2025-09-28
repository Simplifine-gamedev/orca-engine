/**************************************************************************/
/*  node_pattern_utils.h                                                  */
/**************************************************************************/

#pragma once

#include "core/variant/array.h"
#include "core/string/ustring.h"

class Node;

class NodePatternUtils {
public:
    // Find nodes matching a pattern like "Hallway//Column*/Collision"
    static Array find_nodes_by_pattern(const String &p_pattern);
    
    // Check if a node path matches a pattern with wildcards
    static bool path_matches_pattern(const String &p_path, const String &p_pattern);
    
private:
    static void _find_nodes_recursive(Node *p_node, const String &p_pattern, const String &p_current_path, Array &p_results);
};
