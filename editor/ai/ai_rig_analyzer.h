/**************************************************************************/
/*  ai_rig_analyzer.h                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef AI_RIG_ANALYZER_H
#define AI_RIG_ANALYZER_H

#include "core/object/ref_counted.h"
#include "core/variant/dictionary.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/3d/importer_mesh.h"

/**
 * AI-powered rig analyzer for 3D models.
 * Extracts skeleton data and interfaces with backend AI services
 * to analyze and standardize rig structures.
 */
class AIRigAnalyzer : public RefCounted {
	GDCLASS(AIRigAnalyzer, RefCounted);

private:
	static void _bind_methods();

public:
	/**
	 * Extract skeleton data from a Skeleton3D node into a format 
	 * suitable for AI analysis
	 */
	Dictionary extract_skeleton_data(Skeleton3D *p_skeleton);
	
	/**
	 * Extract skin weight data from an ImporterMesh for analysis
	 */
	Dictionary extract_skin_weights(Ref<ImporterMesh> p_mesh);
	
	/**
	 * Apply standardized skeleton data back to a Skeleton3D node
	 */
	Error apply_standardized_skeleton(Skeleton3D *p_skeleton, const Dictionary &p_standardized_data);
	
	/**
	 * Create bone mapping visualization for editor display
	 */
	Dictionary create_bone_mapping_visualization(const Dictionary &p_mapping);
	
	/**
	 * Validate skeleton structure for common issues
	 */
	Array validate_skeleton_structure(Skeleton3D *p_skeleton);
	
	/**
	 * Find skeleton nodes in a scene tree
	 */
	Array find_skeletons_in_scene(Node *p_root);
	
	/**
	 * Get standard humanoid bone names for comparison
	 */
	Array get_standard_humanoid_bones();

	AIRigAnalyzer();
	~AIRigAnalyzer();
};

#endif // AI_RIG_ANALYZER_H
