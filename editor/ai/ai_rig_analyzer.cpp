/**************************************************************************/
/*  ai_rig_analyzer.cpp                                                   */
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

#include "ai_rig_analyzer.h"

#include "core/io/json.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/3d/importer_mesh.h"

void AIRigAnalyzer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("extract_skeleton_data", "skeleton"), &AIRigAnalyzer::extract_skeleton_data);
	ClassDB::bind_method(D_METHOD("extract_skin_weights", "mesh"), &AIRigAnalyzer::extract_skin_weights);
	ClassDB::bind_method(D_METHOD("apply_standardized_skeleton", "skeleton", "standardized_data"), &AIRigAnalyzer::apply_standardized_skeleton);
	ClassDB::bind_method(D_METHOD("create_bone_mapping_visualization", "mapping"), &AIRigAnalyzer::create_bone_mapping_visualization);
	ClassDB::bind_method(D_METHOD("validate_skeleton_structure", "skeleton"), &AIRigAnalyzer::validate_skeleton_structure);
	ClassDB::bind_method(D_METHOD("find_skeletons_in_scene", "root"), &AIRigAnalyzer::find_skeletons_in_scene);
	ClassDB::bind_method(D_METHOD("get_standard_humanoid_bones"), &AIRigAnalyzer::get_standard_humanoid_bones);
}

Dictionary AIRigAnalyzer::extract_skeleton_data(Skeleton3D *p_skeleton) {
	Dictionary data;
	
	if (!p_skeleton) {
		return data;
	}
	
	Array bones;
	int bone_count = p_skeleton->get_bone_count();
	
	for (int i = 0; i < bone_count; i++) {
		Dictionary bone_data;
		
		// Basic bone info
		bone_data["name"] = p_skeleton->get_bone_name(i);
		bone_data["parent"] = p_skeleton->get_bone_parent(i);
		
		// Transform data
		Transform3D rest = p_skeleton->get_bone_rest(i);
		bone_data["rest_position"] = rest.origin;
		bone_data["rest_rotation"] = rest.basis.get_rotation_quaternion();
		bone_data["rest_scale"] = rest.basis.get_scale();
		
		Transform3D pose = p_skeleton->get_bone_pose(i);
		bone_data["pose_position"] = pose.origin;
		bone_data["pose_rotation"] = pose.basis.get_rotation_quaternion();
		bone_data["pose_scale"] = pose.basis.get_scale();
		
		// Global transforms
		Transform3D global_rest = p_skeleton->get_bone_global_rest(i);
		bone_data["global_rest_position"] = global_rest.origin;
		bone_data["global_rest_rotation"] = global_rest.basis.get_rotation_quaternion();
		
		Transform3D global_pose = p_skeleton->get_bone_global_pose(i);
		bone_data["global_pose_position"] = global_pose.origin;
		bone_data["global_pose_rotation"] = global_pose.basis.get_rotation_quaternion();
		
		// Additional properties
		bone_data["enabled"] = p_skeleton->is_bone_enabled(i);
		
		// Children
		Array children;
		Vector<int> bone_children = p_skeleton->get_bone_children(i);
		for (int j = 0; j < bone_children.size(); j++) {
			children.push_back(bone_children[j]);
		}
		bone_data["children"] = children;
		
		bones.push_back(bone_data);
	}
	
	data["bones"] = bones;
	data["bone_count"] = bone_count;
	data["motion_scale"] = p_skeleton->get_motion_scale();
	data["show_rest_only"] = p_skeleton->is_show_rest_only();
	
	// Add concatenated bone names for quick analysis
	data["concatenated_bone_names"] = p_skeleton->get_concatenated_bone_names();
	
	return data;
}

Dictionary AIRigAnalyzer::extract_skin_weights(Ref<ImporterMesh> p_mesh) {
	Dictionary skin_data;
	
	if (p_mesh.is_null()) {
		return skin_data;
	}
	
	Array surfaces_data;
	
	for (int surface_i = 0; surface_i < p_mesh->get_surface_count(); surface_i++) {
		Dictionary surface_data;
		Array mesh_arrays = p_mesh->get_surface_arrays(surface_i);
		
		if (mesh_arrays.size() > Mesh::ARRAY_BONES && mesh_arrays.size() > Mesh::ARRAY_WEIGHTS) {
			PackedInt32Array bones = mesh_arrays[Mesh::ARRAY_BONES];
			PackedFloat32Array weights = mesh_arrays[Mesh::ARRAY_WEIGHTS];
			
			if (bones.size() > 0 && weights.size() > 0) {
				surface_data["bones"] = bones;
				surface_data["weights"] = weights;
				surface_data["vertex_count"] = bones.size() / 4; // Assuming 4 bones per vertex
				
				// Analyze for zero-weight vertices
				int zero_weight_count = 0;
				for (int v = 0; v < bones.size(); v += 4) {
					float total_weight = 0.0f;
					for (int w = 0; w < 4; w++) {
						if (v + w < weights.size()) {
							total_weight += weights[v + w];
						}
					}
					if (total_weight == 0.0f) {
						zero_weight_count++;
					}
				}
				surface_data["zero_weight_vertices"] = zero_weight_count;
			}
		}
		
		surfaces_data.push_back(surface_data);
	}
	
	skin_data["surfaces"] = surfaces_data;
	skin_data["surface_count"] = p_mesh->get_surface_count();
	
	return skin_data;
}

Error AIRigAnalyzer::apply_standardized_skeleton(Skeleton3D *p_skeleton, const Dictionary &p_standardized_data) {
	if (!p_skeleton) {
		return ERR_INVALID_PARAMETER;
	}
	
	if (!p_standardized_data.has("bones")) {
		return ERR_INVALID_DATA;
	}
	
	Array bones = p_standardized_data["bones"];
	
	// Clear existing skeleton
	p_skeleton->clear_bones();
	
	// First pass: Add all bones
	for (int i = 0; i < bones.size(); i++) {
		Dictionary bone_data = bones[i];
		if (!bone_data.has("name")) {
			continue;
		}
		
		String bone_name = bone_data["name"];
		p_skeleton->add_bone(bone_name);
		
		// Set rest transform if available
		if (bone_data.has("rest_position") && bone_data.has("rest_rotation") && bone_data.has("rest_scale")) {
			Transform3D rest;
			rest.origin = bone_data["rest_position"];
			rest.basis.set_quaternion_scale(bone_data["rest_rotation"], bone_data["rest_scale"]);
			p_skeleton->set_bone_rest(i, rest);
		}
		
		// Set pose if available
		if (bone_data.has("pose_position") && bone_data.has("pose_rotation") && bone_data.has("pose_scale")) {
			p_skeleton->set_bone_pose_position(i, bone_data["pose_position"]);
			p_skeleton->set_bone_pose_rotation(i, bone_data["pose_rotation"]);
			p_skeleton->set_bone_pose_scale(i, bone_data["pose_scale"]);
		}
		
		// Set enabled state
		if (bone_data.has("enabled")) {
			p_skeleton->set_bone_enabled(i, bone_data["enabled"]);
		}
	}
	
	// Second pass: Set parent relationships
	for (int i = 0; i < bones.size(); i++) {
		Dictionary bone_data = bones[i];
		if (bone_data.has("parent")) {
			int parent_idx = bone_data["parent"];
			if (parent_idx >= 0 && parent_idx < bones.size()) {
				p_skeleton->set_bone_parent(i, parent_idx);
			}
		}
	}
	
	// Set additional properties
	if (p_standardized_data.has("motion_scale")) {
		p_skeleton->set_motion_scale(p_standardized_data["motion_scale"]);
	}
	
	if (p_standardized_data.has("show_rest_only")) {
		p_skeleton->set_show_rest_only(p_standardized_data["show_rest_only"]);
	}
	
	return OK;
}

Dictionary AIRigAnalyzer::create_bone_mapping_visualization(const Dictionary &p_mapping) {
	Dictionary viz_data;
	
	Array mappings;
	Array keys = p_mapping.keys();
	
	for (int i = 0; i < keys.size(); i++) {
		String original_name = keys[i];
		String standard_name = p_mapping[original_name];
		
		Dictionary mapping_pair;
		mapping_pair["original"] = original_name;
		mapping_pair["standard"] = standard_name;
		mapping_pair["confidence"] = 1.0; // Could be enhanced with actual confidence scores
		
		mappings.push_back(mapping_pair);
	}
	
	viz_data["mappings"] = mappings;
	viz_data["total_mappings"] = mappings.size();
	
	return viz_data;
}

Array AIRigAnalyzer::validate_skeleton_structure(Skeleton3D *p_skeleton) {
	Array issues;
	
	if (!p_skeleton) {
		issues.push_back("No skeleton provided");
		return issues;
	}
	
	int bone_count = p_skeleton->get_bone_count();
	
	if (bone_count == 0) {
		issues.push_back("Skeleton has no bones");
		return issues;
	}
	
	// Check for circular references
	for (int i = 0; i < bone_count; i++) {
		HashSet<int> visited;
		int current = i;
		
		while (current != -1) {
			if (visited.has(current)) {
				issues.push_back(String("Circular reference detected in bone hierarchy starting at: ") + p_skeleton->get_bone_name(i));
				break;
			}
			visited.insert(current);
			current = p_skeleton->get_bone_parent(current);
		}
	}
	
	// Check for orphaned bones (multiple roots)
	Array roots;
	for (int i = 0; i < bone_count; i++) {
		if (p_skeleton->get_bone_parent(i) == -1) {
			roots.push_back(i);
		}
	}
	
	if (roots.size() > 1) {
		String root_names = "";
		for (int i = 0; i < roots.size(); i++) {
			if (i > 0) root_names += ", ";
			root_names += p_skeleton->get_bone_name(roots[i]);
		}
		issues.push_back(String("Multiple root bones detected: ") + root_names);
	}
	
	// Check for disabled bones in hierarchy
	for (int i = 0; i < bone_count; i++) {
		if (!p_skeleton->is_bone_enabled(i)) {
			Vector<int> children = p_skeleton->get_bone_children(i);
			if (children.size() > 0) {
				issues.push_back(String("Disabled bone has children: ") + p_skeleton->get_bone_name(i));
			}
		}
	}
	
	return issues;
}

Array AIRigAnalyzer::find_skeletons_in_scene(Node *p_root) {
	Array skeletons;
	
	if (!p_root) {
		return skeletons;
	}
	
	// Check if current node is a skeleton
	Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(p_root);
	if (skeleton) {
		Dictionary skeleton_info;
		skeleton_info["skeleton"] = skeleton;
		skeleton_info["path"] = p_root->get_path();
		skeleton_info["name"] = p_root->get_name();
		skeleton_info["bone_count"] = skeleton->get_bone_count();
		skeletons.push_back(skeleton_info);
	}
	
	// Recursively check children
	for (int i = 0; i < p_root->get_child_count(); i++) {
		Array child_skeletons = find_skeletons_in_scene(p_root->get_child(i));
		for (int j = 0; j < child_skeletons.size(); j++) {
			skeletons.push_back(child_skeletons[j]);
		}
	}
	
	return skeletons;
}

Array AIRigAnalyzer::get_standard_humanoid_bones() {
	Array standard_bones;
	
	// Standard humanoid skeleton structure
	standard_bones.push_back("Root");
	standard_bones.push_back("Hips");
	standard_bones.push_back("Spine");
	standard_bones.push_back("Spine1");
	standard_bones.push_back("Spine2");
	standard_bones.push_back("Neck");
	standard_bones.push_back("Head");
	
	// Arms
	standard_bones.push_back("LeftShoulder");
	standard_bones.push_back("LeftArm");
	standard_bones.push_back("LeftForeArm");
	standard_bones.push_back("LeftHand");
	
	standard_bones.push_back("RightShoulder");
	standard_bones.push_back("RightArm");
	standard_bones.push_back("RightForeArm");
	standard_bones.push_back("RightHand");
	
	// Legs
	standard_bones.push_back("LeftUpLeg");
	standard_bones.push_back("LeftLeg");
	standard_bones.push_back("LeftFoot");
	
	standard_bones.push_back("RightUpLeg");
	standard_bones.push_back("RightLeg");
	standard_bones.push_back("RightFoot");
	
	return standard_bones;
}

AIRigAnalyzer::AIRigAnalyzer() {
}

AIRigAnalyzer::~AIRigAnalyzer() {
}
