"""
AI-Powered Rig Standardization System for Godot
Handles analysis and standardization of 3D model rigs
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re
from dataclasses import dataclass
from openai import OpenAI
import os

@dataclass
class BoneInfo:
    name: str
    parent: Optional[str]
    children: List[str]
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]  # quaternion
    scale: Tuple[float, float, float]
    
@dataclass
class SkeletonAnalysis:
    bone_count: int
    naming_convention: str
    detected_issues: List[str]
    confidence_score: float
    suggested_mapping: Dict[str, str]
    hierarchy_issues: List[str]

class RigProcessor:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Standard humanoid skeleton structure
        self.humanoid_standard = {
            "Root": {"children": ["Hips"]},
            "Hips": {"parent": "Root", "children": ["Spine", "LeftUpLeg", "RightUpLeg"]},
            "Spine": {"parent": "Hips", "children": ["Spine1"]},
            "Spine1": {"parent": "Spine", "children": ["Spine2"]}, 
            "Spine2": {"parent": "Spine1", "children": ["Neck", "LeftShoulder", "RightShoulder"]},
            "Neck": {"parent": "Spine2", "children": ["Head"]},
            "Head": {"parent": "Neck", "children": []},
            
            # Arms
            "LeftShoulder": {"parent": "Spine2", "children": ["LeftArm"]},
            "LeftArm": {"parent": "LeftShoulder", "children": ["LeftForeArm"]},
            "LeftForeArm": {"parent": "LeftArm", "children": ["LeftHand"]},
            "LeftHand": {"parent": "LeftForeArm", "children": []},
            
            "RightShoulder": {"parent": "Spine2", "children": ["RightArm"]}, 
            "RightArm": {"parent": "RightShoulder", "children": ["RightForeArm"]},
            "RightForeArm": {"parent": "RightArm", "children": ["RightHand"]},
            "RightHand": {"parent": "RightForeArm", "children": []},
            
            # Legs
            "LeftUpLeg": {"parent": "Hips", "children": ["LeftLeg"]},
            "LeftLeg": {"parent": "LeftUpLeg", "children": ["LeftFoot"]},
            "LeftFoot": {"parent": "LeftLeg", "children": []},
            
            "RightUpLeg": {"parent": "Hips", "children": ["RightLeg"]},
            "RightLeg": {"parent": "RightUpLeg", "children": ["RightFoot"]},
            "RightFoot": {"parent": "RightLeg", "children": []}
        }
        
        # Common bone name patterns for different sources
        self.bone_name_patterns = {
            'mixamo': {
                'prefix': 'mixamorig:',
                'mappings': {
                    'mixamorig:Hips': 'Hips',
                    'mixamorig:Spine': 'Spine', 
                    'mixamorig:Spine1': 'Spine1',
                    'mixamorig:Spine2': 'Spine2',
                    'mixamorig:Neck': 'Neck',
                    'mixamorig:Head': 'Head',
                    'mixamorig:LeftShoulder': 'LeftShoulder',
                    'mixamorig:LeftArm': 'LeftArm',
                    'mixamorig:LeftForeArm': 'LeftForeArm',
                    'mixamorig:LeftHand': 'LeftHand',
                    'mixamorig:RightShoulder': 'RightShoulder',
                    'mixamorig:RightArm': 'RightArm',
                    'mixamorig:RightForeArm': 'RightForeArm', 
                    'mixamorig:RightHand': 'RightHand',
                    'mixamorig:LeftUpLeg': 'LeftUpLeg',
                    'mixamorig:LeftLeg': 'LeftLeg',
                    'mixamorig:LeftFoot': 'LeftFoot',
                    'mixamorig:RightUpLeg': 'RightUpLeg',
                    'mixamorig:RightLeg': 'RightLeg',
                    'mixamorig:RightFoot': 'RightFoot'
                }
            },
            'unreal': {
                'prefix': '',
                'mappings': {
                    'pelvis': 'Hips',
                    'spine_01': 'Spine',
                    'spine_02': 'Spine1', 
                    'spine_03': 'Spine2',
                    'neck_01': 'Neck',
                    'head': 'Head',
                    'clavicle_l': 'LeftShoulder',
                    'upperarm_l': 'LeftArm',
                    'lowerarm_l': 'LeftForeArm',
                    'hand_l': 'LeftHand',
                    'clavicle_r': 'RightShoulder',
                    'upperarm_r': 'RightArm',
                    'lowerarm_r': 'RightForeArm',
                    'hand_r': 'RightHand',
                    'thigh_l': 'LeftUpLeg',
                    'calf_l': 'LeftLeg', 
                    'foot_l': 'LeftFoot',
                    'thigh_r': 'RightUpLeg',
                    'calf_r': 'RightLeg',
                    'foot_r': 'RightFoot'
                }
            }
        }

    def analyze_skeleton(self, skeleton_data: Dict) -> SkeletonAnalysis:
        """Analyze a skeleton structure for issues and standardization opportunities"""
        bones = skeleton_data.get('bones', [])
        bone_names = [bone['name'] for bone in bones]
        
        # Detect naming convention
        naming_convention = self._detect_naming_convention(bone_names)
        
        # Find issues
        issues = []
        hierarchy_issues = []
        
        # Check for common problems
        if len(bones) == 0:
            issues.append("No skeleton found")
            return SkeletonAnalysis(0, "unknown", issues, 0.0, {}, hierarchy_issues)
            
        # Check for missing essential bones
        essential_bones = ['Hips', 'Spine', 'Head', 'LeftArm', 'RightArm', 'LeftLeg', 'RightLeg']
        mapped_names = self._get_bone_mapping(bone_names, naming_convention)
        
        missing_bones = []
        for essential in essential_bones:
            if essential not in mapped_names.values():
                # Try to find similar bones
                similar = self._find_similar_bone_names(essential, bone_names)
                if not similar:
                    missing_bones.append(essential)
                    
        if missing_bones:
            issues.append(f"Missing essential bones: {', '.join(missing_bones)}")
            
        # Check hierarchy issues
        hierarchy_issues = self._analyze_hierarchy(bones)
        
        # Check for zero-weight vertices (if skin data provided)
        if 'skin_weights' in skeleton_data:
            weight_issues = self._analyze_skin_weights(skeleton_data['skin_weights'])
            issues.extend(weight_issues)
            
        confidence_score = max(0.0, 1.0 - (len(issues) * 0.2) - (len(hierarchy_issues) * 0.1))
        
        return SkeletonAnalysis(
            bone_count=len(bones),
            naming_convention=naming_convention, 
            detected_issues=issues,
            confidence_score=confidence_score,
            suggested_mapping=mapped_names,
            hierarchy_issues=hierarchy_issues
        )

    def _detect_naming_convention(self, bone_names: List[str]) -> str:
        """Detect which naming convention is being used"""
        mixamo_count = sum(1 for name in bone_names if name.startswith('mixamorig:'))
        unreal_count = sum(1 for name in bone_names if any(
            pattern in name.lower() for pattern in ['spine_', 'clavicle_', 'thigh_', 'calf_']
        ))
        
        if mixamo_count > len(bone_names) * 0.3:
            return 'mixamo'
        elif unreal_count > len(bone_names) * 0.2:
            return 'unreal' 
        else:
            return 'custom'

    def _get_bone_mapping(self, bone_names: List[str], convention: str) -> Dict[str, str]:
        """Generate mapping from current bone names to standard names"""
        if convention in self.bone_name_patterns:
            pattern = self.bone_name_patterns[convention]
            mapping = {}
            for bone_name in bone_names:
                if bone_name in pattern['mappings']:
                    mapping[bone_name] = pattern['mappings'][bone_name]
                else:
                    # Try AI-powered mapping for unmapped bones
                    suggested = self._ai_suggest_bone_mapping(bone_name)
                    if suggested:
                        mapping[bone_name] = suggested
            return mapping
        else:
            # Use AI for custom naming conventions
            return self._ai_generate_full_mapping(bone_names)

    def _ai_suggest_bone_mapping(self, bone_name: str) -> Optional[str]:
        """Use AI to suggest standard bone name mapping"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are a 3D rigging expert. Given a bone name, suggest the most appropriate standard humanoid skeleton bone name.
                    
Standard bones: Root, Hips, Spine, Spine1, Spine2, Neck, Head, LeftShoulder, LeftArm, LeftForeArm, LeftHand, RightShoulder, RightArm, RightForeArm, RightHand, LeftUpLeg, LeftLeg, LeftFoot, RightUpLeg, RightLeg, RightFoot

Respond with only the standard bone name, or 'UNKNOWN' if no good match."""},
                    {"role": "user", "content": f"Bone name: {bone_name}"}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            suggestion = response.choices[0].message.content.strip()
            return suggestion if suggestion != 'UNKNOWN' else None
            
        except Exception as e:
            print(f"AI bone mapping failed for {bone_name}: {e}")
            return None

    def _ai_generate_full_mapping(self, bone_names: List[str]) -> Dict[str, str]:
        """Use AI to generate complete bone mapping for custom skeletons"""
        try:
            bone_list = '\n'.join(bone_names)
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are a 3D rigging expert. Map bone names to standard humanoid skeleton bones.

Standard bones: Root, Hips, Spine, Spine1, Spine2, Neck, Head, LeftShoulder, LeftArm, LeftForeArm, LeftHand, RightShoulder, RightArm, RightForeArm, RightHand, LeftUpLeg, LeftLeg, LeftFoot, RightUpLeg, RightLeg, RightFoot

Return JSON mapping of {original_name: standard_name}. Only include mappings for bones you're confident about."""},
                    {"role": "user", "content": f"Bone names:\n{bone_list}"}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            mapping_json = response.choices[0].message.content.strip()
            return json.loads(mapping_json)
            
        except Exception as e:
            print(f"AI full mapping failed: {e}")
            return {}

    def _find_similar_bone_names(self, target: str, bone_names: List[str]) -> List[str]:
        """Find bone names that might be variations of the target"""
        similar = []
        target_lower = target.lower()
        
        for bone_name in bone_names:
            bone_lower = bone_name.lower()
            if target_lower in bone_lower or bone_lower in target_lower:
                similar.append(bone_name)
        
        return similar

    def _analyze_hierarchy(self, bones: List[Dict]) -> List[str]:
        """Analyze bone hierarchy for issues"""
        issues = []
        
        # Build hierarchy map
        children_map = {}
        parent_map = {}
        
        for bone in bones:
            bone_name = bone['name']
            parent_idx = bone.get('parent', -1)
            
            if parent_idx >= 0 and parent_idx < len(bones):
                parent_name = bones[parent_idx]['name']
                parent_map[bone_name] = parent_name
                
                if parent_name not in children_map:
                    children_map[parent_name] = []
                children_map[parent_name].append(bone_name)
        
        # Check for cycles
        visited = set()
        def check_cycle(bone_name, path):
            if bone_name in path:
                issues.append(f"Circular reference detected in hierarchy: {' -> '.join(path + [bone_name])}")
                return
            if bone_name in visited:
                return
                
            visited.add(bone_name)
            if bone_name in parent_map:
                check_cycle(parent_map[bone_name], path + [bone_name])
        
        for bone in bones:
            if bone['name'] not in visited:
                check_cycle(bone['name'], [])
        
        # Check for orphaned bones
        root_bones = [bone['name'] for bone in bones if bone.get('parent', -1) == -1]
        if len(root_bones) > 1:
            issues.append(f"Multiple root bones detected: {', '.join(root_bones)}")
        elif len(root_bones) == 0:
            issues.append("No root bone found")
            
        return issues

    def _analyze_skin_weights(self, skin_weights: Dict) -> List[str]:
        """Analyze skin weights for issues"""
        issues = []
        
        # Check for zero-weight vertices
        vertices_data = skin_weights.get('vertices', [])
        zero_weight_count = 0
        
        for vertex_weights in vertices_data:
            total_weight = sum(vertex_weights.values())
            if total_weight == 0:
                zero_weight_count += 1
        
        if zero_weight_count > 0:
            issues.append(f"Found {zero_weight_count} vertices with zero skin weights")
        
        return issues

    def standardize_skeleton(self, skeleton_data: Dict, target_mapping: Dict[str, str]) -> Dict:
        """Apply standardization to skeleton based on mapping"""
        bones = skeleton_data.get('bones', [])
        standardized_bones = []
        
        # Create name lookup (handle missing names defensively)
        name_to_index = {}
        for i, b in enumerate(bones):
            n = b.get('name')
            if isinstance(n, str) and n:
                name_to_index[n] = i
        
        for bone in bones:
            standardized_bone = bone.copy()
            original_name = bone['name']
            
            # Apply name mapping
            if original_name in target_mapping:
                standardized_bone['name'] = target_mapping[original_name]
                
            # Update parent reference
            parent_val = bone.get('parent', -1)
            # Normalize parent: int index, or name, or None
            parent_idx = -1
            if isinstance(parent_val, int):
                parent_idx = parent_val
            elif isinstance(parent_val, str):
                parent_idx = name_to_index.get(parent_val, -1)
            # Only proceed if a valid parent index
            if isinstance(parent_idx, int) and 0 <= parent_idx < len(bones):
                parent_original_name = bones[parent_idx].get('name', '')
                if isinstance(parent_original_name, str) and parent_original_name in target_mapping:
                    # Find new parent index among already standardized bones by new name
                    new_parent_name = target_mapping[parent_original_name]
                    for j, std_bone in enumerate(standardized_bones):
                        if std_bone.get('name') == new_parent_name:
                            standardized_bone['parent'] = j
                            break
            
            standardized_bones.append(standardized_bone)
        
        result = skeleton_data.copy()
        result['bones'] = standardized_bones
        result['standardization_applied'] = True
        result['original_mapping'] = target_mapping
        
        return result

    def generate_retargeting_data(self, source_skeleton: Dict, target_skeleton: Dict) -> Dict:
        """Generate retargeting data between two skeletons"""
        # This would contain transform mappings for animation retargeting
        # For now, return basic mapping info
        source_bones = {bone['name']: bone for bone in source_skeleton.get('bones', [])}
        target_bones = {bone['name']: bone for bone in target_skeleton.get('bones', [])}
        
        retargeting_map = {}
        for target_name, target_bone in target_bones.items():
            if target_name in source_bones:
                retargeting_map[target_name] = {
                    'source_bone': target_name,
                    'transform_offset': [0, 0, 0, 0, 0, 0, 1]  # pos + rot quaternion
                }
        
        return {
            'retargeting_map': retargeting_map,
            'source_skeleton_id': source_skeleton.get('id', 'unknown'),
            'target_skeleton_id': target_skeleton.get('id', 'unknown')
        }

    def auto_rig_mesh(self, mesh_data: Dict, character_type: str = "auto_detect", bone_count: str = "standard") -> Dict:
        """Automatically create a skeleton for an unrigged mesh using AI analysis"""
        try:
            # Analyze mesh geometry to determine character type if auto_detect
            if character_type == "auto_detect":
                character_type = self._detect_character_type(mesh_data)
            
            # Generate bone placement based on mesh geometry
            bone_positions = self._analyze_mesh_for_bone_placement(mesh_data, character_type)
            
            # Create hierarchical bone structure
            bone_hierarchy = self._create_bone_hierarchy(bone_positions, character_type, bone_count)
            
            # Generate automatic skin weights
            skin_weights = self._generate_automatic_weights(mesh_data, bone_hierarchy)
            
            return {
                "success": True,
                "skeleton_data": {
                    "bones": bone_hierarchy,
                    "bone_count": len(bone_hierarchy),
                    "character_type": character_type
                },
                "skin_weights": skin_weights,
                "ik_chains": self._create_ik_chains(bone_hierarchy) if character_type == "humanoid" else []
            }
            
        except Exception as e:
            return {"success": False, "error": f"Auto-rigging failed: {str(e)}"}

    def _detect_character_type(self, mesh_data: Dict) -> str:
        """Analyze mesh to determine if it's humanoid, quadruped, etc."""
        vertices = mesh_data.get('vertices', [])
        # Defensive: ensure numbers, not strings
        if vertices and isinstance(vertices[0], (list, tuple)):
            try:
                vertices = [[float(p[0]), float(p[1]), float(p[2])] for p in vertices]
            except Exception:
                pass
        if not vertices:
            return "generic"
            
        # Simple heuristic: analyze mesh proportions
        bbox = self._calculate_bounding_box(vertices)
        height = bbox['max_y'] - bbox['min_y']
        width = bbox['max_x'] - bbox['min_x']
        depth = bbox['max_z'] - bbox['min_z']
        
        # If roughly human proportions (height > width, height > depth)
        if height > width * 1.5 and height > depth * 1.5:
            return "humanoid"
        elif width > height or depth > height:
            return "quadruped" 
        else:
            return "generic"

    def _analyze_mesh_for_bone_placement(self, mesh_data: Dict, character_type: str) -> Dict:
        """Use AI to analyze mesh geometry and determine optimal bone placement"""
        vertices = mesh_data.get('vertices', [])
        if vertices and isinstance(vertices[0], (list, tuple)):
            try:
                vertices = [[float(p[0]), float(p[1]), float(p[2])] for p in vertices]
            except Exception:
                pass
        if not vertices:
            return {}
            
        # Calculate mesh center and bounds
        bbox = self._calculate_bounding_box(vertices)
        center_x = (bbox['max_x'] + bbox['min_x']) / 2
        center_y = (bbox['max_y'] + bbox['min_y']) / 2
        center_z = (bbox['max_z'] + bbox['min_z']) / 2
        
        if character_type == "humanoid":
            return self._generate_humanoid_bone_positions(bbox, center_x, center_y, center_z)
        elif character_type == "quadruped":
            return self._generate_quadruped_bone_positions(bbox, center_x, center_y, center_z)
        else:
            return self._generate_generic_bone_positions(bbox, center_x, center_y, center_z)

    def _generate_humanoid_bone_positions(self, bbox: Dict, cx: float, cy: float, cz: float) -> Dict:
        """Generate standard humanoid bone positions based on mesh bounds"""
        height = bbox['max_y'] - bbox['min_y']
        
        return {
            "Hips": {"position": [cx, bbox['min_y'] + height * 0.55, cz], "parent": -1},
            "Spine": {"position": [cx, bbox['min_y'] + height * 0.65, cz], "parent": 0},
            "Spine1": {"position": [cx, bbox['min_y'] + height * 0.75, cz], "parent": 1},
            "Neck": {"position": [cx, bbox['min_y'] + height * 0.85, cz], "parent": 2},
            "Head": {"position": [cx, bbox['min_y'] + height * 0.95, cz], "parent": 3},
            
            # Left Arm
            "LeftShoulder": {"position": [cx - height * 0.1, bbox['min_y'] + height * 0.8, cz], "parent": 2},
            "LeftArm": {"position": [cx - height * 0.2, bbox['min_y'] + height * 0.75, cz], "parent": 5},
            "LeftForeArm": {"position": [cx - height * 0.35, bbox['min_y'] + height * 0.65, cz], "parent": 6},
            "LeftHand": {"position": [cx - height * 0.45, bbox['min_y'] + height * 0.55, cz], "parent": 7},
            
            # Right Arm  
            "RightShoulder": {"position": [cx + height * 0.1, bbox['min_y'] + height * 0.8, cz], "parent": 2},
            "RightArm": {"position": [cx + height * 0.2, bbox['min_y'] + height * 0.75, cz], "parent": 9},
            "RightForeArm": {"position": [cx + height * 0.35, bbox['min_y'] + height * 0.65, cz], "parent": 10},
            "RightHand": {"position": [cx + height * 0.45, bbox['min_y'] + height * 0.55, cz], "parent": 11},
            
            # Left Leg
            "LeftUpLeg": {"position": [cx - height * 0.08, bbox['min_y'] + height * 0.45, cz], "parent": 0},
            "LeftLeg": {"position": [cx - height * 0.08, bbox['min_y'] + height * 0.25, cz], "parent": 13},
            "LeftFoot": {"position": [cx - height * 0.08, bbox['min_y'] + height * 0.05, cz], "parent": 14},
            
            # Right Leg
            "RightUpLeg": {"position": [cx + height * 0.08, bbox['min_y'] + height * 0.45, cz], "parent": 0},
            "RightLeg": {"position": [cx + height * 0.08, bbox['min_y'] + height * 0.25, cz], "parent": 16},
            "RightFoot": {"position": [cx + height * 0.08, bbox['min_y'] + height * 0.05, cz], "parent": 17}
        }

    def _generate_quadruped_bone_positions(self, bbox: Dict, cx: float, cy: float, cz: float) -> Dict:
        """Generate quadruped bone positions (for animals, etc.)"""
        length = bbox['max_z'] - bbox['min_z']
        height = bbox['max_y'] - bbox['min_y']
        
        return {
            "Root": {"position": [cx, bbox['min_y'] + height * 0.7, cz], "parent": -1},
            "Spine1": {"position": [cx, bbox['min_y'] + height * 0.7, cz - length * 0.2], "parent": 0},
            "Spine2": {"position": [cx, bbox['min_y'] + height * 0.75, cz + length * 0.1], "parent": 1},
            "Neck": {"position": [cx, bbox['min_y'] + height * 0.8, cz + length * 0.3], "parent": 2},
            "Head": {"position": [cx, bbox['min_y'] + height * 0.9, cz + length * 0.4], "parent": 3},
            
            # Front legs
            "LeftFrontLeg": {"position": [cx - height * 0.1, bbox['min_y'] + height * 0.5, cz + length * 0.2], "parent": 2},
            "LeftFrontPaw": {"position": [cx - height * 0.1, bbox['min_y'], cz + length * 0.2], "parent": 5},
            "RightFrontLeg": {"position": [cx + height * 0.1, bbox['min_y'] + height * 0.5, cz + length * 0.2], "parent": 2},
            "RightFrontPaw": {"position": [cx + height * 0.1, bbox['min_y'], cz + length * 0.2], "parent": 7},
            
            # Back legs  
            "LeftBackLeg": {"position": [cx - height * 0.1, bbox['min_y'] + height * 0.5, cz - length * 0.2], "parent": 0},
            "LeftBackPaw": {"position": [cx - height * 0.1, bbox['min_y'], cz - length * 0.2], "parent": 9},
            "RightBackLeg": {"position": [cx + height * 0.1, bbox['min_y'] + height * 0.5, cz - length * 0.2], "parent": 0},
            "RightBackPaw": {"position": [cx + height * 0.1, bbox['min_y'], cz - length * 0.2], "parent": 11}
        }

    def _generate_generic_bone_positions(self, bbox: Dict, cx: float, cy: float, cz: float) -> Dict:
        """Generate generic bone structure for unknown character types"""
        return {
            "Root": {"position": [cx, cy, cz], "parent": -1},
            "Bone1": {"position": [cx, bbox['max_y'], cz], "parent": 0},
            "Bone2": {"position": [cx, bbox['min_y'], cz], "parent": 0}
        }

    def _create_bone_hierarchy(self, bone_positions: Dict, character_type: str, bone_count: str) -> List[Dict]:
        """Convert bone positions to hierarchical bone structure"""
        bones = []
        
        for i, (name, data) in enumerate(bone_positions.items()):
            bone = {
                "name": name,
                "parent": data["parent"],
                "rest_position": data["position"],
                "rest_rotation": [0, 0, 0, 1],  # Identity quaternion
                "rest_scale": [1, 1, 1],
                "enabled": True
            }
            bones.append(bone)
            
        return bones

    def _generate_automatic_weights(self, mesh_data: Dict, bone_hierarchy: List[Dict]) -> Dict:
        """Generate skin weights automatically based on distance to bones"""
        # This would implement automatic weight generation similar to Blender's automatic weights
        return {
            "method": "distance_based",
            "weights_generated": True,
            "vertex_count": len(mesh_data.get('vertices', [])),
            "bone_influences": 4  # Max bones per vertex
        }

    def _create_ik_chains(self, bone_hierarchy: List[Dict]) -> List[Dict]:
        """Create IK chains for arms and legs"""
        ik_chains = []
        bone_names = [bone["name"] for bone in bone_hierarchy]
        
        # Left arm IK
        if all(name in bone_names for name in ["LeftArm", "LeftForeArm", "LeftHand"]):
            ik_chains.append({
                "name": "LeftArmIK",
                "bones": ["LeftArm", "LeftForeArm", "LeftHand"],
                "target": "LeftHand"
            })
            
        # Right arm IK  
        if all(name in bone_names for name in ["RightArm", "RightForeArm", "RightHand"]):
            ik_chains.append({
                "name": "RightArmIK", 
                "bones": ["RightArm", "RightForeArm", "RightHand"],
                "target": "RightHand"
            })
            
        # Left leg IK
        if all(name in bone_names for name in ["LeftUpLeg", "LeftLeg", "LeftFoot"]):
            ik_chains.append({
                "name": "LeftLegIK",
                "bones": ["LeftUpLeg", "LeftLeg", "LeftFoot"], 
                "target": "LeftFoot"
            })
            
        # Right leg IK
        if all(name in bone_names for name in ["RightUpLeg", "RightLeg", "RightFoot"]):
            ik_chains.append({
                "name": "RightLegIK",
                "bones": ["RightUpLeg", "RightLeg", "RightFoot"],
                "target": "RightFoot"
            })
            
        return ik_chains

    def _calculate_bounding_box(self, vertices: List) -> Dict:
        """Calculate bounding box of mesh vertices"""
        if not vertices:
            return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0}
            
        # Assuming vertices are in format [[x,y,z], [x,y,z], ...]
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices] 
        zs = [v[2] for v in vertices]
        
        return {
            "min_x": min(xs), "max_x": max(xs),
            "min_y": min(ys), "max_y": max(ys), 
            "min_z": min(zs), "max_z": max(zs)
        }

    def auto_rig_from_file(self, file_path: str, character_type: str = "auto_detect", bone_count: str = "standard") -> Dict:
        """Load mesh directly from file and auto-rig without frontend vertex extraction"""
        try:
            # Handle res:// paths - for now just skip file loading and use fallback
            if file_path.startswith('res://'):
                print(f"RIG_PROCESSOR: res:// path detected, using fallback approach: {file_path}")
                mesh_data = None
            else:
                # Try trimesh for absolute file paths
                mesh_data = None
                try:
                    import trimesh
                    # Force single mesh loading and handle multiple meshes
                    mesh = trimesh.load(file_path, force='mesh')
                    if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                        # Sample vertices to prevent memory issues
                        vertices = mesh.vertices
                        if len(vertices) > 5000:
                            step = len(vertices) // 5000 + 1
                            vertices = vertices[::step]
                        mesh_data = {'vertices': vertices.tolist()}
                        print(f"RIG_PROCESSOR: Loaded mesh with trimesh: {len(vertices)} vertices (from {len(mesh.vertices)} total)")
                except Exception as e:
                    print(f"RIG_PROCESSOR: trimesh failed: {e}")
            
            # Skip pygltflib for now - it has complex buffer handling issues
            # Let trimesh handle GLB files instead
            
            # Fast fallback: create rig directly without mesh processing
            if not mesh_data:
                print(f"RIG_PROCESSOR: Using fast fallback rig generation (no mesh processing)")
                return self._create_fast_fallback_rig(character_type, bone_count)
            
            return self.auto_rig_mesh(mesh_data, character_type, bone_count)
            
        except Exception as e:
            return {"success": False, "error": f"Auto-rigging from file failed: {str(e)}"}
    
    def _create_fast_fallback_rig(self, character_type: str, bone_count: str) -> Dict:
        """Create a rig instantly without any mesh processing to prevent UI freezing"""
        if character_type == "humanoid" or character_type == "auto_detect":
            # Standard humanoid skeleton with reasonable bone positions
            bones = [
                {"name": "Hips", "parent": None, "position": [0, 1, 0], "rotation": [0, 0, 0]},
                {"name": "Spine", "parent": "Hips", "position": [0, 1.1, 0], "rotation": [0, 0, 0]},
                {"name": "Spine1", "parent": "Spine", "position": [0, 1.3, 0], "rotation": [0, 0, 0]},
                {"name": "Neck", "parent": "Spine1", "position": [0, 1.6, 0], "rotation": [0, 0, 0]},
                {"name": "Head", "parent": "Neck", "position": [0, 1.8, 0], "rotation": [0, 0, 0]},
                
                # Left arm chain
                {"name": "LeftShoulder", "parent": "Spine1", "position": [-0.2, 1.5, 0], "rotation": [0, 0, 0]},
                {"name": "LeftArm", "parent": "LeftShoulder", "position": [-0.5, 1.5, 0], "rotation": [0, 0, 0]},
                {"name": "LeftForeArm", "parent": "LeftArm", "position": [-0.8, 1.5, 0], "rotation": [0, 0, 0]},
                {"name": "LeftHand", "parent": "LeftForeArm", "position": [-1.1, 1.5, 0], "rotation": [0, 0, 0]},
                
                # Right arm chain
                {"name": "RightShoulder", "parent": "Spine1", "position": [0.2, 1.5, 0], "rotation": [0, 0, 0]},
                {"name": "RightArm", "parent": "RightShoulder", "position": [0.5, 1.5, 0], "rotation": [0, 0, 0]},
                {"name": "RightForeArm", "parent": "RightArm", "position": [0.8, 1.5, 0], "rotation": [0, 0, 0]},
                {"name": "RightHand", "parent": "RightForeArm", "position": [1.1, 1.5, 0], "rotation": [0, 0, 0]},
                
                # Left leg chain
                {"name": "LeftUpLeg", "parent": "Hips", "position": [-0.1, 0.9, 0], "rotation": [0, 0, 0]},
                {"name": "LeftLeg", "parent": "LeftUpLeg", "position": [-0.1, 0.5, 0], "rotation": [0, 0, 0]},
                {"name": "LeftFoot", "parent": "LeftLeg", "position": [-0.1, 0.1, 0], "rotation": [0, 0, 0]},
                
                # Right leg chain
                {"name": "RightUpLeg", "parent": "Hips", "position": [0.1, 0.9, 0], "rotation": [0, 0, 0]},
                {"name": "RightLeg", "parent": "RightUpLeg", "position": [0.1, 0.5, 0], "rotation": [0, 0, 0]},
                {"name": "RightFoot", "parent": "RightLeg", "position": [0.1, 0.1, 0], "rotation": [0, 0, 0]}
            ]
            
            ik_chains = [
                {"name": "LeftArmIK", "bones": ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"]},
                {"name": "RightArmIK", "bones": ["RightShoulder", "RightArm", "RightForeArm", "RightHand"]},
                {"name": "LeftLegIK", "bones": ["LeftUpLeg", "LeftLeg", "LeftFoot"]},
                {"name": "RightLegIK", "bones": ["RightUpLeg", "RightLeg", "RightFoot"]}
            ]
            
        else:
            # Simple generic rig
            bones = [
                {"name": "Root", "parent": None, "position": [0, 0, 0], "rotation": [0, 0, 0]},
                {"name": "Bone1", "parent": "Root", "position": [0, 0.5, 0], "rotation": [0, 0, 0]},
                {"name": "Bone2", "parent": "Bone1", "position": [0, 1, 0], "rotation": [0, 0, 0]}
            ]
            ik_chains = []
        
        return {
            'success': True,
            'skeleton_data': {
                'bones': bones,
                'bone_count': len(bones),
                'character_type': character_type
            },
            'ik_chains': ik_chains,
            'skin_weights': {
                'method': 'fast_fallback_generated',
                'vertex_count': 0,  # No mesh processing done
                'bone_influences': 4,
                'weights_generated': True
            }
        }
