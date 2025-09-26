/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "ai_chat_dock_types.h"
#include "core/math/vector2i.h"
#include "core/variant/dictionary.h"
#include "scene/gui/box_container.h"

class AIChatDock;

class AIChatDockMedia {
public:
	// Image processing methods
	static bool is_image_file(const String &p_path);
	static String get_mime_type_from_extension(const String &p_path);
	static bool process_image_attachment(AIChatDockTypes::AttachedFile &p_file);
	static Vector2i calculate_downsampled_size(const Vector2i &p_original, int p_max_dimension = 1024);
	static void process_image_attachment_async(AIChatDock *p_dock, const String &p_file_path, const String &p_name, const String &p_mime_type);
	
	// Image generation handling
	static void handle_generated_image(AIChatDock *p_dock, const String &p_base64_data, const String &p_id);
	static void display_generated_image_deferred(AIChatDock *p_dock, const String &p_base64_data, const String &p_id);
	static void display_generated_image_in_tool_result(VBoxContainer *p_container, const String &p_base64_data, const Dictionary &p_data);
	
	// Unified image display method for all image types
	static void display_image_unified(VBoxContainer *p_container, const String &p_base64_data, const Dictionary &p_metadata = Dictionary());
	
	// Image saving
	static bool save_base64_image_to_path(const String &p_base64_data, const String &p_file_path);
	static void on_save_image_pressed(AIChatDock *p_dock, const String &p_base64_data, const String &p_format);
	static void on_save_image_location_selected(AIChatDock *p_dock, const String &p_file_path);
	static void show_image_warning_dialog(AIChatDock *p_dock, const String &p_filename, const Vector2i &p_original, const Vector2i &p_new_size);
	
	// 3D Model generation handling
	static void on_save_3d_model_pressed(AIChatDock *p_dock, const String &p_glb_data, const String &p_prompt, const String &p_save_path);
	static void on_import_3d_model_to_scene_pressed(AIChatDock *p_dock, const String &p_glb_data, const String &p_prompt, const String &p_save_path);
	static void on_3d_model_save_location_selected(AIChatDock *p_dock, const String &p_file_path);
	static bool save_glb_model_to_path(const String &p_glb_data, const String &p_file_path);
	
	// Utility methods
	static String get_conversation_image(AIChatDock *p_dock, const String &p_image_id);
};
