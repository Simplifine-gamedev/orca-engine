/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */
#ifndef AI_IMAGE_LAZY_LOADER_H
#define AI_IMAGE_LAZY_LOADER_H

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/popup_menu.h"
#include "core/variant/dictionary.h"

class AIImageLazyLoader : public RefCounted {
public:
	// Create a lazy-loading image placeholder that only decodes when clicked
	// Returns a VBoxContainer with a "Show Image" button that loads on demand
	static VBoxContainer* create_lazy_image_placeholder(
		const String &p_base64_data,
		const Dictionary &p_metadata,
		VBoxContainer *p_parent
	);

private:
	// Callback when user clicks to load the image
	static void _on_load_image_pressed(Button *p_button, VBoxContainer *p_container, const String &p_base64_data, const Dictionary &p_metadata);
	
	// Save callback - shows enhanced file dialog
	static void _on_simple_save_pressed(const String &p_base64_data, const Dictionary &p_metadata);
	
	// Show file dialog with built-in resolution dropdown
	static void _show_save_dialog_with_resolution(const String &p_base64_data, const Dictionary &p_metadata, const Vector2i &p_original_size);
	
	// Callback when file location is selected from enhanced dialog
	static void _on_enhanced_file_save_selected(const String &p_file_path);
};

#endif // AI_IMAGE_LAZY_LOADER_H
