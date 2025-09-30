/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */
#ifndef AI_IMAGE_LAZY_LOADER_H
#define AI_IMAGE_LAZY_LOADER_H

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
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
};

#endif // AI_IMAGE_LAZY_LOADER_H
