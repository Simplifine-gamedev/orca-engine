/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */
#include "ai_image_lazy_loader.h"
#include "scene/gui/label.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/button.h"
#include "scene/resources/image_texture.h"
#include "core/io/image.h"
#include "core/io/marshalls.h"
#include "core/core_bind.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"

VBoxContainer* AIImageLazyLoader::create_lazy_image_placeholder(
	const String &p_base64_data,
	const Dictionary &p_metadata,
	VBoxContainer *p_parent
) {
	print_line("AI Image Lazy Loader: create_lazy_image_placeholder called!");
	print_line("  - base64_data length: " + String::num_int64(p_base64_data.length()));
	print_line("  - parent valid: " + String(p_parent ? "yes" : "no"));
	
	if (!p_parent || p_base64_data.is_empty()) {
		print_line("AI Image Lazy Loader: ERROR - parent null or data empty!");
		return nullptr;
	}
	
	VBoxContainer *image_container = memnew(VBoxContainer);
	p_parent->add_child(image_container);
	print_line("AI Image Lazy Loader: Created image_container and added to parent");
	
	// Extract metadata
	String title = p_metadata.get("prompt", p_metadata.get("name", "Image"));
	String model_name = p_metadata.get("model", "");
	
	// Create info header
	HBoxContainer *info_header = memnew(HBoxContainer);
	image_container->add_child(info_header);
	
	Label *icon = memnew(Label);
	icon->add_theme_icon_override("icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Image"), EditorStringName(EditorIcons)));
	info_header->add_child(icon);
	
	Label *title_label = memnew(Label);
	title_label->set_text(title);
	title_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	title_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	title_label->add_theme_font_override("font", EditorNode::get_singleton()->get_editor_theme()->get_font(SNAME("bold"), EditorStringName(EditorFonts)));
	title_label->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("accent_color"), EditorStringName(Editor)));
	info_header->add_child(title_label);
	
	// Create load button (placeholder)
	Button *load_button = memnew(Button);
	load_button->set_text("Show Image");
	load_button->add_theme_icon_override("icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("GuiVisibilityVisible"), EditorStringName(EditorIcons)));
	load_button->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("accent_color"), EditorStringName(Editor)));
	
	// Create placeholder for image (will be populated when button clicked)
	VBoxContainer *image_placeholder = memnew(VBoxContainer);
	image_placeholder->set_name("image_display_placeholder");
	image_container->add_child(image_placeholder);
	
	// Connect button to load image on demand
	load_button->connect("pressed", callable_mp_static(&AIImageLazyLoader::_on_load_image_pressed).bind(load_button, image_placeholder, p_base64_data, p_metadata));
	
	image_container->add_child(load_button);
	
	// Add size info label
	Label *info_label = memnew(Label);
	String info_text = "Click to load image";
	if (!model_name.is_empty()) {
		info_text += " (" + model_name + ")";
	}
	info_label->set_text(info_text);
	info_label->add_theme_font_size_override("font_size", 10);
	info_label->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("font_color"), EditorStringName(Editor)) * Color(1, 1, 1, 0.6));
	image_container->add_child(info_label);
	
	return image_container;
}

void AIImageLazyLoader::_on_load_image_pressed(Button *p_button, VBoxContainer *p_container, const String &p_base64_data, const Dictionary &p_metadata) {
	if (!p_button || !p_container || p_base64_data.is_empty()) {
		return;
	}
	
	// Hide the load button after clicking
	p_button->set_visible(false);
	
	// Decode base64 to image
	Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(p_base64_data);
	if (image_data.size() == 0) {
		Label *error_label = memnew(Label);
		error_label->set_text("Failed to decode image data");
		error_label->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("error_color"), EditorStringName(Editor)));
		p_container->add_child(error_label);
		return;
	}
	
	// Create image from data
	Ref<Image> display_image = memnew(Image);
	Error err = display_image->load_png_from_buffer(image_data);
	if (err != OK) {
		// Try JPEG if PNG fails
		err = display_image->load_jpg_from_buffer(image_data);
		if (err != OK) {
			Label *error_label = memnew(Label);
			error_label->set_text("Failed to load image");
			error_label->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("error_color"), EditorStringName(Editor)));
			p_container->add_child(error_label);
			return;
		}
	}
	
	if (display_image->is_empty()) {
		Label *error_label = memnew(Label);
		error_label->set_text("Image is empty");
		error_label->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("error_color"), EditorStringName(Editor)));
		p_container->add_child(error_label);
		return;
	}
	
	// Resize image for display (max 400px for tool results)
	Vector2i original_size = Vector2i(display_image->get_width(), display_image->get_height());
	int max_dimension = 400;
	Vector2i display_size = original_size;
	
	if (original_size.x > max_dimension || original_size.y > max_dimension) {
		float aspect_ratio = (float)original_size.x / (float)original_size.y;
		if (original_size.x > original_size.y) {
			display_size.x = max_dimension;
			display_size.y = (int)(max_dimension / aspect_ratio);
		} else {
			display_size.y = max_dimension;
			display_size.x = (int)(max_dimension * aspect_ratio);
		}
		display_image->resize(display_size.x, display_size.y, Image::INTERPOLATE_LANCZOS);
	}
	
	// Create texture and display
	Ref<ImageTexture> image_texture = ImageTexture::create_from_image(display_image);
	
	TextureRect *image_display = memnew(TextureRect);
	image_display->set_texture(image_texture);
	image_display->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	image_display->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	image_display->set_custom_minimum_size(Size2(display_size.x, display_size.y));
	p_container->add_child(image_display);
	
	// Add size info
	Label *size_label = memnew(Label);
	size_label->set_text(String::num_int64(original_size.x) + "x" + String::num_int64(original_size.y) + " pixels");
	size_label->add_theme_font_size_override("font_size", 10);
	size_label->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("font_color"), EditorStringName(Editor)) * Color(1, 1, 1, 0.7));
	p_container->add_child(size_label);
	
	print_line("AI Image Lazy Loader: Successfully loaded and displayed image (" + String::num_int64(original_size.x) + "x" + String::num_int64(original_size.y) + ")");
}
