/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */
#include "ai_image_lazy_loader.h"
#include "scene/gui/label.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/button.h"
#include "scene/gui/control.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/option_button.h"
#include "scene/resources/image_texture.h"
#include "core/io/image.h"
#include "core/io/marshalls.h"
#include "core/core_bind.h"
#include "core/io/file_access.h"
#include "core/io/resource.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/file_system/editor_file_system.h"

// Static storage for save workflow
static String g_pending_base64_data;
static Dictionary g_pending_metadata;
static OptionButton *g_resolution_dropdown = nullptr;

VBoxContainer* AIImageLazyLoader::create_lazy_image_placeholder(
	const String &p_base64_data,
	const Dictionary &p_metadata,
	VBoxContainer *p_parent
) {
	print_line("  - base64_data length: " + String::num_int64(p_base64_data.length()));
	print_line("  - parent valid: " + String(p_parent ? "yes" : "no"));
	
	if (!p_parent || p_base64_data.is_empty()) {
		return nullptr;
	}
	
	VBoxContainer *image_container = memnew(VBoxContainer);
	p_parent->add_child(image_container);
	
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
	
	// Create buttons container
	HBoxContainer *buttons_container = memnew(HBoxContainer);
	image_container->add_child(buttons_container);
	
	// Create load button (placeholder)
	Button *load_button = memnew(Button);
	load_button->set_text("Show Image");
	load_button->add_theme_icon_override("icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("GuiVisibilityVisible"), EditorStringName(EditorIcons)));
	load_button->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("accent_color"), EditorStringName(Editor)));
	buttons_container->add_child(load_button);
	
	// CRITICAL: Add save button BEFORE image is loaded (lazy state)
	Button *save_button_lazy = memnew(Button);
	save_button_lazy->set_text("Save");
	save_button_lazy->set_flat(true);
	save_button_lazy->add_theme_icon_override("icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Save"), EditorStringName(EditorIcons)));
	save_button_lazy->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("accent_color"), EditorStringName(Editor)));
	save_button_lazy->set_tooltip_text("Save this image to your project");
	save_button_lazy->connect("pressed", callable_mp_static(&AIImageLazyLoader::_on_simple_save_pressed).bind(p_base64_data, p_metadata));
	buttons_container->add_child(save_button_lazy);
	
	// Create placeholder for image (will be populated when button clicked)
	VBoxContainer *image_placeholder = memnew(VBoxContainer);
	image_placeholder->set_name("image_display_placeholder");
	image_container->add_child(image_placeholder);
	
	// Connect button to load image on demand
	load_button->connect("pressed", callable_mp_static(&AIImageLazyLoader::_on_load_image_pressed).bind(load_button, image_placeholder, p_base64_data, p_metadata));
	
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
	
	// Add controls (size info + save button)
	HBoxContainer *controls_container = memnew(HBoxContainer);
	p_container->add_child(controls_container);
	
	Label *size_label = memnew(Label);
	size_label->set_text(String::num_int64(original_size.x) + "x" + String::num_int64(original_size.y) + " pixels");
	size_label->add_theme_font_size_override("font_size", 10);
	size_label->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("font_color"), EditorStringName(Editor)) * Color(1, 1, 1, 0.7));
	controls_container->add_child(size_label);
	
	// Spacer to push save button to the right
	Control *spacer = memnew(Control);
	spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	controls_container->add_child(spacer);
	
	// Simple save button - use the chat dock's existing save functionality
	Button *save_button = memnew(Button);
	save_button->set_text("Save");
	save_button->set_flat(true);
	save_button->add_theme_icon_override("icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Save"), EditorStringName(EditorIcons)));
	save_button->add_theme_color_override("font_color", EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("accent_color"), EditorStringName(Editor)));
	save_button->set_tooltip_text("Save this image to your project");
	
	// Connect to the simple save callback (no dropdown - just use original size)
	save_button->connect("pressed", callable_mp_static(&AIImageLazyLoader::_on_simple_save_pressed).bind(p_base64_data, p_metadata));
	controls_container->add_child(save_button);
	
}

void AIImageLazyLoader::_on_simple_save_pressed(const String &p_base64_data, const Dictionary &p_metadata) {
	if (p_base64_data.is_empty()) {
		return;
	}
	
	// Store data globally for the resolution workflow
	g_pending_base64_data = p_base64_data;
	g_pending_metadata = p_metadata;
	
	// Decode image to get original size
	Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(p_base64_data);
	if (image_data.size() == 0) {
		return;
	}
	
	Ref<Image> img = memnew(Image);
	Error err = img->load_png_from_buffer(image_data);
	if (err != OK) {
		img->load_jpg_from_buffer(image_data);
	}
	
	Vector2i original_size = Vector2i(1024, 1024);
	if (!img->is_empty()) {
		original_size = Vector2i(img->get_width(), img->get_height());
	}
	
	// Create resolution popup menu
	PopupMenu *resolution_menu = memnew(PopupMenu);
	EditorNode::get_singleton()->add_child(resolution_menu);
	
	// Add standard resolutions up to original size
	int max_dim = MAX(original_size.x, original_size.y);
	Vector<int> resolutions = {8, 16, 32, 64, 128, 256, 512, 1024, 2048};
	
	int menu_idx = 0;
	for (int i = 0; i < resolutions.size(); i++) {
		if (resolutions[i] <= max_dim) {
			String label = String::num_int64(resolutions[i]) + "x" + String::num_int64(resolutions[i]);
			resolution_menu->add_item(label, menu_idx);
			resolution_menu->set_item_metadata(menu_idx, resolutions[i]);
			menu_idx++;
		}
	}
	
	// Add original resolution
	resolution_menu->add_separator();
	String orig_label = "Original (" + String::num_int64(original_size.x) + "x" + String::num_int64(original_size.y) + ")";
	resolution_menu->add_item(orig_label, menu_idx);
	resolution_menu->set_item_metadata(menu_idx, -1); // -1 = original
	
	// Skip the popup menu - instead go directly to enhanced file dialog
	// The file dialog will have a resolution dropdown built-in
	resolution_menu->queue_free(); // Don't need this anymore
	
	// Show enhanced file dialog with resolution selector
	AIImageLazyLoader::_show_save_dialog_with_resolution(p_base64_data, p_metadata, original_size);
}

void AIImageLazyLoader::_show_save_dialog_with_resolution(const String &p_base64_data, const Dictionary &p_metadata, const Vector2i &p_original_size) {
	
	// Create file dialog
	EditorFileDialog *file_dialog = memnew(EditorFileDialog);
	file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	file_dialog->set_access(EditorFileDialog::ACCESS_RESOURCES);
	file_dialog->add_filter("*.png", "PNG Images");
	
	// Create compact resolution selector
	VBoxContainer *res_container = memnew(VBoxContainer);
	res_container->set_custom_minimum_size(Size2(180, 0)); // Compact width
	
	Label *res_label = memnew(Label);
	res_label->set_text("Resolution:");
	res_label->add_theme_font_size_override("font_size", 12);
	res_container->add_child(res_label);
	
	// Create resolution dropdown
	OptionButton *res_dropdown = memnew(OptionButton);
	res_dropdown->set_name("resolution_selector");
	res_dropdown->set_custom_minimum_size(Size2(160, 0));
	
	// Add resolution options
	int max_dim = MAX(p_original_size.x, p_original_size.y);
	Vector<int> resolutions = {8, 16, 32, 64, 128, 256, 512, 1024, 2048};
	
	int default_idx = 0;
	for (int i = 0; i < resolutions.size(); i++) {
		if (resolutions[i] <= max_dim) {
			String label = String::num_int64(resolutions[i]) + "x" + String::num_int64(resolutions[i]);
			res_dropdown->add_item(label);
			res_dropdown->set_item_metadata(res_dropdown->get_item_count() - 1, resolutions[i]);
			default_idx = res_dropdown->get_item_count() - 1;
		}
	}
	
	// Add original resolution
	String orig_label = "Original (" + String::num_int64(p_original_size.x) + "x" + String::num_int64(p_original_size.y) + ")";
	res_dropdown->add_item(orig_label);
	res_dropdown->set_item_metadata(res_dropdown->get_item_count() - 1, -1); // -1 = original
	res_dropdown->select(default_idx); // Select highest available by default
	
	res_container->add_child(res_dropdown);
	
	// Add resolution selector to the file dialog using add_side_menu()
	file_dialog->add_side_menu(res_container, "Resolution Options");
	
	// Store globally for callback (simpler than searching for dialog)
	g_pending_base64_data = p_base64_data;
	g_pending_metadata = p_metadata;
	g_resolution_dropdown = res_dropdown;
	
	// Set default filename
	String default_name = "generated_image";
	if (p_metadata.has("image_id")) {
		default_name = p_metadata.get("image_id", "");
	}
	default_name += ".png";
	file_dialog->set_current_file(default_name);
	
	// Connect to save callback
	file_dialog->connect("file_selected", callable_mp_static(&AIImageLazyLoader::_on_enhanced_file_save_selected), CONNECT_ONE_SHOT);
	
	// Show dialog
	EditorNode::get_singleton()->add_child(file_dialog);
	file_dialog->popup_centered(Size2(800, 600));
}

void AIImageLazyLoader::_on_enhanced_file_save_selected(const String &p_file_path) {
	if (p_file_path.is_empty() || g_pending_base64_data.is_empty()) {
		return;
	}
	
	// Get selected resolution from the global dropdown reference
	int selected_resolution = -1;
	if (g_resolution_dropdown && g_resolution_dropdown->get_selected() >= 0) {
		selected_resolution = g_resolution_dropdown->get_item_metadata(g_resolution_dropdown->get_selected());
	}
	
	
	// Decode image
	Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(g_pending_base64_data);
	Ref<Image> image = memnew(Image);
	Error err = image->load_png_from_buffer(image_data);
	if (err != OK) {
		image->load_jpg_from_buffer(image_data);
	}
	
	if (image->is_empty()) {
		EditorNode::get_singleton()->show_warning("Failed to load image");
		return;
	}
	
	Vector2i original_size = Vector2i(image->get_width(), image->get_height());
	
	// Resize if requested
	if (selected_resolution > 0) {
		float aspect = (float)original_size.x / (float)original_size.y;
		Vector2i new_size;
		
		if (aspect > 1.0f) {
			new_size.x = selected_resolution;
			new_size.y = (int)(selected_resolution / aspect);
		} else {
			new_size.y = selected_resolution;
			new_size.x = (int)(selected_resolution * aspect);
		}
		
		image->resize(new_size.x, new_size.y, Image::INTERPOLATE_LANCZOS);
	}
	
	// Save and trigger import
	Vector<uint8_t> png_buffer = image->save_png_to_buffer();
	Ref<FileAccess> file = FileAccess::open(p_file_path, FileAccess::WRITE);
	if (file.is_null()) {
		EditorNode::get_singleton()->show_warning("Failed to save");
		return;
	}
	file->store_buffer(png_buffer);
	file->close();
	
	// Trigger Godot import system
	if (EditorFileSystem::get_singleton()) {
		EditorFileSystem::get_singleton()->update_file(p_file_path);
		Vector<String> to_reimport;
		to_reimport.push_back(p_file_path);
		EditorFileSystem::get_singleton()->reimport_files(to_reimport);
		if (ResourceCache::has(p_file_path)) {
			Ref<Resource> cached = ResourceCache::get_ref(p_file_path);
			if (cached.is_valid()) {
				cached->reload_from_file();
			}
		}
		EditorFileSystem::get_singleton()->scan_changes();
	}
	
	EditorNode::get_singleton()->show_warning("✅ Image saved: " + p_file_path.get_file());
	
	// Clear static data
	g_pending_base64_data = "";
	g_resolution_dropdown = nullptr;
}
