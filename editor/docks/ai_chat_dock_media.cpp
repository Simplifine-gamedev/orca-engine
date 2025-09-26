/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_chat_dock_media.h"
#include "ai_chat_dock.h"
#include "core/io/file_access.h"
#include "core/io/image.h"
#include "core/io/marshalls.h"
#include "core/core_bind.h"
#include "core/os/os.h"
#include "core/os/time.h"
#include "scene/resources/image_texture.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/label.h"
#include "scene/gui/button.h"
#include "scene/gui/panel_container.h"
#include "scene/resources/style_box_flat.h"
#include "scene/3d/mesh_instance_3d.h"
#include "core/io/resource_saver.h"
#include "core/io/resource_loader.h"
#include "core/io/json.h"
#include "editor/editor_node.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/file_system/editor_file_system.h"
#include "servers/display_server.h"

// ========== IMAGE/MEDIA PROCESSING IMPLEMENTATION ==========
// Image processing methods
bool AIChatDock::_is_image_file(const String &p_path) {
	String ext = p_path.get_extension().to_lower();
	return ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "gif" || 
		   ext == "bmp" || ext == "webp" || ext == "svg";
}

String AIChatDock::_get_mime_type_from_extension(const String &p_path) {
	String ext = p_path.get_extension().to_lower();
	if (ext == "png") return "image/png";
	if (ext == "jpg" || ext == "jpeg") return "image/jpeg";
	if (ext == "gif") return "image/gif";
	if (ext == "bmp") return "image/bmp";
	if (ext == "webp") return "image/webp";
	if (ext == "svg") return "image/svg+xml";
	return "text/plain";
}
bool AIChatDock::_process_image_attachment(AttachedFile &p_file) {
	Ref<Image> image = Image::load_from_file(p_file.path);
	if (image.is_null() || image->is_empty()) {
		return false;
	}

	Vector2i original_size = Vector2i(image->get_width(), image->get_height());
	p_file.original_size = original_size;
	
	// Check if image needs to be downsampled (max 1024px on any side)
	const int MAX_DIMENSION = 1024;
	Vector2i target_size = _calculate_downsampled_size(original_size, MAX_DIMENSION);
	
	if (target_size != original_size) {
		p_file.was_downsampled = true;
		image->resize(target_size.x, target_size.y, Image::INTERPOLATE_LANCZOS);
		
		// Show warning dialog
		call_deferred("_show_image_warning_dialog", p_file.name, original_size, target_size);
	}
	
	p_file.display_size = target_size;
	
	// Convert to base64 for API transmission
	Vector<uint8_t> png_buffer;
	if (p_file.mime_type == "image/jpeg" || p_file.mime_type == "image/jpg") {
		png_buffer = image->save_jpg_to_buffer(0.85f); // Good quality JPEG
	} else {
		png_buffer = image->save_png_to_buffer();
	}
	
	if (png_buffer.size() == 0) {
		return false;
	}
	
	// Encode to base64
	p_file.base64_data = CoreBind::Marshalls::get_singleton()->raw_to_base64(png_buffer);
	
	return true;
}

Vector2i AIChatDock::_calculate_downsampled_size(const Vector2i &p_original, int p_max_dimension) {
	if (p_original.x <= p_max_dimension && p_original.y <= p_max_dimension) {
		return p_original;
	}
	
	float aspect_ratio = (float)p_original.x / (float)p_original.y;
	Vector2i new_size;
	
	if (p_original.x > p_original.y) {
		// Landscape
		new_size.x = p_max_dimension;
		new_size.y = (int)(p_max_dimension / aspect_ratio);
	} else {
		// Portrait or square
		new_size.y = p_max_dimension;
		new_size.x = (int)(p_max_dimension * aspect_ratio);
	}
	
	return new_size;
}

void AIChatDock::_show_image_warning_dialog(const String &p_filename, const Vector2i &p_original, const Vector2i &p_new_size) {
	if (!image_warning_dialog) {
		return;
	}
	
	String message = String("Image '{0}' was downsampled from {1}x{2} to {3}x{4} to reduce file size for transmission.")
		.format(varray(p_filename, p_original.x, p_original.y, p_new_size.x, p_new_size.y));
	
	image_warning_dialog->set_text(message);
	image_warning_dialog->popup_centered(Size2i(500, 150));
}

void AIChatDock::_handle_generated_image(const String &p_base64_data, const String &p_id) {
	print_line("AI Chat: _handle_generated_image called with ID: " + p_id + ", data length: " + String::num_int64(p_base64_data.length()));
	
	if (p_base64_data.is_empty()) {
		print_line("AI Chat: _handle_generated_image - base64 data is empty, aborting");
		return;
	}
	
	// Defer image display to next frame to avoid UI race conditions during streaming
	print_line("AI Chat: _handle_generated_image - calling deferred _display_generated_image_deferred");
	call_deferred("_display_generated_image_deferred", p_base64_data, p_id);
}
void AIChatDock::_display_generated_image_deferred(const String &p_base64_data, const String &p_id) {
	// Decode base64 to image
	Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(p_base64_data);
	if (image_data.size() == 0) {
		print_line("AI Chat: Failed to decode generated image data");
		return;
	}
	
	// Create image from data
	Ref<Image> generated_image = memnew(Image);
	Error err = generated_image->load_png_from_buffer(image_data);
	if (err != OK) {
		// Try JPEG if PNG fails
		err = generated_image->load_jpg_from_buffer(image_data);
		if (err != OK) {
			print_line("AI Chat: Failed to load generated image");
			return;
		}
	}
	
	if (generated_image->is_empty()) {
		print_line("AI Chat: Generated image is empty");
		return;
	}
	
	// Safely find the last assistant message bubble without creating new ones
	PanelContainer *bubble_panel = nullptr;
	if (chat_container) {
		print_line("AI Chat: Searching for assistant message bubble, total children: " + String::num_int64(chat_container->get_child_count()));
		// Look for the last panel container (which should be our assistant message)
		for (int i = chat_container->get_child_count() - 1; i >= 0; i--) {
			Node *child = chat_container->get_child(i);
			print_line("AI Chat: Child " + String::num_int64(i) + " type: " + child->get_class());
			PanelContainer *panel = Object::cast_to<PanelContainer>(child);
			if (panel) {
				print_line("AI Chat: Found PanelContainer at index " + String::num_int64(i));
				bubble_panel = panel;
				break;
			}
		}
	}
	
	if (!bubble_panel) {
		print_line("AI Chat: Could not find assistant message bubble for generated image");
		return;
	} else {
		print_line("AI Chat: Successfully found bubble panel for image display");
	}

	
	// Find the VBoxContainer inside the message bubble
	VBoxContainer *message_vbox = nullptr;
	print_line("AI Chat: Searching for VBoxContainer in bubble panel, children count: " + String::num_int64(bubble_panel->get_child_count()));
	for (int i = 0; i < bubble_panel->get_child_count(); i++) {
		Node *child = bubble_panel->get_child(i);
		print_line("AI Chat: Bubble child " + String::num_int64(i) + " type: " + child->get_class());
		message_vbox = Object::cast_to<VBoxContainer>(child);
		if (message_vbox) {
			print_line("AI Chat: Found VBoxContainer at index " + String::num_int64(i));
			break;
		}
	}
	
	if (!message_vbox) {
		print_line("AI Chat: Could not find VBoxContainer in message bubble - aborting image display");
		return;
	}
	
    // Find and clear the tool placeholder containing "Running tool..." text
	print_line("AI Chat: Searching for tool placeholder in message vbox, children count: " + String::num_int64(message_vbox->get_child_count()));
	bool found_placeholder = false;
	for (int i = 0; i < message_vbox->get_child_count(); i++) {
		Node *child = message_vbox->get_child(i);
		print_line("AI Chat: VBox child " + String::num_int64(i) + " type: " + child->get_class() + " name: " + child->get_name());
		
		// Look for tool placeholder panels
		PanelContainer *panel = Object::cast_to<PanelContainer>(child);
		if (panel && String(panel->get_name()).begins_with("tool_placeholder_")) {
			print_line("AI Chat: Found tool placeholder panel: " + panel->get_name());
			// Clear the tool placeholder content and replace with success message
			while (panel->get_child_count() > 0) {
				Node *panel_child = panel->get_child(0);
				panel->remove_child(panel_child);
				panel_child->queue_free();
			}
			
			// Add success message
			Label *success_label = memnew(Label);
			success_label->set_text("Generated image");
			success_label->add_theme_color_override("font_color", get_theme_color(SNAME("success_color"), SNAME("Editor")));
			success_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
			panel->add_child(success_label);
			
			found_placeholder = true;
			break;
		}
		
		// Also check for RichTextLabel (fallback for other cases)
		RichTextLabel *label = Object::cast_to<RichTextLabel>(child);
		if (label && label->get_text().contains("Calling tool")) {
			print_line("AI Chat: Found RichTextLabel with tool text, updating");
			label->clear();
			label->append_text("Generated image\n\n");
			found_placeholder = true;
			break;
		}
	}
	
	if (!found_placeholder) {
		print_line("AI Chat: No tool placeholder found to clear - this might be okay for some flows");
	}
	
	// Create image display container
	print_line("AI Chat: Creating image display container");
	PanelContainer *image_panel = memnew(PanelContainer);
	message_vbox->add_child(image_panel);
	print_line("AI Chat: Added image panel to message vbox");
	
	// Style the image panel
	Ref<StyleBoxFlat> image_style = memnew(StyleBoxFlat);
	image_style->set_bg_color(get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
	image_style->set_border_width_all(2);
	image_style->set_border_color(get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	image_style->set_corner_radius_all(8);
	image_style->set_content_margin_all(8);
	image_panel->add_theme_style_override("panel", image_style);
	
	VBoxContainer *image_container = memnew(VBoxContainer);
	image_panel->add_child(image_container);
	
	// Add "Generated Image" label
    Label *header_label = memnew(Label);
    header_label->set_text("Generated Image");
	header_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
	header_label->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	image_container->add_child(header_label);
	
	// Resize image for display (max 512px to keep it reasonable)
	Vector2i original_size = Vector2i(generated_image->get_width(), generated_image->get_height());
	Vector2i display_size = _calculate_downsampled_size(original_size, 512);
	
	if (display_size != original_size) {
		generated_image->resize(display_size.x, display_size.y, Image::INTERPOLATE_LANCZOS);
	}
	
	// Create texture and display
	Ref<ImageTexture> generated_texture = ImageTexture::create_from_image(generated_image);
	
	TextureRect *image_display = memnew(TextureRect);
	image_display->set_texture(generated_texture);
	image_display->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	image_display->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	image_display->set_custom_minimum_size(Size2(display_size.x, display_size.y));
	image_container->add_child(image_display);
	
    // Add image info
    HBoxContainer *info_container = memnew(HBoxContainer);
	image_container->add_child(info_container);
	
	Label *size_label = memnew(Label);
	size_label->set_text(String::num_int64(original_size.x) + "x" + String::num_int64(original_size.y));
	size_label->add_theme_font_size_override("font_size", 10);
	size_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
	info_container->add_child(size_label);
	
	// Add spacer to push save button to the right
	Control *spacer = memnew(Control);
	spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	info_container->add_child(spacer);
	
	// Save button for generated images
	Button *save_button = memnew(Button);
	save_button->set_text("Save");
	save_button->set_flat(true);
	save_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Save"), SNAME("EditorIcons")));
	save_button->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	save_button->add_theme_color_override("icon_normal_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	save_button->set_tooltip_text("Save this image to your project");
	save_button->connect("pressed", callable_mp(this, &AIChatDock::_on_save_image_pressed).bind(p_base64_data, "png"));
	info_container->add_child(save_button);
	
	// Store the generated image in the current message
	Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
	print_line("AI Chat: Attempting to save image to chat history, total messages: " + String::num_int64(chat_history.size()));
	
	if (!chat_history.is_empty()) {
		ChatMessage &last_msg = chat_history.write[chat_history.size() - 1];
		print_line("AI Chat: Last message role: " + last_msg.role + ", content: '" + last_msg.content.substr(0, 50) + "...'");
		print_line("AI Chat: Last message current attached files count: " + String::num_int64(last_msg.attached_files.size()));
		
		if (last_msg.role == "assistant") {
					// Add the generated image as an attachment for persistence
		AIChatDock::AttachedFile generated_file;
		generated_file.path = "generated://" + p_id;
		// Create proper unique ID for generated images
		generated_file.name = "gen_img_" + String::num_int64(OS::get_singleton()->get_ticks_msec());
		generated_file.is_image = true;
		generated_file.mime_type = "image/png";
			generated_file.base64_data = p_base64_data;
			generated_file.original_size = original_size;
			generated_file.display_size = display_size;
			generated_file.was_downsampled = (display_size != original_size);
					last_msg.attached_files.push_back(generated_file);
		print_line("AI Chat: Successfully added generated image ID: " + generated_file.name + " to assistant message");
		} else {
			print_line("AI Chat: Cannot save image - last message is not from assistant (role: " + last_msg.role + ")");
		}
	} else {
		print_line("AI Chat: Cannot save image - chat history is empty");
	}
	
	// Update conversation and scroll
	if (current_conversation_index >= 0) {
		conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
		_queue_delayed_save();
	}
	
	print_line("AI Chat: Image display complete, forcing UI refresh");
	
	// Force UI update to show the newly added image immediately
	if (bubble_panel) {
		bubble_panel->queue_redraw();
		print_line("AI Chat: Queued bubble_panel redraw");
	}
	if (chat_container) {
		chat_container->queue_redraw();
		print_line("AI Chat: Queued chat_container redraw");
	}
	// Also update the main dock
	queue_redraw();
	print_line("AI Chat: Queued main dock redraw");
	
	call_deferred("_scroll_to_bottom");
	print_line("AI Chat: _display_generated_image_deferred completed successfully");
}
void AIChatDock::_display_generated_image_in_tool_result(VBoxContainer *p_container, const String &p_base64_data, const Dictionary &p_data) {
	if (!p_container || p_base64_data.is_empty()) {
		return;
	}
	
	// Decode base64 to image
	Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(p_base64_data);
	if (image_data.size() == 0) {
		print_line("AI Chat: Failed to decode generated image data from tool result");
		return;
	}
	
	// Create image from data
	Ref<Image> generated_image = memnew(Image);
	Error err = generated_image->load_png_from_buffer(image_data);
	if (err != OK) {
		// Try JPEG if PNG fails
		err = generated_image->load_jpg_from_buffer(image_data);
		if (err != OK) {
			print_line("AI Chat: Failed to load generated image from tool result");
			return;
		}
	}
	
	if (generated_image->is_empty()) {
		print_line("AI Chat: Generated image from tool result is empty");
		return;
	}
	
	// Create image display container
	VBoxContainer *image_container = memnew(VBoxContainer);
	p_container->add_child(image_container);
	
	// Add image info
	HBoxContainer *info_container = memnew(HBoxContainer);
	image_container->add_child(info_container);
	
	HBoxContainer *prompt_container = memnew(HBoxContainer);
	info_container->add_child(prompt_container);
	
	Label *prompt_icon = memnew(Label);
	prompt_icon->add_theme_icon_override("icon", get_theme_icon(SNAME("Image"), SNAME("EditorIcons")));
	prompt_container->add_child(prompt_icon);
	
    Label *prompt_label = memnew(Label);
	String prompt = p_data.get("prompt", "Generated Image");
	prompt_label->set_text(prompt);
	prompt_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
	prompt_label->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
    prompt_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    prompt_label->set_clip_text(false);
    prompt_label->set_custom_minimum_size(Size2(0, 0));
	prompt_container->add_child(prompt_label);
	
	// Resize image for display (max 200px in tool results to keep them compact)
	Vector2i original_size = Vector2i(generated_image->get_width(), generated_image->get_height());
	Vector2i display_size = _calculate_downsampled_size(original_size, 200);
	
	if (display_size != original_size) {
		generated_image->resize(display_size.x, display_size.y, Image::INTERPOLATE_LANCZOS);
	}
	
	// Create texture and display
	Ref<ImageTexture> generated_texture = ImageTexture::create_from_image(generated_image);
	
	TextureRect *image_display = memnew(TextureRect);
	image_display->set_texture(generated_texture);
	image_display->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	image_display->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	image_display->set_custom_minimum_size(Size2(display_size.x, display_size.y));
	image_container->add_child(image_display);
	
	// Add technical details and save button
	HBoxContainer *tech_container = memnew(HBoxContainer);
	image_container->add_child(tech_container);
	
	Label *size_label = memnew(Label);
	size_label->set_text(String::num_int64(original_size.x) + "x" + String::num_int64(original_size.y));
	size_label->add_theme_font_size_override("font_size", 10);
	size_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
	tech_container->add_child(size_label);
	
	Label *model_label = memnew(Label);
	String model_name = p_data.get("model", "DALL-E");
	model_label->set_text(" | " + model_name);
	model_label->add_theme_font_size_override("font_size", 10);
	model_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
	tech_container->add_child(model_label);
	
	// Add spacer to push save button to the right
	Control *spacer = memnew(Control);
	spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tech_container->add_child(spacer);
	
	// Save button
	Button *save_button = memnew(Button);
	save_button->set_text("Save to...");
	save_button->set_flat(true);
	save_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Save"), SNAME("EditorIcons")));
	save_button->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	save_button->add_theme_color_override("icon_normal_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	save_button->set_tooltip_text("Save this image to your project");
	save_button->connect("pressed", callable_mp(this, &AIChatDock::_on_save_image_pressed).bind(p_base64_data, "png"));
	tech_container->add_child(save_button);
}
void AIChatDock::_display_image_unified(VBoxContainer *p_container, const String &p_base64_data, const Dictionary &p_metadata) {
	if (!p_container || p_base64_data.is_empty()) {
		return;
	}
	
	// Decode base64 to image
	Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(p_base64_data);
	if (image_data.size() == 0) {
		print_line("AI Chat: Failed to decode image data");
		return;
	}
	
	// Create image from data
	Ref<Image> display_image = memnew(Image);
	Error err = display_image->load_png_from_buffer(image_data);
	if (err != OK) {
		// Try JPEG if PNG fails
		err = display_image->load_jpg_from_buffer(image_data);
		if (err != OK) {
			print_line("AI Chat: Failed to load image");
			return;
		}
	}
	
	if (display_image->is_empty()) {
		print_line("AI Chat: Image is empty");
		return;
	}
	
	// Create unified image display container
	VBoxContainer *image_container = memnew(VBoxContainer);
	p_container->add_child(image_container);
	
	// Extract metadata with defaults
	String title = p_metadata.get("prompt", p_metadata.get("name", "Image"));
	String model_name = p_metadata.get("model", "");
	String file_path = p_metadata.get("path", "");
	bool is_generated = file_path.begins_with("generated://");
	int max_display_size = is_generated ? 200 : 150; // Generated images slightly larger
	
    // Add image title/info with proper wrapping so long text doesn't expand panel width
    HBoxContainer *info_container = memnew(HBoxContainer);
    info_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    image_container->add_child(info_container);

    // Left: icon
    Label *icon = memnew(Label);
    icon->add_theme_icon_override("icon", get_theme_icon(SNAME("Image"), SNAME("EditorIcons")));
    info_container->add_child(icon);

    // Right: a wrapping label in a VBox so it can take full width and wrap
    VBoxContainer *title_vbox = memnew(VBoxContainer);
    title_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    info_container->add_child(title_vbox);

    Label *title_label = memnew(Label);
    title_label->set_text(title);
    title_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    title_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    title_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
    title_label->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
    title_vbox->add_child(title_label);
	
	// Resize image for display
	Vector2i original_size = Vector2i(display_image->get_width(), display_image->get_height());
	Vector2i display_size = _calculate_downsampled_size(original_size, max_display_size);
	
	if (display_size != original_size) {
		display_image->resize(display_size.x, display_size.y, Image::INTERPOLATE_LANCZOS);
	}
	
	// Create texture and display
	Ref<ImageTexture> image_texture = ImageTexture::create_from_image(display_image);
	
	TextureRect *image_display = memnew(TextureRect);
	image_display->set_texture(image_texture);
	image_display->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	image_display->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	image_display->set_custom_minimum_size(Size2(display_size.x, display_size.y));
	image_container->add_child(image_display);
	
	// Add technical details and controls
	HBoxContainer *details_container = memnew(HBoxContainer);
	image_container->add_child(details_container);
	
	// Size info
	Label *size_label = memnew(Label);
	size_label->set_text(String::num_int64(original_size.x) + "x" + String::num_int64(original_size.y));
	size_label->add_theme_font_size_override("font_size", 10);
	size_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
	details_container->add_child(size_label);
	
	// Model info (for generated images)
	if (!model_name.is_empty()) {
		Label *model_label = memnew(Label);
		model_label->set_text(" | " + model_name);
		model_label->add_theme_font_size_override("font_size", 10);
		model_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
		details_container->add_child(model_label);
	}
	
	// File path button (for attached images)
	if (!file_path.is_empty() && !is_generated) {
		Button *file_link = memnew(Button);
		file_link->set_text(" | " + file_path.get_file());
		file_link->set_flat(true);
		file_link->add_theme_font_size_override("font_size", 10);
		file_link->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
		file_link->set_tooltip_text("Click to open: " + file_path);
		file_link->connect("pressed", callable_mp(this, &AIChatDock::_on_tool_file_link_pressed).bind(file_path));
		details_container->add_child(file_link);
	}
	
	// Add spacer to push save button to the right
	Control *spacer = memnew(Control);
	spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	details_container->add_child(spacer);
	
	// Save button
	Button *save_button = memnew(Button);
	save_button->set_text("Save to...");
	save_button->set_flat(true);
	save_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Save"), SNAME("EditorIcons")));
	save_button->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	save_button->add_theme_color_override("icon_normal_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	save_button->set_tooltip_text("Save this image to your project");
	save_button->connect("pressed", callable_mp(this, &AIChatDock::_on_save_image_pressed).bind(p_base64_data, "png"));
	details_container->add_child(save_button);
}

bool AIChatDock::_save_base64_image_to_path(const String &p_base64_data, const String &p_file_path) {
	if (p_base64_data.is_empty() || p_file_path.is_empty()) {
		return false;
	}
	Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(p_base64_data);
	if (image_data.size() == 0) {
		return false;
	}
	Ref<FileAccess> file = FileAccess::open(p_file_path, FileAccess::WRITE);
	if (file.is_null()) {
		return false;
	}
	file->store_buffer(image_data);
	file->close();

	// Immediately notify the editor's file system so the resource is imported and discoverable.
	if (EditorFileSystem::get_singleton()) {
		EditorFileSystem::get_singleton()->update_file(p_file_path);
		// Debounce costly/rescanning calls to avoid re-entrancy crashes in SceneTreeDock.
		// Schedule a single scan shortly after the last write.
		static uint64_t s_last_scan_request_ms = 0;
		s_last_scan_request_ms = OS::get_singleton()->get_ticks_msec();
		uint64_t scheduled_at = s_last_scan_request_ms;
		Ref<SceneTreeTimer> timer = get_tree()->create_timer(0.4, true);
		timer->connect("timeout", callable_mp(this, &AIChatDock::_on_filesystem_debounced_scan).bind(scheduled_at));
		
		// Force immediate reimport to generate .import file
		Vector<String> to_reimport;
		to_reimport.push_back(p_file_path);
		EditorFileSystem::get_singleton()->reimport_files(to_reimport);
		
		// Clear any cached load failure so the resource can be loaded fresh
		if (ResourceCache::has(p_file_path)) {
			Ref<Resource> cached = ResourceCache::get_ref(p_file_path);
			if (cached.is_valid()) {
				cached->reload_from_file();
			}
		}
	}
	return true;
}

void AIChatDock::_on_save_image_pressed(const String &p_base64_data, const String &p_format) {
	if (p_base64_data.is_empty()) {
		return;
	}
	
	// Store the image data for saving
	pending_save_image_data = p_base64_data;
	pending_save_image_format = p_format;
	
	// Set up the save dialog
	save_image_dialog->set_current_file("generated_image." + p_format);
	save_image_dialog->popup_centered(Size2(800, 600));
}
void AIChatDock::_on_save_image_location_selected(const String &p_file_path) {
	if (p_file_path.is_empty() || pending_save_image_data.is_empty()) {
		return;
	}
	
	String save_path = p_file_path;
	
	// Decode base64 to image data
	Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(pending_save_image_data);
	if (image_data.size() == 0) {
		print_line("AI Chat: Failed to decode image data for saving");
		return;
	}
	
	// Save the image file
	Ref<FileAccess> file = FileAccess::open(save_path, FileAccess::WRITE);
	if (file.is_null()) {
		print_line("AI Chat: Failed to open file for writing: " + save_path);
		return;
	}
	
	file->store_buffer(image_data);
	file->close();
	
	// Notify the editor file system so the image is imported and usable immediately.
	if (EditorFileSystem::get_singleton()) {
		EditorFileSystem::get_singleton()->update_file(save_path);
		// Debounce a follow-up scan to avoid re-entrancy issues in editor docks.
		static uint64_t s_last_scan_request_ms = 0;
		s_last_scan_request_ms = OS::get_singleton()->get_ticks_msec();
		uint64_t scheduled_at = s_last_scan_request_ms;
		Ref<SceneTreeTimer> timer = get_tree()->create_timer(0.4, true);
		timer->connect("timeout", callable_mp(this, &AIChatDock::_on_filesystem_debounced_scan).bind(scheduled_at));
		// Force immediate reimport for this file to avoid stale previews until next scan.
		Vector<String> to_reimport;
		to_reimport.push_back(save_path);
		EditorFileSystem::get_singleton()->reimport_files(to_reimport);
		
		// Clear any cached load failure so the resource can be loaded fresh
		if (ResourceCache::has(save_path)) {
			Ref<Resource> cached = ResourceCache::get_ref(save_path);
			if (cached.is_valid()) {
				cached->reload_from_file();
			}
		}
	}
	
	print_line("AI Chat: Image saved successfully to: " + save_path);
	
	// Show success notification to user
	if (EditorNode::get_singleton()) {
		EditorNode::get_singleton()->show_warning("Image saved successfully to: " + save_path.get_file());
	}
	
	// Clear pending data
	pending_save_image_data = "";
	pending_save_image_format = "";
}

void AIChatDock::_on_export_button_pressed() {
	// Check if there's a current conversation to export
	if (current_conversation_index < 0 || current_conversation_index >= conversations.size()) {
		print_line("AI Chat: No conversation to export");
		return;
	}
	
	const Conversation &conv = conversations[current_conversation_index];
	
	// Generate a default filename based on conversation title and timestamp
	String safe_title = conv.title.replace(" ", "_").replace("/", "_").replace("\\", "_");
	safe_title = safe_title.replace(":", "_").replace("*", "_").replace("?", "_");
	safe_title = safe_title.replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_");
	
	String default_filename = "conversation_" + safe_title + "_" + conv.created_timestamp.replace(":", "-").replace(" ", "_") + ".json";
	
	// Show file dialog
	export_dialog->set_current_file(default_filename);
	export_dialog->popup_centered(Size2(800, 600));
}

void AIChatDock::_on_export_file_selected(const String &p_file_path) {
	// Check if there's a current conversation to export
	if (current_conversation_index < 0 || current_conversation_index >= conversations.size()) {
		print_line("AI Chat: No conversation to export");
		return;
	}
	
	const Conversation &conv = conversations[current_conversation_index];
	
	// Create JSON structure for export
	Dictionary export_data;
	export_data["conversation_id"] = conv.id;
	export_data["title"] = conv.title;
	export_data["created_timestamp"] = conv.created_timestamp;
	export_data["last_modified_timestamp"] = conv.last_modified_timestamp;
	export_data["exported_at"] = _get_timestamp();
		export_data["godot_ai_chat_version"] = FRONTEND_VERSION;
	
	// Export messages
	Array messages_array;
	for (const ChatMessage &msg : conv.messages) {
		Dictionary msg_dict;
		msg_dict["role"] = msg.role;
		msg_dict["content"] = msg.content;
		msg_dict["timestamp"] = msg.timestamp;
		
		// Include tool calls if present
		if (msg.tool_calls.size() > 0) {
			msg_dict["tool_calls"] = msg.tool_calls;
		}
		
		// Include tool response data if present
		if (!msg.tool_call_id.is_empty()) {
			msg_dict["tool_call_id"] = msg.tool_call_id;
			msg_dict["name"] = msg.name;
		}
		
		// Include attached files info (without binary data)
		if (msg.attached_files.size() > 0) {
			Array files_info;
			for (const AttachedFile &file : msg.attached_files) {
				Dictionary file_info;
				file_info["path"] = file.path;
				file_info["name"] = file.name;
				file_info["is_image"] = file.is_image;
				file_info["mime_type"] = file.mime_type;
				file_info["is_node"] = file.is_node;
				if (file.is_node) {
					file_info["node_path"] = file.node_path;
					file_info["node_type"] = file.node_type;
				}
				files_info.push_back(file_info);
			}
			msg_dict["attached_files"] = files_info;
		}
		
		// Include tool results if present
		if (msg.tool_results.size() > 0) {
			msg_dict["tool_results"] = msg.tool_results;
		}
		
		// Include reasoning content if present (thinking mode)
		if (!msg.reasoning_content.is_empty()) {
			msg_dict["reasoning_content"] = msg.reasoning_content;
		}
		
		if (msg.thinking_blocks.size() > 0) {
			msg_dict["thinking_blocks"] = msg.thinking_blocks;
		}
		
		messages_array.push_back(msg_dict);
	}
	export_data["messages"] = messages_array;
	
	// Convert to JSON string
	String json_string = JSON::stringify(export_data, "\t");
	
	// Save to file
	Ref<FileAccess> file = FileAccess::open(p_file_path, FileAccess::WRITE);
	if (file.is_null()) {
		print_line("AI Chat: Failed to open file for writing: " + p_file_path);
		return;
	}
	
	file->store_string(json_string);
	file->close();
	
	print_line("AI Chat: Conversation exported to: " + p_file_path);
	
	// Show success notification if available
	if (status_notification_panel) {
		_show_status_notification("success", "Conversation exported successfully", "Save", 2.0);
	}
}

