/**************************************************************************/
/*  design_studio_2d_editor_plugin.cpp                                    */
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

#include "design_studio_2d_editor_plugin.h"

#include "core/core_bind.h"
#include "core/io/http_client.h"
#include "core/io/http_client_tcp.h"
#include "core/io/json.h"
#include "core/os/os.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/themes/editor_scale.h"
#include "scene/main/http_request.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/center_container.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/option_button.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/split_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/dialogs.h"
#include "scene/resources/image_texture.h"

// SpriteSheetConfigDialog implementation
void SpriteSheetConfigDialog::_notification(int p_what) {
}

void SpriteSheetConfigDialog::setup(int p_num_rows, DesignStudio2DEditor *p_parent) {
	num_rows = p_num_rows;
	parent_editor = p_parent;
	
	// Clear previous setup
	row_description_fields.clear();
	
	set_title("Configure Sprite Sheet Animation");
	set_min_size(Size2(600, 400));
	
	// Clear existing content (keep dialog buttons)
	for (int i = get_child_count() - 1; i >= 0; i--) {
		Node *child = get_child(i);
		// Skip dialog buttons
		if (String(child->get_name()).begins_with("@")) {
			continue;
		}
		child->queue_free();
	}
	
	VBoxContainer *main_vbox = memnew(VBoxContainer);
	add_child(main_vbox);
	
	// Format selection
	Label *format_label = memnew(Label);
	format_label->set_text("Choose Animation Format:");
	format_label->add_theme_font_size_override("font_size", 16);
	main_vbox->add_child(format_label);
	
	format_dropdown = memnew(OptionButton);
	format_dropdown->add_item("Humanoid Walking");
	format_dropdown->add_item("Fire/Flame");
	format_dropdown->add_item("Custom");
	format_dropdown->select(2); // Default to Custom
	format_dropdown->connect("item_selected", callable_mp(this, &SpriteSheetConfigDialog::_on_format_selected));
	main_vbox->add_child(format_dropdown);
	
	main_vbox->add_child(memnew(HSeparator));
	
	// Row configurations
	Label *rows_label = memnew(Label);
	rows_label->set_text("Describe each row of your sprite sheet:");
	main_vbox->add_child(rows_label);
	
	// Scrollable container for row configs
	ScrollContainer *scroll = memnew(ScrollContainer);
	scroll->set_custom_minimum_size(Size2(0, 200));
	main_vbox->add_child(scroll);
	
	row_configs_container = memnew(VBoxContainer);
	scroll->add_child(row_configs_container);
	
	// Create row description fields
	for (int i = 0; i < num_rows; i++) {
		VBoxContainer *row_vbox = memnew(VBoxContainer);
		row_configs_container->add_child(row_vbox);
		
		Label *row_label = memnew(Label);
		row_label->set_text(vformat("Row %d:", i + 1));
		row_vbox->add_child(row_label);
		
		LineEdit *row_input = memnew(LineEdit);
		row_input->set_placeholder(vformat("e.g., Character walking to the right"));
		row_vbox->add_child(row_input);
		
		row_description_fields.push_back(row_input);
		
		if (i < num_rows - 1) {
			row_vbox->add_child(memnew(HSeparator));
		}
	}
}

void SpriteSheetConfigDialog::_on_format_selected(int p_index) {
	String format_name;
	switch (p_index) {
		case 0: format_name = "Humanoid Walking"; break;
		case 1: format_name = "Fire/Flame"; break;
		case 2: format_name = "Custom"; return; // No template for custom
	}
	
	_apply_template(format_name);
}

void SpriteSheetConfigDialog::_apply_template(const String &p_template_name) {
	if (p_template_name == "Humanoid Walking") {
		// Template for humanoid walking animation
		Vector<String> templates;
		templates.push_back("Walking right - leg forward");
		templates.push_back("Walking right - mid stride");
		templates.push_back("Walking right - leg back");
		
		for (int i = 0; i < MIN(templates.size(), row_description_fields.size()); i++) {
			row_description_fields[i]->set_text(templates[i]);
		}
	} else if (p_template_name == "Fire/Flame") {
		// Template for fire/flame animation
		Vector<String> templates;
		templates.push_back("Flame - small flicker");
		templates.push_back("Flame - medium intensity");
		templates.push_back("Flame - large burst");
		
		for (int i = 0; i < MIN(templates.size(), row_description_fields.size()); i++) {
			row_description_fields[i]->set_text(templates[i]);
		}
	}
}

Vector<String> SpriteSheetConfigDialog::get_row_descriptions() const {
	Vector<String> descriptions;
	for (int i = 0; i < row_description_fields.size(); i++) {
		descriptions.push_back(row_description_fields[i]->get_text());
	}
	return descriptions;
}

String SpriteSheetConfigDialog::get_selected_format() const {
	if (!format_dropdown) {
		return "Custom";
	}
	return format_dropdown->get_item_text(format_dropdown->get_selected());
}

SpriteSheetConfigDialog::SpriteSheetConfigDialog() {
	set_title("Configure Sprite Sheet");
	set_hide_on_ok(true);
}

// AnimationPreviewPopup implementation
void AnimationPreviewPopup::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PROCESS: {
			if (is_playing && !animation_frames.is_empty()) {
				frame_timer += get_process_delta_time();
				float frame_duration = 1.0f / fps;
				
				if (frame_timer >= frame_duration) {
					frame_timer = 0.0f;
					current_frame = (current_frame + 1) % animation_frames.size();
					_update_preview_frame();
				}
			}
		} break;
	}
}

void AnimationPreviewPopup::set_animation_frames(const Vector<Ref<ImageTexture>> &p_frames, const String &p_title) {
	animation_frames = p_frames;
	current_frame = 0;
	is_playing = true;
	
	set_title(p_title);
	_update_preview_frame();
	
	if (play_pause_button) {
		play_pause_button->set_text("Pause");
	}
}

void AnimationPreviewPopup::_on_play_pause_pressed() {
	is_playing = !is_playing;
	if (play_pause_button) {
		play_pause_button->set_text(is_playing ? "Pause" : "Play");
	}
}

void AnimationPreviewPopup::_on_fps_changed(double p_value) {
	fps = p_value;
	print_line(vformat("Animation FPS changed to: %.0f", fps));
}

void AnimationPreviewPopup::_update_preview_frame() {
	if (animation_frames.is_empty() || current_frame >= animation_frames.size()) {
		return;
	}
	
	Ref<ImageTexture> frame = animation_frames[current_frame];
	if (frame.is_valid() && preview_display) {
		preview_display->set_texture(frame);
	}
	
	if (frame_info_label) {
		frame_info_label->set_text(vformat("Frame %d/%d (%.0f FPS)", current_frame + 1, animation_frames.size(), fps));
	}
}

AnimationPreviewPopup::AnimationPreviewPopup() {
	set_title("Animation Preview");
	
	VBoxContainer *main_vbox = memnew(VBoxContainer);
	add_child(main_vbox);
	
	// Preview display
	preview_display = memnew(TextureRect);
	preview_display->set_custom_minimum_size(Size2(300, 300));
	preview_display->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	preview_display->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	main_vbox->add_child(preview_display);
	
	// Frame info
	frame_info_label = memnew(Label);
	frame_info_label->set_text("Frame 0/0");
	frame_info_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	main_vbox->add_child(frame_info_label);
	
	// Controls
	HBoxContainer *controls = memnew(HBoxContainer);
	main_vbox->add_child(controls);
	
	play_pause_button = memnew(Button);
	play_pause_button->set_text("Pause");
	play_pause_button->connect("pressed", callable_mp(this, &AnimationPreviewPopup::_on_play_pause_pressed));
	controls->add_child(play_pause_button);
	
	Label *fps_label = memnew(Label);
	fps_label->set_text("FPS:");
	controls->add_child(fps_label);
	
	fps_spinbox = memnew(SpinBox);
	fps_spinbox->set_min(1);
	fps_spinbox->set_max(60);
	fps_spinbox->set_value(8);
	fps_spinbox->connect("value_changed", callable_mp(this, &AnimationPreviewPopup::_on_fps_changed));
	controls->add_child(fps_spinbox);
	
	// Don't call set_min_size() in constructor - it crashes before parent is set
	// The preview_display already has a custom_minimum_size which will size the popup
	
	set_process(true);
}

// ImagePreviewPanel implementation
void ImagePreviewPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
		} break;
	}
}

void ImagePreviewPanel::set_image(const String &p_image_id, const String &p_base64_data, int p_width, int p_height, const String &p_format) {
	current_image_id = p_image_id;
	current_image_base64 = p_base64_data;
	current_image_width = p_width;
	current_image_height = p_height;
	
	// Decode base64 to get PNG/JPEG/WebP bytes
	PackedByteArray image_bytes = CoreBind::Marshalls::get_singleton()->base64_to_raw(p_base64_data);
	
	print_line(vformat("IMAGE_PREVIEW: Decoding image - expected %dx%d, got %d bytes", p_width, p_height, image_bytes.size()));
	
	// Load image from the encoded bytes (PNG/JPEG/WebP format)
	Ref<Image> img;
	img.instantiate();
	
	Error err = OK;
	if (p_format == "png") {
		err = img->load_png_from_buffer(image_bytes);
	} else if (p_format == "jpeg" || p_format == "jpg") {
		err = img->load_jpg_from_buffer(image_bytes);
	} else if (p_format == "webp") {
		err = img->load_webp_from_buffer(image_bytes);
	} else {
		// Try PNG as default
		err = img->load_png_from_buffer(image_bytes);
	}
	
	if (err != OK) {
		print_line(vformat("IMAGE_PREVIEW: Failed to load %s: error %d", p_format, err));
		image_info_label->set_text(vformat("Error loading image: %d", err));
		image_info_label->set_modulate(Color(1.0, 0.5, 0.5));
		return;
	}
	
	print_line(vformat("IMAGE_PREVIEW: Loaded image successfully - actual size: %dx%d", img->get_width(), img->get_height()));
	
	// Create texture from the loaded image
	Ref<ImageTexture> texture = ImageTexture::create_from_image(img);
	image_display->set_texture(texture);
	image_display->set_visible(true);
	
	// Update info label
	image_info_label->set_text(vformat("Image: %s (%dx%d px, %s)", p_image_id, img->get_width(), img->get_height(), p_format));
	image_info_label->set_modulate(Color(0.7, 1.0, 0.7));
	
	// Show chat interface
	chat_input->set_visible(true);
	send_chat_button->set_visible(true);
	generate_spritesheet_button->set_visible(true);
	
	print_line(vformat("IMAGE_PREVIEW: Successfully displayed %s (%dx%d)", p_image_id, img->get_width(), img->get_height()));
}

void ImagePreviewPanel::clear_image() {
	image_display->set_texture(Ref<Texture2D>());
	image_display->set_visible(false);
	current_image_id = String();
	current_image_base64 = String();
	
	// Hide chat interface
	chat_input->set_visible(false);
	send_chat_button->set_visible(false);
	generate_spritesheet_button->set_visible(false);
	
	// Clear chat history
	while (chat_history->get_child_count() > 0) {
		chat_history->get_child(0)->queue_free();
	}
	
	image_info_label->set_text("No image loaded");
}

void ImagePreviewPanel::_on_send_chat_pressed() {
	String message = chat_input->get_text().strip_edges();
	if (message.is_empty() || current_image_id.is_empty()) {
		return;
	}
	
	print_line(vformat("IMAGE_CHAT: User message: %s", message));
	
	// Add user message to chat history
	Label *user_msg = memnew(Label);
	user_msg->set_text("You: " + message);
	user_msg->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	chat_history->add_child(user_msg);
	
	// Clear input
	chat_input->set_text("");
	
	// Determine backend URL
	String backend_url;
	bool is_dev = false;
	if (OS::get_singleton()->has_environment("IS_DEV")) {
		is_dev = OS::get_singleton()->get_environment("IS_DEV").to_lower() == "true";
	} else if (OS::get_singleton()->has_environment("DEV_MODE")) {
		is_dev = OS::get_singleton()->get_environment("DEV_MODE").to_lower() == "true";
	}
	
	if (is_dev) {
		backend_url = "http://localhost:3031";
	} else {
		backend_url = OS::get_singleton()->get_environment("IMAGEN_SERVICE_URL");
		if (backend_url.is_empty()) {
			backend_url = "https://PLACEHOLDER-UPDATE-ME.run.app";
		}
	}
	
	// Use unified process endpoint
	String endpoint = backend_url + "/api/image/process";
	
	// Build payload with unified format
	Dictionary payload;
	payload["prompt"] = message;
	
	// Add current image using unified format
	Array images;
	Dictionary img_spec;
	img_spec["id"] = current_image_id;
	img_spec["format"] = "png";
	images.push_back(img_spec);
	payload["images"] = images;
	
	payload["size"] = "1024x1024";
	payload["quality"] = "high";
	payload["output_format"] = "png";
	
	// Create HTTP request
	HTTPRequest *http_request = memnew(HTTPRequest);
	add_child(http_request);
	http_request->connect("request_completed", callable_mp(this, &ImagePreviewPanel::_on_chat_response));
	
	Ref<JSON> json;
	json.instantiate();
	String json_string = json->stringify(payload);
	
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	Error err = http_request->request(endpoint, headers, HTTPClient::METHOD_POST, json_string);
	
	if (err != OK) {
		print_line("IMAGE_CHAT: Failed to send edit request");
		return;
	}
	
	// Disable send button while processing
	send_chat_button->set_disabled(true);
	send_chat_button->set_text("Processing...");
	
	print_line("IMAGE_CHAT: Edit request sent to unified endpoint");
}

void ImagePreviewPanel::_on_chat_response(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	// Re-enable button
	send_chat_button->set_disabled(false);
	send_chat_button->set_text("Send");
	
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_response_code != 200) {
		Label *error_msg = memnew(Label);
		error_msg->set_text("AI: Error generating image");
		error_msg->set_modulate(Color(1.0, 0.5, 0.5));
		chat_history->add_child(error_msg);
		return;
	}
	
	// Parse response
	String response_string = String::utf8((const char *)p_body.ptr(), p_body.size());
	
	Ref<JSON> json;
	json.instantiate();
	Error err = json->parse(response_string);
	
	if (err != OK) {
		print_line("IMAGE_CHAT: Failed to parse response");
		return;
	}
	
	Dictionary response = json->get_data();
	
	if (response.get("success", false)) {
		String new_image_id = response.get("image_id", "");
		String new_image_data = response.get("image_data", "");
		int width = response.get("width", 0);
		int height = response.get("height", 0);
		String format = response.get("format", "png");
		
		// Update displayed image
		set_image(new_image_id, new_image_data, width, height, format);
		
		// Add AI response to chat
		Label *ai_msg = memnew(Label);
		ai_msg->set_text(vformat("AI: Updated image! (%dx%d)", width, height));
		ai_msg->set_modulate(Color(0.7, 1.0, 0.7));
		chat_history->add_child(ai_msg);
		
		print_line(vformat("IMAGE_CHAT: Image updated to %s", new_image_id));
	}
}

void ImagePreviewPanel::_on_generate_spritesheet_pressed() {
	if (current_image_id.is_empty()) {
		EditorNode::get_singleton()->show_warning("No image loaded to generate sprite sheet from");
		return;
	}
	
	if (!parent_editor) {
		print_line("ERROR: Parent editor not set");
		return;
	}
	
	// Open configuration dialog
	if (parent_editor->spritesheet_config_dialog) {
		int grid_height = (int)parent_editor->grid_height->get_value();
		parent_editor->spritesheet_config_dialog->setup(grid_height, parent_editor);
		parent_editor->spritesheet_config_dialog->popup_centered();
	}
}

ImagePreviewPanel::ImagePreviewPanel() {
	set_h_size_flags(Control::SIZE_EXPAND_FILL);
	set_v_size_flags(Control::SIZE_EXPAND_FILL);
	
	// Image display area
	PanelContainer *image_container = memnew(PanelContainer);
	image_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(image_container);
	
	CenterContainer *center = memnew(CenterContainer);
	image_container->add_child(center);
	
	image_display = memnew(TextureRect);
	image_display->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	image_display->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	image_display->set_custom_minimum_size(Size2(400, 400));
	image_display->set_visible(false);
	center->add_child(image_display);
	
	// Image info
	image_info_label = memnew(Label);
	image_info_label->set_text("No image loaded");
	image_info_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	add_child(image_info_label);
	
	add_child(memnew(HSeparator));
	
	// Chat history (scrollable)
	Label *chat_label = memnew(Label);
	chat_label->set_text("Edit with AI:");
	add_child(chat_label);
	
	ScrollContainer *chat_scroll = memnew(ScrollContainer);
	chat_scroll->set_custom_minimum_size(Size2(0, 150));
	chat_scroll->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(chat_scroll);
	
	chat_history = memnew(VBoxContainer);
	chat_scroll->add_child(chat_history);
	
	// Chat input
	chat_input = memnew(TextEdit);
	chat_input->set_custom_minimum_size(Size2(0, 60));
	chat_input->set_placeholder("Ask AI to edit the image... (e.g., 'make it brighter', 'add a hat')");
	chat_input->set_visible(false);
	add_child(chat_input);
	
	// Chat buttons
	HBoxContainer *chat_buttons = memnew(HBoxContainer);
	add_child(chat_buttons);
	
	send_chat_button = memnew(Button);
	send_chat_button->set_text("Send");
	send_chat_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	send_chat_button->connect("pressed", callable_mp(this, &ImagePreviewPanel::_on_send_chat_pressed));
	send_chat_button->set_visible(false);
	chat_buttons->add_child(send_chat_button);
	
	add_child(memnew(HSeparator));
	
	// Generate sprite sheet button
	generate_spritesheet_button = memnew(Button);
	generate_spritesheet_button->set_text("Generate Sprite Sheet from This Image");
	generate_spritesheet_button->connect("pressed", callable_mp(this, &ImagePreviewPanel::_on_generate_spritesheet_pressed));
	generate_spritesheet_button->set_visible(false);
	add_child(generate_spritesheet_button);
}

// SpriteSheetGridDisplay implementation
void SpriteSheetGridDisplay::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			Size2 size = get_size();
			
			// Fixed large cell size for better visuals and stable sizing
			const int CELL_SIZE = 128; // Large, fixed cell size
			const int CELL_PADDING = 6; // Padding between cells
			
			// Calculate actual grid size
			int total_grid_width = (CELL_SIZE + CELL_PADDING) * grid_width + CELL_PADDING;
			int total_grid_height = (CELL_SIZE + CELL_PADDING) * grid_height + CELL_PADDING;
			
			// Center the grid
			int offset_x = (size.width - total_grid_width) / 2;
			int offset_y = (size.height - total_grid_height) / 2;
			
			// Draw overall background panel with rounded corners feel
			Color panel_bg = Color(0.12, 0.12, 0.14, 1.0);
			draw_rect(Rect2(offset_x, offset_y, total_grid_width, total_grid_height), panel_bg, true);
			
			// Draw cells
			for (int y = 0; y < grid_height; y++) {
				for (int x = 0; x < grid_width; x++) {
					int cell_x = offset_x + CELL_PADDING + x * (CELL_SIZE + CELL_PADDING);
					int cell_y = offset_y + CELL_PADDING + y * (CELL_SIZE + CELL_PADDING);
					
					Rect2 cell_rect = Rect2(cell_x, cell_y, CELL_SIZE, CELL_SIZE);
					
					// Check if this cell has an image
					bool has_image = false;
					if (y < cell_textures.size() && x < cell_textures[y].size()) {
						Ref<ImageTexture> tex = cell_textures[y][x];
						if (tex.is_valid()) {
							has_image = true;
							// Draw the image texture
							draw_texture_rect(tex, cell_rect, false);
						}
					}
					
					if (!has_image) {
						// Draw cell background with slight gradient effect
						Color cell_bg = Color(0.18, 0.18, 0.20, 1.0);
						draw_rect(cell_rect, cell_bg, true);
						
						// Draw subtle inner shadow effect
						Color shadow_color = Color(0.08, 0.08, 0.10, 0.5);
						draw_line(cell_rect.position, cell_rect.position + Vector2(CELL_SIZE, 0), shadow_color, 1);
						draw_line(cell_rect.position, cell_rect.position + Vector2(0, CELL_SIZE), shadow_color, 1);
						
						// Draw placeholder text in center
						Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Label"));
						String placeholder = "Empty";
						Vector2 placeholder_size = font->get_string_size(placeholder, HORIZONTAL_ALIGNMENT_CENTER, -1, 12);
						Vector2 placeholder_pos = cell_rect.position + (cell_rect.size - placeholder_size) / 2;
						draw_string(font, placeholder_pos, placeholder, HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.35, 0.35, 0.4, 0.6));
					}
					
					// Always draw cell border on top
					Color border_color = Color(0.28, 0.28, 0.32, 1.0);
					draw_rect(cell_rect, border_color, false, 1);
					
					// Draw cell index in top-left corner
					String cell_text = String::num(y * grid_width + x + 1);
					Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Label"));
					int font_size = 14;
					Vector2 text_pos = cell_rect.position + Vector2(8, 20);
					
					// Draw text shadow for better readability
					draw_string(font, text_pos + Vector2(1, 1), cell_text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, Color(0, 0, 0, 0.5));
					draw_string(font, text_pos, cell_text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, Color(0.5, 0.5, 0.55, 0.8));
				}
			}
			
			// Draw outer border with accent
			Color outer_border = Color(0.35, 0.35, 0.40, 1.0);
			draw_rect(Rect2(offset_x, offset_y, total_grid_width, total_grid_height), outer_border, false, 2);
		} break;
	}
}

void SpriteSheetGridDisplay::set_grid_size(int p_width, int p_height) {
	grid_width = CLAMP(p_width, 1, 64);
	grid_height = CLAMP(p_height, 1, 64);
	
	// Resize texture storage
	cell_textures.resize(grid_height);
	for (int i = 0; i < grid_height; i++) {
		cell_textures.write[i].resize(grid_width);
	}
	
	queue_redraw();
	update_minimum_size();
}

void SpriteSheetGridDisplay::set_cell_image(int p_row, int p_col, const Ref<ImageTexture> &p_texture) {
	if (p_row < 0 || p_row >= grid_height || p_col < 0 || p_col >= grid_width) {
		print_line(vformat("GRID_DISPLAY: Invalid cell [%d,%d]", p_row, p_col));
		return;
	}
	
	cell_textures.write[p_row].write[p_col] = p_texture;
	queue_redraw();
	print_line(vformat("GRID_DISPLAY: Set cell [%d,%d] texture", p_row, p_col));
}

void SpriteSheetGridDisplay::clear_all_cells() {
	for (int row = 0; row < cell_textures.size(); row++) {
		for (int col = 0; col < cell_textures[row].size(); col++) {
			cell_textures.write[row].write[col] = Ref<ImageTexture>();
		}
	}
	queue_redraw();
}

Vector<Ref<ImageTexture>> SpriteSheetGridDisplay::get_row_frames(int p_row) const {
	Vector<Ref<ImageTexture>> frames;
	if (p_row >= 0 && p_row < cell_textures.size()) {
		for (int col = 0; col < cell_textures[p_row].size(); col++) {
			Ref<ImageTexture> tex = cell_textures[p_row][col];
			if (tex.is_valid()) {
				frames.push_back(tex);
			}
		}
	}
	return frames;
}

Vector<Ref<ImageTexture>> SpriteSheetGridDisplay::get_all_frames() const {
	Vector<Ref<ImageTexture>> frames;
	// Traverse row by row, column by column
	for (int row = 0; row < cell_textures.size(); row++) {
		for (int col = 0; col < cell_textures[row].size(); col++) {
			Ref<ImageTexture> tex = cell_textures[row][col];
			if (tex.is_valid()) {
				frames.push_back(tex);
			}
		}
	}
	return frames;
}

Size2 SpriteSheetGridDisplay::get_minimum_size() const {
	// Fixed large cell size for stable layout
	const int CELL_SIZE = 128;
	const int CELL_PADDING = 6;
	int total_width = (CELL_SIZE + CELL_PADDING) * grid_width + CELL_PADDING;
	int total_height = (CELL_SIZE + CELL_PADDING) * grid_height + CELL_PADDING;
	// Add some breathing room around the grid
	return Size2(total_width + 100, total_height + 100);
}

SpriteSheetGridDisplay::SpriteSheetGridDisplay() {
	set_custom_minimum_size(Size2(600, 600));
	
	// Initialize texture storage
	cell_textures.resize(grid_height);
	for (int i = 0; i < grid_height; i++) {
		cell_textures.write[i].resize(grid_width);
	}
}

// DesignStudio2DEditor implementation

DesignStudio2DEditor::DesignStudio2DEditor() {
	// Build main UI once in the constructor
	// Row: left controls, right canvas
	main_split = memnew(HSplitContainer);
	add_child(main_split);
	main_split->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);

	// Left panel (scrollable controls)
	ScrollContainer *left_scroll = memnew(ScrollContainer);
	left_scroll->set_custom_minimum_size(Size2(300 * EDSCALE, 0));
	left_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	main_split->add_child(left_scroll);

	left_panel = memnew(VBoxContainer);
	left_scroll->add_child(left_panel);
	left_panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	// Add margin for better spacing
	MarginContainer *margin = memnew(MarginContainer);
	margin->add_theme_constant_override("margin_left", 10 * EDSCALE);
	margin->add_theme_constant_override("margin_right", 10 * EDSCALE);
	margin->add_theme_constant_override("margin_top", 10 * EDSCALE);
	margin->add_theme_constant_override("margin_bottom", 10 * EDSCALE);
	left_panel->add_child(margin);

	VBoxContainer *content_vbox = memnew(VBoxContainer);
	content_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	margin->add_child(content_vbox);

	// Title: Sprite Sheet Creation
	Label *title = memnew(Label);
	title->set_text("Sprite Sheet Creation");
	title->add_theme_font_size_override("font_size", 18 * EDSCALE);
	content_vbox->add_child(title);

	content_vbox->add_child(memnew(HSeparator));

	// Grid Size section
	Label *grid_label = memnew(Label);
	grid_label->set_text("Grid Size:");
	content_vbox->add_child(grid_label);

	HBoxContainer *grid_size_hbox = memnew(HBoxContainer);
	content_vbox->add_child(grid_size_hbox);

	Label *width_label = memnew(Label);
	width_label->set_text("Width:");
	grid_size_hbox->add_child(width_label);

	grid_width = memnew(SpinBox);
	grid_width->set_min(1);
	grid_width->set_max(64);
	grid_width->set_value(4);
	grid_width->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	grid_width->connect("value_changed", callable_mp(this, &DesignStudio2DEditor::_on_grid_size_changed));
	grid_size_hbox->add_child(grid_width);

	Label *height_label = memnew(Label);
	height_label->set_text("Height:");
	grid_size_hbox->add_child(height_label);

	grid_height = memnew(SpinBox);
	grid_height->set_min(1);
	grid_height->set_max(64);
	grid_height->set_value(3);
	grid_height->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	grid_height->connect("value_changed", callable_mp(this, &DesignStudio2DEditor::_on_grid_size_changed));
	grid_size_hbox->add_child(grid_height);

	content_vbox->add_child(memnew(HSeparator));

	// Seed Image section
	Label *seed_label = memnew(Label);
	seed_label->set_text("Seed Image:");
	content_vbox->add_child(seed_label);

	import_seed_button = memnew(Button);
	import_seed_button->set_text("Import Seed Image...");
	import_seed_button->connect("pressed", callable_mp(this, &DesignStudio2DEditor::_on_import_seed_pressed));
	content_vbox->add_child(import_seed_button);

	// Create Image from Text button
	create_image_button = memnew(Button);
	create_image_button->set_text("Create Image from Text");
	create_image_button->connect("pressed", callable_mp(this, &DesignStudio2DEditor::_on_create_image_pressed));
	content_vbox->add_child(create_image_button);

	seed_image_label = memnew(Label);
	seed_image_label->set_text("No image selected");
	seed_image_label->set_modulate(Color(0.7, 0.7, 0.7));
	content_vbox->add_child(seed_image_label);

	content_vbox->add_child(memnew(HSeparator));

	// Description text box
	Label *desc_label = memnew(Label);
	desc_label->set_text("Describe what your sprite sheet should look like:");
	desc_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	content_vbox->add_child(desc_label);

	description_text = memnew(TextEdit);
	description_text->set_custom_minimum_size(Size2(0, 120 * EDSCALE));
	description_text->set_placeholder("e.g., A pixel art character walking animation with 4 frames, facing right, vibrant colors...");
	description_text->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);
	content_vbox->add_child(description_text);

	content_vbox->add_child(memnew(HSeparator));

	// Image Model dropdown
	Label *model_label = memnew(Label);
	model_label->set_text("Image Model:");
	content_vbox->add_child(model_label);

	model_dropdown = memnew(OptionButton);
	model_dropdown->add_item("Nano-Banana");
	model_dropdown->add_item("GPT-Image");
	model_dropdown->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	content_vbox->add_child(model_dropdown);

	content_vbox->add_child(memnew(HSeparator));

	// Advanced button
	advanced_button = memnew(Button);
	advanced_button->set_text("Advanced ▶");
	advanced_button->set_button_icon(nullptr);
	advanced_button->connect("pressed", callable_mp(this, &DesignStudio2DEditor::_on_advanced_toggled));
	content_vbox->add_child(advanced_button);

	// Advanced options container (initially hidden)
	advanced_options = memnew(VBoxContainer);
	advanced_options->set_visible(false);
	content_vbox->add_child(advanced_options);

	// Placeholder for future advanced options
	Label *advanced_placeholder = memnew(Label);
	advanced_placeholder->set_text("Advanced options coming soon...");
	advanced_placeholder->set_modulate(Color(0.6, 0.6, 0.6));
	advanced_options->add_child(advanced_placeholder);

	// File dialog for seed image
	seed_image_dialog = memnew(FileDialog);
	seed_image_dialog->set_file_mode(FileDialog::FILE_MODE_OPEN_FILE);
	seed_image_dialog->set_access(FileDialog::ACCESS_FILESYSTEM); // Allow access to entire filesystem
	seed_image_dialog->set_title("Select Seed Image");
	seed_image_dialog->add_filter("*.png", "PNG Images");
	seed_image_dialog->add_filter("*.jpg,*.jpeg", "JPEG Images");
	seed_image_dialog->add_filter("*.webp", "WebP Images");
	seed_image_dialog->add_filter("*.bmp", "BMP Images");

	// Set initial directory to user's home directory or Pictures folder
	String home_dir = OS::get_singleton()->get_system_dir(OS::SYSTEM_DIR_PICTURES);
	if (home_dir.is_empty()) {
		home_dir = OS::get_singleton()->get_system_dir(OS::SYSTEM_DIR_DOCUMENTS);
	}
	if (home_dir.is_empty()) {
		home_dir = OS::get_singleton()->get_environment("HOME"); // Fallback to home directory
		if (home_dir.is_empty()) {
			home_dir = OS::get_singleton()->get_environment("USERPROFILE"); // Windows fallback
		}
	}
	if (!home_dir.is_empty()) {
		seed_image_dialog->set_current_dir(home_dir);
	}

	seed_image_dialog->connect("file_selected", callable_mp(this, &DesignStudio2DEditor::_on_seed_image_selected));
	add_child(seed_image_dialog);

	// Right side - Canvas area that switches between image preview and grid
	canvas_area = memnew(PanelContainer);
	main_split->add_child(canvas_area);
	canvas_area->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	canvas_area->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	// Container to hold both views
	VBoxContainer *view_container = memnew(VBoxContainer);
	canvas_area->add_child(view_container);
	view_container->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);

	// Add image preview panel (shown by default)
	MarginContainer *preview_margin = memnew(MarginContainer);
	preview_margin->add_theme_constant_override("margin_left", 20 * EDSCALE);
	preview_margin->add_theme_constant_override("margin_right", 20 * EDSCALE);
	preview_margin->add_theme_constant_override("margin_top", 20 * EDSCALE);
	preview_margin->add_theme_constant_override("margin_bottom", 20 * EDSCALE);
	view_container->add_child(preview_margin);

	image_preview = memnew(ImagePreviewPanel);
	image_preview->set_parent_editor(this);
	preview_margin->add_child(image_preview);

	// Add sprite sheet grid display (hidden initially)
	grid_scroll_container = memnew(ScrollContainer);
	grid_scroll_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	grid_scroll_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	grid_scroll_container->set_visible(false);  // Hidden by default
	view_container->add_child(grid_scroll_container);

	CenterContainer *grid_center = memnew(CenterContainer);
	grid_scroll_container->add_child(grid_center);
	grid_center->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	grid_center->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	grid_display = memnew(SpriteSheetGridDisplay);
	grid_center->add_child(grid_display);
	grid_display->set_grid_size(4, 3);

	// Create sprite sheet configuration dialog
	spritesheet_config_dialog = memnew(SpriteSheetConfigDialog);
	add_child(spritesheet_config_dialog);
	spritesheet_config_dialog->connect("confirmed", callable_mp(this, &DesignStudio2DEditor::_on_spritesheet_config_confirmed));

	// Initial split offset
	main_split->set_split_offset(320 * EDSCALE);
}
void DesignStudio2DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			// Set a minimum size for the panel
			set_custom_minimum_size(Size2(200, 200) * EDSCALE);
		} break;
		case NOTIFICATION_PROCESS: {
			// Poll for streaming sprite sheet generation (like AI chat dock does)
			if (spritesheet_in_progress && spritesheet_http.is_valid()) {
				_poll_spritesheet_stream();
			}
		} break;
	}
}

void DesignStudio2DEditor::_on_import_seed_pressed() {
	seed_image_dialog->popup_file_dialog();
}

void DesignStudio2DEditor::_on_seed_image_selected(const String &p_path) {
	current_seed_image_path = p_path;
	seed_image_label->set_text("Seed Image: " + p_path.get_file());
	seed_image_label->set_modulate(Color(0.7, 1.0, 0.7));  // Green tint to show success
	
	// Load image and convert to base64 for API
	Ref<Image> img = Image::load_from_file(p_path);
	if (img.is_valid()) {
		Vector<uint8_t> png_data = img->save_png_to_buffer();
		current_seed_image_base64 = CoreBind::Marshalls::get_singleton()->raw_to_base64(png_data);
		print_line(vformat("SEED_IMAGE: Loaded and encoded %d bytes", png_data.size()));
		
		// Display the seed image immediately in the preview panel
		if (image_preview) {
			image_preview->set_image("seed_" + p_path.get_file(), current_seed_image_base64, img->get_width(), img->get_height(), "png");
		}
	} else {
		print_line("SEED_IMAGE: Failed to load image");
		current_seed_image_base64 = String();
	}
}

void DesignStudio2DEditor::_on_advanced_toggled() {
	bool is_visible = advanced_options->is_visible();
	advanced_options->set_visible(!is_visible);
	
	// Update button text based on state
	if (!is_visible) {
		advanced_button->set_text("Advanced ▼");
	} else {
		advanced_button->set_text("Advanced ▶");
	}
}

void DesignStudio2DEditor::_on_grid_size_changed(double p_value) {
	// Update the grid display when either width or height changes
	if (grid_display) {
		grid_display->set_grid_size((int)grid_width->get_value(), (int)grid_height->get_value());
	}
}

void DesignStudio2DEditor::_on_create_image_pressed() {
	String description = description_text->get_text();
	
	if (description.is_empty()) {
		EditorNode::get_singleton()->show_warning("Please enter a description for the image");
		return;
	}
	
	print_line("CREATE_IMAGE: Starting single image generation...");
	print_line(vformat("CREATE_IMAGE: Description: %s", description));
	
	// Generate a single image first
	_generate_single_image();
}

void DesignStudio2DEditor::_generate_single_image() {
	String description = description_text->get_text();
	int selected_model = model_dropdown->get_selected();
	
	if (description.is_empty()) {
		EditorNode::get_singleton()->show_warning("Please enter a description for the image");
		return;
	}
	
	// Determine backend URL (dev vs production)
	String backend_url;
	bool is_dev = false;
	if (OS::get_singleton()->has_environment("IS_DEV")) {
		is_dev = OS::get_singleton()->get_environment("IS_DEV").to_lower() == "true";
	} else if (OS::get_singleton()->has_environment("DEV_MODE")) {
		is_dev = OS::get_singleton()->get_environment("DEV_MODE").to_lower() == "true";
	}
	
	if (is_dev) {
		backend_url = "http://localhost:3031";
		print_line("IMAGEN: Using local development server at http://localhost:3031");
	} else {
		backend_url = OS::get_singleton()->get_environment("IMAGEN_SERVICE_URL");
		if (backend_url.is_empty()) {
			backend_url = "https://PLACEHOLDER-UPDATE-ME.run.app";
		}
		print_line(vformat("IMAGEN: Using production server at %s", backend_url));
	}
	
	// Use unified process endpoint
	String endpoint = backend_url + "/api/image/process";
	
	// Build payload using unified format
	Dictionary payload;
	payload["prompt"] = description;
	
	// Add seed image if available (using unified images array format)
	if (!current_seed_image_base64.is_empty()) {
		Array images;
		Dictionary seed_img;
		seed_img["data"] = current_seed_image_base64;
		seed_img["format"] = "png";
		images.push_back(seed_img);
		payload["images"] = images;
		print_line("IMAGE_GEN: Including seed image for editing");
	} else {
		print_line("IMAGE_GEN: Generating new image from text");
	}
	
	// Add style based on selected model
	Array model_options;
	model_options.push_back("pixel art");
	model_options.push_back("hand-drawn illustration");
	model_options.push_back("photorealistic");
	
	if (selected_model >= 0 && selected_model < model_options.size()) {
		payload["style"] = model_options[selected_model];
	}
	
	payload["size"] = "1024x1024";
	payload["quality"] = "high";
	payload["output_format"] = "png";
	
	print_line(vformat("IMAGE_GEN: Calling unified endpoint %s", endpoint));
	print_line(vformat("IMAGE_GEN: Description: %s", description));
	
	// Create HTTP request
	HTTPRequest *http_request = memnew(HTTPRequest);
	add_child(http_request);
	
	// Connect completion signal
	http_request->connect("request_completed", callable_mp(this, &DesignStudio2DEditor::_on_single_image_generated));
	
	// Convert payload to JSON string
	Ref<JSON> json;
	json.instantiate();
	String json_string = json->stringify(payload);
	
	// Set headers
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	// Make request
	Error err = http_request->request(endpoint, headers, HTTPClient::METHOD_POST, json_string);
	
	if (err != OK) {
		print_line(vformat("IMAGE_GEN: HTTP request failed with error %d", err));
		EditorNode::get_singleton()->show_warning("Failed to start image generation request");
		http_request->queue_free();
		return;
	}
	
	print_line("IMAGE_GEN: Request sent, waiting for response...");
	
	// Disable button while generating
	create_image_button->set_disabled(true);
	create_image_button->set_text("Generating...");
}

void DesignStudio2DEditor::_on_single_image_generated(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	// Re-enable button
	create_image_button->set_disabled(false);
	create_image_button->set_text("Create Image from Text");
	
	print_line(vformat("IMAGE_RESPONSE: Result: %d, Response code: %d", p_result, p_response_code));
	
	if (p_result != HTTPRequest::RESULT_SUCCESS) {
		EditorNode::get_singleton()->show_warning(vformat("Image generation request failed: Result %d", p_result));
		return;
	}
	
	if (p_response_code != 200) {
		EditorNode::get_singleton()->show_warning(vformat("Image generation failed: HTTP %d", p_response_code));
		return;
	}
	
	// Parse response
	String response_string = String::utf8((const char *)p_body.ptr(), p_body.size());
	
	Ref<JSON> json;
	json.instantiate();
	Error err = json->parse(response_string);
	
	if (err != OK) {
		print_line(vformat("IMAGE_RESPONSE: JSON parse error at line %d: %s", json->get_error_line(), json->get_error_message()));
		EditorNode::get_singleton()->show_warning("Failed to parse image generation response");
		return;
	}
	
	Dictionary response = json->get_data();
	
	if (!response.get("success", false)) {
		String error = response.get("error", "Unknown error");
		print_line(vformat("IMAGE_RESPONSE: Generation failed: %s", error));
		EditorNode::get_singleton()->show_warning(vformat("Image generation failed: %s", error));
		return;
	}
	
	// Extract image data
	String image_id = response.get("image_id", "");
	String image_data_b64 = response.get("image_data", "");
	int width = response.get("width", 0);
	int height = response.get("height", 0);
	String format = response.get("format", "png");
	
	print_line(vformat("IMAGE_SUCCESS: Generated %s (%dx%d)", image_id, width, height));
	
	// Display the generated image in the preview panel
	if (image_preview) {
		image_preview->set_image(image_id, image_data_b64, width, height, format);
	}
	
	// Update label
	seed_image_label->set_text(vformat("Generated: %s (%dx%d)", image_id, width, height));
	seed_image_label->set_modulate(Color(0.7, 1.0, 0.7));
}

void DesignStudio2DEditor::_start_progressive_spritesheet_generation(const String &p_description) {
	if (!image_preview || image_preview->current_image_id.is_empty()) {
		EditorNode::get_singleton()->show_warning("No seed image available");
		return;
	}
	
	int g_width = (int)grid_width->get_value();
	int g_height = (int)grid_height->get_value();
	int selected_model = model_dropdown->get_selected();
	
	// Initialize cell storage
	spritesheet_grid_width = g_width;
	spritesheet_grid_height = g_height;
	spritesheet_cells.clear();
	spritesheet_cells.resize(g_height);
	for (int i = 0; i < g_height; i++) {
		spritesheet_cells.write[i].resize(g_width);
	}
	
	// Setup grid display
	if (grid_display) {
		grid_display->set_grid_size(g_width, g_height);
		grid_display->clear_all_cells();
	}
	
	// Hide image preview, show grid
	if (image_preview && image_preview->get_parent()) {
		Control *parent_control = Object::cast_to<Control>(image_preview->get_parent());
		if (parent_control) {
			parent_control->set_visible(false);  // Hide the preview margin container
		}
	}
	if (grid_scroll_container) {
		grid_scroll_container->set_visible(true);
	}
	
	// Determine backend URL
	String backend_url;
	bool is_dev = false;
	if (OS::get_singleton()->has_environment("IS_DEV")) {
		is_dev = OS::get_singleton()->get_environment("IS_DEV").to_lower() == "true";
	} else if (OS::get_singleton()->has_environment("DEV_MODE")) {
		is_dev = OS::get_singleton()->get_environment("DEV_MODE").to_lower() == "true";
	}
	
	if (is_dev) {
		backend_url = "http://localhost:3031";
	} else {
		backend_url = OS::get_singleton()->get_environment("IMAGEN_SERVICE_URL");
		if (backend_url.is_empty()) {
			backend_url = "https://PLACEHOLDER-UPDATE-ME.run.app";
		}
	}
	
	String endpoint = backend_url + "/api/spritesheet/generate_progressive";
	
	// Build payload
	Dictionary payload;
	payload["prompt"] = p_description.is_empty() ? description_text->get_text() : p_description;
	// Include row_descriptions as structured data to drive consistent per-row generation
	if (current_row_descriptions.size() > 0) {
		Array rows;
		for (int i = 0; i < current_row_descriptions.size(); i++) {
			rows.push_back(current_row_descriptions[i]);
		}
		payload["row_descriptions"] = rows;
	}
	payload["seed_image_id"] = image_preview->current_image_id;
	payload["grid_width"] = g_width;
	payload["grid_height"] = g_height;
	
	// Add style
	Array model_options;
	model_options.push_back("pixel art");
	model_options.push_back("hand-drawn illustration");
	model_options.push_back("photorealistic");
	
	if (selected_model >= 0 && selected_model < model_options.size()) {
		payload["style"] = model_options[selected_model];
	}
	
	print_line(vformat("PROGRESSIVE_SPRITESHEET: Starting %dx%d generation", g_width, g_height));
	print_line(vformat("PROGRESSIVE_SPRITESHEET: Seed: %s", image_preview->current_image_id));
	
	// Parse endpoint URL
	String host = backend_url;
	int port = 3031;
	
	if (host.begins_with("https://")) {
		host = host.substr(8);
		port = 443;
	} else if (host.begins_with("http://")) {
		host = host.substr(7);
	}
	
	String path = "/api/spritesheet/generate_progressive";
	if (host.find(":") != -1) {
		int colon_pos = host.find(":");
		String port_str = host.substr(colon_pos + 1);
		port = port_str.to_int();
		host = host.substr(0, colon_pos);
	}
	
	print_line(vformat("PROGRESSIVE_SPRITESHEET: Connecting to %s:%d%s", host, port, path));
	
	// Create HTTPClient for streaming
	spritesheet_http = HTTPClient::create();
	spritesheet_http->set_read_chunk_size(4096);
	
	// Convert payload to JSON  
	Ref<JSON> json;
	json.instantiate();
	String json_body = json->stringify(payload);
	
	// Store headers and body for sending after connection
	spritesheet_pending_headers.clear();
	spritesheet_pending_headers.push_back("Host: " + host);
	spritesheet_pending_headers.push_back("Content-Type: application/json");
	spritesheet_pending_headers.push_back("Content-Length: " + String::num_int64(json_body.length()));
	spritesheet_pending_headers.push_back("Accept: application/x-ndjson");
	spritesheet_pending_body = json_body;
	
	// Connect to server
	Error err = spritesheet_http->connect_to_host(host, port);
	if (err != OK) {
		print_line(vformat("PROGRESSIVE_SPRITESHEET: Connection failed with error %d", err));
		EditorNode::get_singleton()->show_warning("Failed to connect to imagen service");
		spritesheet_http.unref();
		return;
	}
	
	print_line("PROGRESSIVE_SPRITESHEET: Connecting... will stream progress updates live!");
	spritesheet_response_buffer = "";
	spritesheet_in_progress = true;
	
	// Enable processing to poll for chunks
	set_process(true);
}

void DesignStudio2DEditor::_poll_spritesheet_stream() {
	if (!spritesheet_http.is_valid()) {
		return;
	}
	
	spritesheet_http->poll();
	HTTPClient::Status status = spritesheet_http->get_status();
	
	// Handle connection states
	if (status == HTTPClient::STATUS_CONNECTING || status == HTTPClient::STATUS_RESOLVING) {
		return; // Still connecting
	}
	
	if (status == HTTPClient::STATUS_CONNECTED && !spritesheet_pending_body.is_empty()) {
		// Send the request
		PackedByteArray body_bytes = spritesheet_pending_body.to_utf8_buffer();
		Error err = spritesheet_http->request(HTTPClient::METHOD_POST, "/api/spritesheet/generate_progressive", 
		                                       spritesheet_pending_headers, body_bytes.ptr(), body_bytes.size());
		if (err != OK) {
			print_line(vformat("SPRITESHEET_STREAM: Request send failed: %d", err));
			spritesheet_in_progress = false;
			set_process(false);
			return;
		}
		spritesheet_pending_body = ""; // Clear after sending
		print_line("SPRITESHEET_STREAM: Request sent, awaiting response...");
	}
	
	if (status == HTTPClient::STATUS_REQUESTING) {
		return; // Still sending request
	}
	
	if (status == HTTPClient::STATUS_BODY) {
		// Read response chunks as they arrive
		PackedByteArray chunk = spritesheet_http->read_response_body_chunk();
		if (chunk.size() > 0) {
			// Process chunk immediately
			String chunk_str = String::utf8((const char *)chunk.ptr(), chunk.size());
			spritesheet_response_buffer += chunk_str;
			
			// Process complete NDJSON lines
			int newline_pos;
			while ((newline_pos = spritesheet_response_buffer.find("\n")) != -1) {
				String line = spritesheet_response_buffer.substr(0, newline_pos);
				spritesheet_response_buffer = spritesheet_response_buffer.substr(newline_pos + 1);
				
				if (!line.strip_edges().is_empty()) {
					_process_spritesheet_ndjson_line(line);
				}
			}
		}
	}
	
	if (status == HTTPClient::STATUS_DISCONNECTED || status == HTTPClient::STATUS_CONNECTION_ERROR) {
		print_line("SPRITESHEET_STREAM: Connection closed or errored");
		spritesheet_in_progress = false;
		set_process(false);
		spritesheet_http.unref();
		
		// Re-enable button
		if (image_preview && image_preview->generate_spritesheet_button) {
			image_preview->generate_spritesheet_button->set_disabled(false);
			image_preview->generate_spritesheet_button->set_text("Generate Sprite Sheet from This Image");
		}
	}
}

void DesignStudio2DEditor::_process_spritesheet_ndjson_line(const String &p_line) {
	Ref<JSON> json;
	json.instantiate();
	Error err = json->parse(p_line);
	
	if (err != OK) {
		print_line(vformat("NDJSON_PARSE_ERROR: %s", json->get_error_message()));
		return;
	}
	
	Dictionary data = json->get_data();
	String status = data.get("status", "");
	
	if (status == "started") {
		int total = data.get("total_cells", 0);
		print_line(vformat("SPRITESHEET: Started - %d total cells", total));
		
	} else if (status == "progress") {
		int phase = data.get("phase", 0);
		int completed = data.get("completed", 0);
		int total = data.get("total", 0);
		float percent = data.get("progress_percent", 0.0);
		
		print_line(vformat("SPRITESHEET: Phase %d - %d/%d cells (%.1f%%)", phase, completed, total, percent));
		
		// Extract cell data
		if (data.has("cell")) {
			Dictionary cell = data.get("cell", Dictionary());
			int row = cell.get("row", 0);
			int col = cell.get("col", 0);
			String img_data = cell.get("image_data", "");
			
			// Store cell data
			if (row >= 0 && row < spritesheet_cells.size() && 
			    col >= 0 && col < spritesheet_cells[row].size()) {
				spritesheet_cells.write[row].write[col] = cell;
				print_line(vformat("SPRITESHEET: Cell [%d,%d] received (%dx%d)", 
				    row, col, cell.get("width", 0), cell.get("height", 0)));
			}
			
			// Update grid display progressively
			_update_spritesheet_grid_display();
		}
		
	} else if (status == "completed") {
		int completed_cells = data.get("completed_cells", 0);
		int total_cells = data.get("total_cells", 0);
		float total_time = data.get("total_time", 0.0);
		
		print_line(vformat("SPRITESHEET: COMPLETED - %d/%d cells in %.2fs", 
		    completed_cells, total_cells, total_time));
		
		// Final update
		_update_spritesheet_grid_display();
		
		// Mark as complete and show animation buttons
		spritesheet_generation_complete = true;
		_show_animation_buttons();
		
		EditorNode::get_singleton()->show_warning(vformat(
		    "Sprite sheet generation complete! %d/%d cells generated in %.1fs", 
		    completed_cells, total_cells, total_time
		));
		
	} else if (status == "error") {
		String error = data.get("error", "Unknown error");
		print_line(vformat("SPRITESHEET ERROR: %s", error));
		EditorNode::get_singleton()->show_warning(vformat("Sprite sheet generation error: %s", error));
	}
}

void DesignStudio2DEditor::_update_spritesheet_grid_display() {
	if (!grid_display) {
		return;
	}
	
	// Update each cell that has new data
	int completed = 0;
	for (int row = 0; row < spritesheet_cells.size(); row++) {
		for (int col = 0; col < spritesheet_cells[row].size(); col++) {
			Dictionary cell_data = spritesheet_cells[row][col];
			
			if (!cell_data.is_empty() && cell_data.has("image_data")) {
				completed++;
				
				// Check if we already set this texture
				String img_data_b64 = cell_data.get("image_data", "");
				if (img_data_b64.is_empty()) {
					continue;
				}
				
				// Decode and create texture
				PackedByteArray image_bytes = CoreBind::Marshalls::get_singleton()->base64_to_raw(img_data_b64);
				
				Ref<Image> img;
				img.instantiate();
				Error err = img->load_png_from_buffer(image_bytes);
				
				if (err == OK) {
					Ref<ImageTexture> texture = ImageTexture::create_from_image(img);
					grid_display->set_cell_image(row, col, texture);
				} else {
					print_line(vformat("GRID_UPDATE: Failed to load image for cell [%d,%d]", row, col));
				}
			}
		}
	}
	
	print_line(vformat("GRID_UPDATE: %d/%d cells displayed", completed, spritesheet_grid_width * spritesheet_grid_height));
}

void DesignStudio2DEditor::_show_animation_buttons() {
	print_line("ANIMATION_BUTTONS: Showing animation preview controls");
	
	// Ensure buttons container exists under the same parent as the grid (below it)
	if (!animation_buttons_container) {
		animation_buttons_container = memnew(VBoxContainer);
		if (grid_scroll_container && grid_scroll_container->get_parent()) {
			grid_scroll_container->get_parent()->add_child(animation_buttons_container);
		}
	}

	// Clear existing buttons
	for (int i = animation_buttons_container->get_child_count() - 1; i >= 0; i--) {
		animation_buttons_container->get_child(i)->queue_free();
	}

	Label *title = memnew(Label);
	title->set_text("Animation Previews:");
	title->add_theme_font_size_override("font_size", 14);
	animation_buttons_container->add_child(title);

	// See All Animations button
	Button *all_btn = memnew(Button);
	all_btn->set_text("See All Animations");
	all_btn->connect("pressed", callable_mp(this, &DesignStudio2DEditor::_on_preview_all_animations));
	animation_buttons_container->add_child(all_btn);

	// Reset button
	Button *reset_btn = memnew(Button);
	reset_btn->set_text("Reset Sprite Sheet");
	reset_btn->connect("pressed", callable_mp(this, &DesignStudio2DEditor::_on_reset_spritesheet));
	animation_buttons_container->add_child(reset_btn);

	// Per-row buttons
	for (int row = 0; row < spritesheet_grid_height; row++) {
		HBoxContainer *row_h = memnew(HBoxContainer);
		Label *row_lbl = memnew(Label);
		row_lbl->set_text(vformat("Row %d", row + 1));
		row_h->add_child(row_lbl);

		Button *row_btn = memnew(Button);
		row_btn->set_text("See Animation");
		Callable cb = callable_mp(this, &DesignStudio2DEditor::_on_preview_row_animation);
		cb = cb.bind(row);
		row_btn->connect("pressed", cb);
		row_h->add_child(row_btn);

		animation_buttons_container->add_child(row_h);
	}

	// Ensure popup exists
	if (!animation_preview_popup) {
		animation_preview_popup = memnew(AnimationPreviewPopup);
		add_child(animation_preview_popup);
	}
}

void DesignStudio2DEditor::_on_preview_row_animation(int p_row) {
	if (!grid_display) {
		return;
	}
	Vector<Ref<ImageTexture>> frames = grid_display->get_row_frames(p_row);
	if (frames.is_empty()) {
		EditorNode::get_singleton()->show_warning(vformat("No frames available for row %d", p_row + 1));
		return;
	}
	if (!animation_preview_popup) {
		animation_preview_popup = memnew(AnimationPreviewPopup);
		add_child(animation_preview_popup);
	}
	animation_preview_popup->set_animation_frames(frames, vformat("Row %d Animation", p_row + 1));
	animation_preview_popup->popup_centered();
}

void DesignStudio2DEditor::_on_preview_all_animations() {
	if (!grid_display) {
		return;
	}
	Vector<Ref<ImageTexture>> frames = grid_display->get_all_frames();
	if (frames.is_empty()) {
		EditorNode::get_singleton()->show_warning("No frames available in grid");
		return;
	}
	if (!animation_preview_popup) {
		animation_preview_popup = memnew(AnimationPreviewPopup);
		add_child(animation_preview_popup);
	}
	animation_preview_popup->set_animation_frames(frames, "All Animations");
	animation_preview_popup->popup_centered();
}

void DesignStudio2DEditor::_on_reset_spritesheet() {
	print_line("RESET: Clearing sprite sheet and returning to image preview");
	
	// Stop any in-progress generation
	if (spritesheet_in_progress) {
		spritesheet_in_progress = false;
		set_process(false);
		if (spritesheet_http.is_valid()) {
			spritesheet_http.unref();
		}
	}
	
	// Clear grid
	if (grid_display) {
		grid_display->clear_all_cells();
	}
	
	// Clear cell storage
	spritesheet_cells.clear();
	spritesheet_grid_width = 0;
	spritesheet_grid_height = 0;
	spritesheet_generation_complete = false;
	spritesheet_response_buffer = "";
	current_row_descriptions.clear();
	
	// Hide animation buttons
	if (animation_buttons_container) {
		animation_buttons_container->set_visible(false);
	}
	
	// Hide grid, show preview
	if (grid_scroll_container) {
		grid_scroll_container->set_visible(false);
	}
	if (image_preview && image_preview->get_parent()) {
		Control *parent_control = Object::cast_to<Control>(image_preview->get_parent());
		if (parent_control) {
			parent_control->set_visible(true);
		}
	}
	
	// Re-enable generate button
	if (image_preview && image_preview->generate_spritesheet_button) {
		image_preview->generate_spritesheet_button->set_disabled(false);
		image_preview->generate_spritesheet_button->set_text("Generate Sprite Sheet from This Image");
	}
	
	print_line("RESET: Complete - ready for new generation");
}

void DesignStudio2DEditor::_on_spritesheet_config_confirmed() {
	// Get row descriptions from dialog
	current_row_descriptions = spritesheet_config_dialog->get_row_descriptions();
	String format = spritesheet_config_dialog->get_selected_format();
	
	print_line(vformat("SPRITE_CONFIG: Format: %s, Rows: %d", format, current_row_descriptions.size()));
	for (int i = 0; i < current_row_descriptions.size(); i++) {
		print_line(vformat("  Row %d: %s", i + 1, current_row_descriptions[i]));
	}
	
	// Build comprehensive description from row descriptions
	String full_description = "Sprite sheet animation:\n";
	for (int i = 0; i < current_row_descriptions.size(); i++) {
		full_description += vformat("Row %d: %s\n", i + 1, current_row_descriptions[i]);
	}
	
	// Trigger generation with row-specific descriptions
	_start_progressive_spritesheet_generation(full_description);
	
	// Disable button
	if (image_preview && image_preview->generate_spritesheet_button) {
		image_preview->generate_spritesheet_button->set_disabled(true);
		image_preview->generate_spritesheet_button->set_text("Generating Sprite Sheet...");
	}
}

// DesignStudio2DEditorPlugin implementation
void DesignStudio2DEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			design_studio_editor->hide();
		} break;
	}
}

void DesignStudio2DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		design_studio_editor->show();
	} else {
		design_studio_editor->hide();
	}
}

DesignStudio2DEditorPlugin::DesignStudio2DEditorPlugin() {
	design_studio_editor = memnew(DesignStudio2DEditor);
	EditorNode::get_singleton()->get_editor_main_screen()->get_control()->add_child(design_studio_editor);
	design_studio_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	design_studio_editor->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	design_studio_editor->hide();
}

