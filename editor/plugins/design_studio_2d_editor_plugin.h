/**************************************************************************/
/*  design_studio_2d_editor_plugin.h                                      */
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

#pragma once

#include "editor/plugins/editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/panel_container.h"
#include "scene/resources/image_texture.h"

class Button;
class FileDialog;
class GridContainer;
class HSplitContainer;
class HTTPClient;
class HTTPRequest;
class Label;
class LineEdit;
class OptionButton;
class ScrollContainer;
class SpinBox;
class TextEdit;
class TextureRect;

class DesignStudio2DEditor;

class SpriteSheetConfigDialog : public AcceptDialog {
	GDCLASS(SpriteSheetConfigDialog, AcceptDialog);

	OptionButton *format_dropdown = nullptr;
	VBoxContainer *row_configs_container = nullptr;
	Vector<LineEdit *> row_description_fields;
	int num_rows = 0;
	
	DesignStudio2DEditor *parent_editor = nullptr;
	
	void _on_format_selected(int p_index);
	void _apply_template(const String &p_template_name);

protected:
	void _notification(int p_what);

public:
	void setup(int p_num_rows, DesignStudio2DEditor *p_parent);
	Vector<String> get_row_descriptions() const;
	String get_selected_format() const;
	
	SpriteSheetConfigDialog();
};

class ImagePreviewPanel : public VBoxContainer {
	GDCLASS(ImagePreviewPanel, VBoxContainer);

	TextureRect *image_display = nullptr;
	Label *image_info_label = nullptr;
	TextEdit *chat_input = nullptr;
	Button *send_chat_button = nullptr;
	VBoxContainer *chat_history = nullptr;
	
	DesignStudio2DEditor *parent_editor = nullptr;
	
	void _on_send_chat_pressed();
	void _on_generate_spritesheet_pressed();
	void _on_chat_response(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);

protected:
	void _notification(int p_what);

public:
	Button *generate_spritesheet_button = nullptr;
	String current_image_id;
	String current_image_base64;
	int current_image_width = 0;
	int current_image_height = 0;
	
	void set_image(const String &p_image_id, const String &p_base64_data, int p_width, int p_height, const String &p_format);
	void clear_image();
	void set_parent_editor(DesignStudio2DEditor *p_editor) { parent_editor = p_editor; }
	
	ImagePreviewPanel();
};

class AnimationPreviewPopup : public PopupPanel {
	GDCLASS(AnimationPreviewPopup, PopupPanel);

	TextureRect *preview_display = nullptr;
	Label *frame_info_label = nullptr;
	Button *play_pause_button = nullptr;
	SpinBox *fps_spinbox = nullptr;
	
	Vector<Ref<ImageTexture>> animation_frames;
	int current_frame = 0;
	bool is_playing = false;
	float frame_timer = 0.0f;
	float fps = 8.0f;
	
	void _on_play_pause_pressed();
	void _on_fps_changed(double p_value);
	void _update_preview_frame();

protected:
	void _notification(int p_what);

public:
	void set_animation_frames(const Vector<Ref<ImageTexture>> &p_frames, const String &p_title);
	
	AnimationPreviewPopup();
};

class SpriteSheetGridDisplay : public Control {
	GDCLASS(SpriteSheetGridDisplay, Control);

	int grid_width = 4;
	int grid_height = 4;
	Vector<Vector<Ref<ImageTexture>>> cell_textures;  // [row][col] = texture

protected:
	void _notification(int p_what);

public:
	void set_grid_size(int p_width, int p_height);
	void set_cell_image(int p_row, int p_col, const Ref<ImageTexture> &p_texture);
	void clear_all_cells();
	Vector<Ref<ImageTexture>> get_row_frames(int p_row) const;
	Vector<Ref<ImageTexture>> get_all_frames() const;
	Size2 get_minimum_size() const override;
	
	SpriteSheetGridDisplay();
};

class DesignStudio2DEditor : public PanelContainer {
	GDCLASS(DesignStudio2DEditor, PanelContainer);

	HSplitContainer *main_split = nullptr;
	VBoxContainer *left_panel = nullptr;
	PanelContainer *canvas_area = nullptr;
	ImagePreviewPanel *image_preview = nullptr;
	SpriteSheetGridDisplay *grid_display = nullptr;
	ScrollContainer *grid_scroll_container = nullptr;
	
	// Sprite sheet creation UI elements
	SpinBox *grid_width = nullptr;
	Button *import_seed_button = nullptr;
	Button *create_image_button = nullptr;
	Label *seed_image_label = nullptr;
	TextEdit *description_text = nullptr;
	OptionButton *model_dropdown = nullptr;
	Button *advanced_button = nullptr;
	VBoxContainer *advanced_options = nullptr;
	
	FileDialog *seed_image_dialog = nullptr;
	String current_seed_image_path;
	String current_seed_image_base64;
	
	Ref<HTTPClient> spritesheet_http;
	String spritesheet_response_buffer;
	Vector<Vector<Dictionary>> spritesheet_cells;  // [row][col] = {image_data, width, height}
	int spritesheet_grid_width = 0;
	int spritesheet_grid_height = 0;
	bool spritesheet_in_progress = false;
	bool spritesheet_generation_complete = false;
	PackedStringArray spritesheet_pending_headers;
	String spritesheet_pending_body;
	
	VBoxContainer *animation_buttons_container = nullptr;
	AnimationPreviewPopup *animation_preview_popup = nullptr;
	
	void _on_import_seed_pressed();
	void _on_seed_image_selected(const String &p_path);
	void _on_create_image_pressed();
	void _on_advanced_toggled();
	void _on_grid_size_changed(double p_value);
	void _generate_single_image();
	void _on_single_image_generated(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _process_spritesheet_ndjson_line(const String &p_line);
	void _update_spritesheet_grid_display();
	void _poll_spritesheet_stream();
	void _on_spritesheet_config_confirmed();
	void _show_animation_buttons();
	void _on_preview_row_animation(int p_row);
	void _on_preview_all_animations();
	void _on_reset_spritesheet();
	Vector<String> current_row_descriptions;

protected:
	void _notification(int p_what);

public:
	SpriteSheetConfigDialog *spritesheet_config_dialog = nullptr;
	SpinBox *grid_height = nullptr;
	void _start_progressive_spritesheet_generation(const String &p_description);
	
	DesignStudio2DEditor();
};

class DesignStudio2DEditorPlugin : public EditorPlugin {
	GDCLASS(DesignStudio2DEditorPlugin, EditorPlugin);

	DesignStudio2DEditor *design_studio_editor = nullptr;

protected:
	void _notification(int p_what);

public:
	virtual String get_plugin_name() const override { return TTRC("2D Design Studio"); }
	bool has_main_screen() const override { return true; }
	virtual void make_visible(bool p_visible) override;

	DesignStudio2DEditorPlugin();
};

