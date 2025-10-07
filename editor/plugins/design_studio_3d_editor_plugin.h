/**************************************************************************/
/*  design_studio_3d_editor_plugin.h                                      */
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
#include "scene/gui/panel_container.h"
#include "scene/main/http_request.h"

class Button;
class Camera3D;
class DirectionalLight3D;
class HTTPRequest;
class Label;
class LineEdit;
class MeshInstance3D;
class OptionButton;
class SubViewport;
class SubViewportContainer;
class TextureRect;
class Timer;
class VBoxContainer;

class DesignStudio3DEditor : public PanelContainer {
	GDCLASS(DesignStudio3DEditor, PanelContainer);

	// API Configuration
	const String API_URL = "https://gpu-proxy-awdwh5ovsa-uc.a.run.app";
	
	// UI Elements - Left Panel
	LineEdit *prompt_input = nullptr;
	OptionButton *quality_selector = nullptr;
	Button *generate_button = nullptr;
	Label *status_label = nullptr;
	
	// UI Elements - 3D Viewer
	SubViewportContainer *viewport_container = nullptr;
	SubViewport *viewport = nullptr;
	Camera3D *camera = nullptr;
	DirectionalLight3D *light = nullptr;
	MeshInstance3D *preview_mesh = nullptr;
	
	// HTTP Requests
	HTTPRequest *submit_request = nullptr;
	HTTPRequest *poll_request = nullptr;
	HTTPRequest *download_request = nullptr;
	Timer *poll_timer = nullptr;
	
	// State
	String current_job_id;
	String current_user_id = "godot_user";
	bool is_generating = false;
	
	// Download retry logic
	String download_url_to_retry;
	int download_retry_count = 0;
	Timer *download_retry_timer = nullptr;
	
	// 3D Viewer controls
	bool is_rotating = false;
	Vector2 last_mouse_pos;
	Vector3 orbit_center;
	float orbit_distance = 5.0f;
	float orbit_pitch = -20.0f; // degrees
	float orbit_yaw = 45.0f; // degrees
	
	void _setup_ui();
	void _setup_3d_viewer();
	
	void _on_generate_pressed();
	void _on_job_submitted(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _on_poll_timeout();
	void _on_job_status_received(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _on_model_downloaded(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	
	void _start_polling(const String &p_job_id);
	void _stop_polling();
	void _load_model_from_data(const PackedByteArray &p_data);
	Ref<ArrayMesh> _parse_obj_to_mesh(const String &p_obj_content);
	void _setup_camera_orbit();
	void _update_camera_from_orbit();
	void _on_viewport_input(const Ref<InputEvent> &p_event);
	void _on_download_retry_timeout();
	void _start_download_with_headers(const String &p_url);
	void _load_imported_mesh(const String &p_path);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	DesignStudio3DEditor();
};

class DesignStudio3DEditorPlugin : public EditorPlugin {
	GDCLASS(DesignStudio3DEditorPlugin, EditorPlugin);

	DesignStudio3DEditor *design_studio_editor = nullptr;

protected:
	void _notification(int p_what);

public:
	virtual String get_plugin_name() const override { return TTRC("3D Design Studio"); }
	bool has_main_screen() const override { return true; }
	virtual void make_visible(bool p_visible) override;

	DesignStudio3DEditor *get_design_studio_editor() { return design_studio_editor; }

	DesignStudio3DEditorPlugin();
	~DesignStudio3DEditorPlugin();
};

