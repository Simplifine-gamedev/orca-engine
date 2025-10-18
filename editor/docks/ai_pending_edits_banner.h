/**************************************************************************/
/*  ai_pending_edits_banner.h                                             */
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

#include "core/object/class_db.h"
#include "core/templates/hash_map.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/separator.h"
#include "scene/resources/style_box_flat.h"

class AIPendingEditsBanner : public VBoxContainer {
	GDCLASS(AIPendingEditsBanner, VBoxContainer);

private:
	PanelContainer *banner_panel = nullptr;
	HBoxContainer *banner_content = nullptr;
	TextureRect *banner_icon = nullptr;
	Label *count_label = nullptr;
	TextureRect *chevron_icon = nullptr;

	VBoxContainer *details_container = nullptr;
	HBoxContainer *actions_row = nullptr;
	Button *accept_all_btn = nullptr;
	Button *reject_all_btn = nullptr;
	VBoxContainer *files_list = nullptr;

	Ref<StyleBoxFlat> banner_style;

	bool expanded = false;

	void _toggle_details();
	void _rebuild_details(const HashMap<String, Array> &p_file_to_tool_ids);
	void _on_banner_gui_input(const Ref<InputEvent> &p_event);
	void _emit_accept_all();
	void _emit_reject_all();
	void _update_theme();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	AIPendingEditsBanner();

	void update_counts(int p_unique_files, int p_total_edits);
	void update_file_map(const HashMap<String, Array> &p_file_to_tool_ids);
};


