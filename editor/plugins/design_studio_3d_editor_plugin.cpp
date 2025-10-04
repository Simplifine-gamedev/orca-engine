/**************************************************************************/
/*  design_studio_3d_editor_plugin.cpp                                    */
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

#include "design_studio_3d_editor_plugin.h"

#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/center_container.h"
#include "scene/gui/label.h"

// DesignStudio3DEditor implementation
void DesignStudio3DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			set_custom_minimum_size(Size2(200, 200) * EDSCALE);
		} break;
	}
}

DesignStudio3DEditor::DesignStudio3DEditor() {
	// Create a centered placeholder
	CenterContainer *center = memnew(CenterContainer);
	add_child(center);
	center->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);

	VBoxContainer *vbox = memnew(VBoxContainer);
	center->add_child(vbox);

	placeholder_label = memnew(Label);
	placeholder_label->set_text("3D Design Studio\n\nComing Soon...");
	placeholder_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	placeholder_label->add_theme_font_size_override("font_size", 24 * EDSCALE);
	vbox->add_child(placeholder_label);

	Label *subtitle = memnew(Label);
	subtitle->set_text("AI-powered 3D asset generation and editing");
	subtitle->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	subtitle->set_modulate(Color(0.7, 0.7, 0.7));
	vbox->add_child(subtitle);
}

// DesignStudio3DEditorPlugin implementation
void DesignStudio3DEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			design_studio_editor->hide();
		} break;
	}
}

void DesignStudio3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		design_studio_editor->show();
	} else {
		design_studio_editor->hide();
	}
}

DesignStudio3DEditorPlugin::DesignStudio3DEditorPlugin() {
	design_studio_editor = memnew(DesignStudio3DEditor);
	EditorNode::get_singleton()->get_editor_main_screen()->get_control()->add_child(design_studio_editor);
	design_studio_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	design_studio_editor->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	design_studio_editor->hide();
}

