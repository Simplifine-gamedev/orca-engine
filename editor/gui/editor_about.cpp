/**************************************************************************/
/*  editor_about.cpp                                                      */
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

#include "editor_about.h"

#include "core/authors.gen.h"
#include "core/donors.gen.h"
#include "core/license.gen.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/credits_roll.h"
#include "editor/gui/editor_version_button.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/button.h"
#include "scene/gui/item_list.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tree.h"
#include "scene/resources/style_box.h"

void EditorAbout::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
			_about_text_label->set_text(
					String(U"Built with Orca Engine\n") +
					String(U"Join our Discord!\n"));

			// Discord link setup
			if (_discord_button) {
				_discord_button->set_text("Join our Discord!");
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			_logo->set_texture(get_editor_theme_icon(SNAME("Logo")));
		} break;
	}
}

void EditorAbout::_license_tree_selected() {
	TreeItem *selected = _tpl_tree->get_selected();
	_tpl_text->scroll_to_line(0);
	_tpl_text->set_text(selected->get_metadata(0));
}

void EditorAbout::_item_activated(int p_idx, ItemList *p_il) {
	const Variant val = p_il->get_item_metadata(p_idx);
	if (val.get_type() == Variant::STRING) {
		OS::get_singleton()->shell_open(val);
	} else {
		// Easter egg :D
		if (!EditorNode::get_singleton()) {
			// Don't allow in Project Manager.
			return;
		}

		if (!credits_roll) {
			credits_roll = memnew(CreditsRoll);
			add_child(credits_roll);
		}
		credits_roll->roll_credits();
	}
}

void EditorAbout::_item_list_resized(ItemList *p_il) {
	p_il->set_fixed_column_width(p_il->get_size().x / 3.0 - 16 * EDSCALE * 2.5); // Weird. Should be 3.0 and that's it?.
}

void EditorAbout::_discord_pressed() {
	OS::get_singleton()->shell_open("https://discord.gg/jwqTUP4CgJ");
}

Label *EditorAbout::_create_section(Control *p_parent, const String &p_name, const char *const *p_src, BitField<SectionFlags> p_flags) {
	Label *lbl = memnew(Label(p_name));
	lbl->set_theme_type_variation("HeaderSmall");
	lbl->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	p_parent->add_child(lbl);

	ItemList *il = memnew(ItemList);
	il->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	il->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	il->set_same_column_width(true);
	il->set_auto_height(true);
	il->set_max_columns(p_flags.has_flag(FLAG_SINGLE_COLUMN) ? 1 : 16);
	il->add_theme_constant_override("h_separation", 16 * EDSCALE);

	if (p_flags.has_flag(FLAG_ALLOW_WEBSITE) || (p_flags.has_flag(FLAG_EASTER_EGG) && EditorNode::get_singleton())) {
		Ref<StyleBoxEmpty> empty_stylebox = memnew(StyleBoxEmpty);
		il->add_theme_style_override("focus", empty_stylebox);
		il->add_theme_style_override("selected", empty_stylebox);

		il->connect("item_activated", callable_mp(this, &EditorAbout::_item_activated).bind(il));
	} else {
		il->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
		il->set_focus_mode(Control::FOCUS_NONE);
	}

	const char *const *names_ptr = p_src;
	if (p_flags.has_flag(FLAG_ALLOW_WEBSITE)) {
		il->connect(SceneStringName(resized), callable_mp(this, &EditorAbout::_item_list_resized).bind(il));
		il->connect(SceneStringName(focus_exited), callable_mp(il, &ItemList::deselect_all));

		while (*names_ptr) {
			const String name = String::utf8(*names_ptr++);
			const String identifier = name.get_slicec('<', 0);
			const String website = name.get_slice_count("<") == 1 ? "" : name.get_slicec('<', 1).trim_suffix(">");

			il->add_item(identifier, nullptr, !website.is_empty());

			if (website.is_empty()) {
				il->set_item_tooltip_enabled(-1, false);
			} else {
				il->set_item_metadata(-1, website);
			}

			if (!*names_ptr && name.contains(" anonymous ")) {
				il->set_item_disabled(-1, true);
			}
		}
	} else {
		while (*names_ptr) {
			il->add_item(String::utf8(*names_ptr++), nullptr, false);
			il->set_item_tooltip_enabled(-1, false);
		}
	}

	name_lists.append(il);

	p_parent->add_child(il);

	HSeparator *hs = memnew(HSeparator);
	hs->set_modulate(Color(0, 0, 0, 0));
	p_parent->add_child(hs);

	return lbl;
}

EditorAbout::EditorAbout() {
	set_title(TTRC("About Orca Engine"));
	set_hide_on_ok(true);

	// Create simple about dialog with basic layout
	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	vbc->add_theme_constant_override("separation", 20 * EDSCALE);
	add_child(vbc);

	// Add some padding at the top
	Control *top_spacer = memnew(Control);
	top_spacer->set_custom_minimum_size(Size2(0, 20) * EDSCALE);
	vbc->add_child(top_spacer);

	// Logo
	_logo = memnew(TextureRect);
	_logo->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	_logo->set_custom_minimum_size(Size2(64, 64) * EDSCALE);
	_logo->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	vbc->add_child(_logo);

	// Version button
	EditorVersionButton *version_btn = memnew(EditorVersionButton(EditorVersionButton::FORMAT_WITH_NAME_AND_BUILD));
	version_btn->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	vbc->add_child(version_btn);

	// About text
	_about_text_label = memnew(Label);
	_about_text_label->set_text("Built with Orca Engine\nJoin our Discord!");
	_about_text_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	_about_text_label->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	_about_text_label->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	vbc->add_child(_about_text_label);

	// Discord button
	_discord_button = memnew(Button);
	_discord_button->set_text("Join our Discord!");
	_discord_button->set_custom_minimum_size(Size2(160, 32) * EDSCALE);
	_discord_button->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	_discord_button->connect(SceneStringName(pressed), callable_mp(this, &EditorAbout::_discord_pressed));
	vbc->add_child(_discord_button);

	// Set dialog size
	Size2 dialog_size = Size2(350, 250) * EDSCALE;
	set_min_size(dialog_size);
}
