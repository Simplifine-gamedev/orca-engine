/**************************************************************************/
/*  ai_pending_edits_banner.cpp                                           */
/**************************************************************************/

#include "ai_pending_edits_banner.h"

#include "core/os/keyboard.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/button.h"
#include "scene/gui/separator.h"
#include "scene/resources/style_box_flat.h"
#include "core/input/input_event.h"

void AIPendingEditsBanner::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_banner_gui_input", "event"), &AIPendingEditsBanner::_on_banner_gui_input);
    ClassDB::bind_method(D_METHOD("_emit_accept_all"), &AIPendingEditsBanner::_emit_accept_all);
    ClassDB::bind_method(D_METHOD("_emit_reject_all"), &AIPendingEditsBanner::_emit_reject_all);
	ADD_SIGNAL(MethodInfo("accept_all"));
	ADD_SIGNAL(MethodInfo("reject_all"));
}

AIPendingEditsBanner::AIPendingEditsBanner() {
	set_h_size_flags(Control::SIZE_EXPAND_FILL);

	// Details block shown above the banner when expanded
	details_container = memnew(VBoxContainer);
	add_child(details_container);
	details_container->set_visible(false);

	actions_row = memnew(HBoxContainer);
	details_container->add_child(actions_row);

    accept_all_btn = memnew(Button);
    accept_all_btn->set_text("Accept All");
    actions_row->add_child(accept_all_btn);

    reject_all_btn = memnew(Button);
    reject_all_btn->set_text("Reject All");
    actions_row->add_child(reject_all_btn);

	files_list = memnew(VBoxContainer);
	files_list->add_theme_constant_override("separation", 4);
	details_container->add_child(files_list);

	// Banner panel (clickable area) under the details
    banner_panel = memnew(PanelContainer);
    add_child(banner_panel);

	banner_content = memnew(HBoxContainer);
	banner_panel->add_child(banner_content);

    banner_icon = memnew(TextureRect);
    banner_icon->set_custom_minimum_size(Size2(16, 16));
    banner_content->add_child(banner_icon);

	count_label = memnew(Label);
	count_label->set_text("0 files pending review");
	count_label->add_theme_color_override("font_color", Color(0.18, 0.36, 0.72, 0.9));
	count_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	banner_content->add_child(count_label);

    chevron_icon = memnew(TextureRect);
    chevron_icon->set_custom_minimum_size(Size2(12, 12));
    banner_content->add_child(chevron_icon);

	// Click to expand/collapse
	banner_panel->connect("gui_input", callable_mp(this, &AIPendingEditsBanner::_on_banner_gui_input));

    // Relay Accept/Reject All actions as signals via local wrappers
    accept_all_btn->connect("pressed", callable_mp(this, &AIPendingEditsBanner::_emit_accept_all));
    reject_all_btn->connect("pressed", callable_mp(this, &AIPendingEditsBanner::_emit_reject_all));
}

void AIPendingEditsBanner::update_counts(int p_unique_files, int p_total_edits) {
	
	if (p_unique_files <= 0 || p_total_edits <= 0) {
		set_visible(false);
		return;
	}
	String text;
	if (p_unique_files == 1) {
		text = "1 file pending review";
	} else {
		text = String::num_int64(p_unique_files) + " files pending review";
	}
	if (p_total_edits > p_unique_files) {
		text += " (" + String::num_int64(p_total_edits) + " edits)";
	}
	count_label->set_text(text);
	if (!is_visible()) {
		set_visible(true);
	}
}

void AIPendingEditsBanner::update_file_map(const HashMap<String, Array> &p_file_to_tool_ids) {
	_rebuild_details(p_file_to_tool_ids);
}

void AIPendingEditsBanner::_toggle_details() {
	expanded = !expanded;
	details_container->set_visible(expanded);
	chevron_icon->set_texture(get_theme_icon(expanded ? SNAME("GuiTreeArrowDown") : SNAME("GuiTreeArrowRight"), SNAME("EditorIcons")));
}

void AIPendingEditsBanner::_rebuild_details(const HashMap<String, Array> &p_file_to_tool_ids) {
	// Clear old list
	for (int i = 0; i < files_list->get_child_count(); i++) {
		Node *c = files_list->get_child(i);
		c->queue_free();
	}

	// Rows per file: icon, name, count
	for (const KeyValue<String, Array> &E : p_file_to_tool_ids) {
		const String &file_path = E.key;
		const Array &tool_ids = E.value;

		HBoxContainer *row = memnew(HBoxContainer);
		files_list->add_child(row);

		TextureRect *icon = memnew(TextureRect);
		icon->set_texture(get_theme_icon(SNAME("GDScript"), SNAME("EditorIcons")));
		icon->set_custom_minimum_size(Size2(16, 16));
		row->add_child(icon);

		Label *name = memnew(Label);
		name->set_text(file_path.get_file());
		name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		row->add_child(name);

		Label *count = memnew(Label);
		count->set_text("(" + String::num_int64(tool_ids.size()) + (tool_ids.size() == 1 ? " edit)" : " edits)"));
		row->add_child(count);
	}
}

void AIPendingEditsBanner::_on_banner_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		_toggle_details();
	}
}

void AIPendingEditsBanner::_emit_accept_all() {
    emit_signal("accept_all");
}

void AIPendingEditsBanner::_emit_reject_all() {
    emit_signal("reject_all");
}

void AIPendingEditsBanner::_notification(int p_what) {
    if (p_what == NOTIFICATION_POSTINITIALIZE || p_what == NOTIFICATION_THEME_CHANGED) {
        _update_theme();
    }
}

void AIPendingEditsBanner::_update_theme() {
    if (!is_inside_tree()) {
        return;
    }
    if (banner_panel) {
        if (!banner_style.is_valid()) {
            banner_style.instantiate();
            banner_style->set_bg_color(Color(0.18, 0.36, 0.72, 0.15));
            banner_style->set_border_width_all(1);
            banner_style->set_border_color(Color(0.18, 0.36, 0.72, 0.4));
            banner_style->set_corner_radius_all(6);
            banner_style->set_content_margin_all(8);
        }
        banner_panel->add_theme_style_override("panel", banner_style);
    }
    if (banner_icon) {
        banner_icon->set_texture(get_theme_icon(SNAME("Edit"), SNAME("EditorIcons")));
    }
    if (chevron_icon) {
        chevron_icon->set_texture(get_theme_icon(expanded ? SNAME("GuiTreeArrowDown") : SNAME("GuiTreeArrowRight"), SNAME("EditorIcons")));
    }
    if (accept_all_btn) {
        accept_all_btn->add_theme_icon_override("icon", get_theme_icon(SNAME("ImportCheck"), SNAME("EditorIcons")));
    }
    if (reject_all_btn) {
        reject_all_btn->add_theme_icon_override("icon", get_theme_icon(SNAME("ImportFail"), SNAME("EditorIcons")));
    }
}


