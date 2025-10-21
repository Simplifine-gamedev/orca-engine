/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_chat_mode_selector.h"

#include "ai_chat_dock.h"
#include "scene/gui/option_button.h"
#include "scene/resources/style_box.h"

void AIChatModeSelector::_bind_methods() {
    ClassDB::bind_method(D_METHOD("_on_mode_selected"), &AIChatModeSelector::_on_mode_selected);
    ADD_SIGNAL(MethodInfo("mode_changed", PropertyInfo(Variant::STRING, "mode")));
}

void AIChatModeSelector::_apply_mode_colors() {
    if (!mode_dropdown || !chat_dock) {
        return;
    }
    const int idx = mode_dropdown->get_selected();
    // Colors: Ask = greenish; Agent = light orange
    const Color base_font = chat_dock->get_theme_color(SNAME("font_color"), SNAME("Editor"));
    Color ask_color = Color(0.45, 0.9, 0.45);      // soft green
    Color agent_color = Color(1.0, 0.7, 0.3);      // light orange

    Color target = (idx == MODE_AGENT) ? agent_color : ask_color;
    // Slightly blend with base to respect theme
    target = target.lerp(base_font, 0.25);

    mode_dropdown->add_theme_color_override("font_color", target);
    mode_dropdown->add_theme_color_override("font_hover_color", target);
    mode_dropdown->add_theme_color_override("font_pressed_color", target);
}

void AIChatModeSelector::setup(AIChatDock *p_chat_dock) {
    chat_dock = p_chat_dock;

    mode_dropdown = memnew(OptionButton);
    mode_dropdown->set_flat(true);
    mode_dropdown->set_clip_text(false);
    mode_dropdown->set_custom_minimum_size(Size2(0, 28));

    Ref<StyleBoxEmpty> empty_style = memnew(StyleBoxEmpty);
    mode_dropdown->add_theme_style_override("normal", empty_style);
    mode_dropdown->add_theme_style_override("hover", empty_style);
    mode_dropdown->add_theme_style_override("pressed", empty_style);
    mode_dropdown->add_theme_style_override("focus", empty_style);

    Color text_color = chat_dock->get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.6);
    Color hover_color = chat_dock->get_theme_color(SNAME("font_color"), SNAME("Editor"));
    mode_dropdown->add_theme_color_override("font_color", text_color);
    mode_dropdown->add_theme_color_override("font_hover_color", hover_color);
    mode_dropdown->add_theme_color_override("font_pressed_color", hover_color);
    mode_dropdown->add_theme_font_size_override("font_size", 18);
    mode_dropdown->set_fit_to_longest_item(false);

    mode_dropdown->add_item("Ask");
    mode_dropdown->add_item("Agent");

    mode_dropdown->connect("item_selected", callable_mp(this, &AIChatModeSelector::_on_mode_selected));

    add_child(mode_dropdown);

    // Initial selection: Agent is default
    if (mode_dropdown->get_item_count() > 1) {
        mode_dropdown->select(MODE_AGENT);
    }
    _apply_mode_colors();
}

void AIChatModeSelector::_on_mode_selected(int p_index) {
    if (!chat_dock) {
        return;
    }
    _apply_mode_colors();
    emit_signal("mode_changed", get_mode_string());
}

String AIChatModeSelector::get_mode_string() const {
    if (!mode_dropdown) {
        return "ask";
    }
    int idx = mode_dropdown->get_selected();
    if (idx == MODE_AGENT) {
        return "agent";
    }
    return "ask";
}

int AIChatModeSelector::get_mode_index() const {
    if (!mode_dropdown) {
        return MODE_ASK;
    }
    return mode_dropdown->get_selected();
}

void AIChatModeSelector::set_mode_by_string(const String &p_mode) {
    if (!mode_dropdown) {
        return;
    }
    if (p_mode == "agent") {
        mode_dropdown->select(MODE_AGENT);
    } else {
        mode_dropdown->select(MODE_ASK);
    }
}


