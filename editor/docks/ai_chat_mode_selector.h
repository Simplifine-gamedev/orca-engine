/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "core/object/class_db.h"
#include "scene/gui/box_container.h"

class OptionButton;
class AIChatDock;

// Compact selector for chat mode (Ask / Agent)
class AIChatModeSelector : public HBoxContainer {
    GDCLASS(AIChatModeSelector, HBoxContainer);

public:
    enum ChatMode {
        MODE_ASK = 0,
        MODE_AGENT = 1,
    };

private:
    OptionButton *mode_dropdown = nullptr;
    AIChatDock *chat_dock = nullptr;
    void _apply_mode_colors(int p_index = -1);
    void _apply_mode_style();

protected:
    static void _bind_methods();

public:
    void setup(AIChatDock *p_chat_dock);
    String get_mode_string() const;
    int get_mode_index() const;
    void set_mode_by_string(const String &p_mode);

private:
    void _on_mode_selected(int p_index);
};


