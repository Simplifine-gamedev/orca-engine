#pragma once

#include "scene/gui/popup.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "dtl.hpp"
#include <vector>
#include "common.h"

struct DiffLine {
    String text;
    dtl::edit_t type;
};

struct DiffHunk {
    dtl::uniHunk<String> hunk;
    Vector<DiffLine> lines;
    bool accepted = true; // Track acceptance state
};

class DiffViewer : public PopupPanel {
    GDCLASS(DiffViewer, PopupPanel);

private:
    VBoxContainer *hunks_container;
    Button *accept_button;
    Button *reject_button;
    Button *accept_all_button;
    Button *reject_all_button;

    String original_text;
    String modified_text;
    String path;

    Vector<DiffHunk> hunks;

protected:
    void _notification(int p_what);
    static void _bind_methods();

    void _on_accept_pressed();
    void _on_reject_pressed();
    void _on_accept_all_pressed();
    void _on_reject_all_pressed();

public:
    DiffViewer();

    void set_diff(const String &p_path, const String &p_original, const String &p_modified);
    
    // New methods for direct script editor integration
    String get_final_content();
    void apply_to_script_editor();
    bool has_script_open(const String &p_path);
};
