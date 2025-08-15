#include "diff_viewer.h"

#include "editor/editor_string_names.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/label.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/script/script_text_editor.h"
#include "editor/editor_interface.h"

void DiffViewer::_bind_methods() {
    ADD_SIGNAL(MethodInfo("diff_accepted", PropertyInfo(Variant::STRING, "path"), PropertyInfo(Variant::STRING, "content")));
    ClassDB::bind_method(D_METHOD("_on_accept_pressed"), &DiffViewer::_on_accept_pressed);
    ClassDB::bind_method(D_METHOD("_on_reject_pressed"), &DiffViewer::_on_reject_pressed);
    ClassDB::bind_method(D_METHOD("_on_accept_all_pressed"), &DiffViewer::_on_accept_all_pressed);
    ClassDB::bind_method(D_METHOD("_on_reject_all_pressed"), &DiffViewer::_on_reject_all_pressed);
}

DiffViewer::DiffViewer() {
    set_title(TTRC("Script Changes"));
    set_exclusive(true);

    VBoxContainer *main_vb = memnew(VBoxContainer);
    add_child(main_vb);

    ScrollContainer *scroll_container = memnew(ScrollContainer);
    scroll_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
    main_vb->add_child(scroll_container);

    hunks_container = memnew(VBoxContainer);
    hunks_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    scroll_container->add_child(hunks_container);

    HBoxContainer *button_hb = memnew(HBoxContainer);
    main_vb->add_child(button_hb);

    accept_all_button = memnew(Button);
    accept_all_button->set_text(TTRC("Accept All"));
    accept_all_button->connect("pressed", callable_mp(this, &DiffViewer::_on_accept_all_pressed));
    button_hb->add_child(accept_all_button);

    reject_all_button = memnew(Button);
    reject_all_button->set_text(TTRC("Reject All"));
    reject_all_button->connect("pressed", callable_mp(this, &DiffViewer::_on_reject_all_pressed));
    button_hb->add_child(reject_all_button);

    button_hb->add_spacer();
    
    Button *apply_to_editor_button = memnew(Button);
    apply_to_editor_button->set_text(TTRC("Apply to Editor"));
    apply_to_editor_button->connect("pressed", callable_mp(this, &DiffViewer::apply_to_script_editor));
    button_hb->add_child(apply_to_editor_button);

    accept_button = memnew(Button);
    accept_button->set_text(TTRC("Accept Selected"));
    accept_button->connect("pressed", callable_mp(this, &DiffViewer::_on_accept_pressed));
    button_hb->add_child(accept_button);

    reject_button = memnew(Button);
    reject_button->set_text(TTRC("Reject"));
    reject_button->connect("pressed", callable_mp(this, &DiffViewer::_on_reject_pressed));
    button_hb->add_child(reject_button);
}

void DiffViewer::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_POST_ENTER_TREE: {
            set_min_size(Size2(800, 600) * EDSCALE);
            // Add custom theme overrides here if needed
        } break;
    }
}

void DiffViewer::set_diff(const String &p_path, const String &p_original, const String &p_modified) {
    path = p_path;
    original_text = p_original;
    modified_text = p_modified;

    // Clear previous hunks
    for (int i = 0; i < hunks_container->get_child_count(); i++) {
        hunks_container->get_child(i)->queue_free();
    }
    hunks.clear();

    // Debug: Print the texts being compared
    print_line("DEBUG: Original text length: " + String::num_int64(p_original.length()));
    print_line("DEBUG: Modified text length: " + String::num_int64(p_modified.length()));

    Vector<String> original_lines_vec = original_text.split("\n");
    std::vector<String> original_lines;
    for (int i = 0; i < original_lines_vec.size(); i++) {
        original_lines.push_back(original_lines_vec[i]);
    }

    Vector<String> modified_lines_vec = modified_text.split("\n");
    std::vector<String> modified_lines;
    for (int i = 0; i < modified_lines_vec.size(); i++) {
        modified_lines.push_back(modified_lines_vec[i]);
    }

    dtl::Diff<String, std::vector<String>> d(original_lines, modified_lines);
    d.compose();
    d.composeUnifiedHunks();

    const auto &uni_hunks = d.getUniHunks();
    print_line("DEBUG: Number of hunks found: " + String::num_int64(uni_hunks.size()));

    if (uni_hunks.empty()) {
        // If no hunks, create a simple message
        PanelContainer *panel = memnew(PanelContainer);
        hunks_container->add_child(panel);
        
        VBoxContainer *vb = memnew(VBoxContainer);
        panel->add_child(vb);
        
        Label *no_changes_label = memnew(Label);
        no_changes_label->set_text("No differences found between original and modified content.");
        vb->add_child(no_changes_label);
        return;
    }

    for (const auto &hunk : uni_hunks) {
        DiffHunk diff_hunk;
        diff_hunk.hunk.a = hunk.a;
        diff_hunk.hunk.b = hunk.b;
        diff_hunk.hunk.c = hunk.c;
        diff_hunk.hunk.d = hunk.d;
        diff_hunk.accepted = true; // Default to accepted
        for (const auto &line : hunk.change) {
            DiffLine diff_line;
            diff_line.text = line.first;
            diff_line.type = line.second.type;
            diff_hunk.lines.push_back(diff_line);
        }
        hunks.push_back(diff_hunk);

        PanelContainer *panel = memnew(PanelContainer);
        hunks_container->add_child(panel);

        VBoxContainer *vb = memnew(VBoxContainer);
        panel->add_child(vb);

        CheckBox *checkbox = memnew(CheckBox);
        checkbox->set_text("@@ -" + String::num_int64(hunk.a) + "," + String::num_int64(hunk.b) + " +" + String::num_int64(hunk.c) + "," + String::num_int64(hunk.d) + " @@");
        checkbox->set_pressed(true);
        vb->add_child(checkbox);

        RichTextLabel *diff_label = memnew(RichTextLabel);
        diff_label->set_use_bbcode(true);
        diff_label->set_fit_content(true);
        diff_label->set_selection_enabled(true);
        diff_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
        diff_label->set_custom_minimum_size(Size2(0, 100));
        vb->add_child(diff_label);

        String diff_text = "";
        for (const auto &line : diff_hunk.lines) {
            if (line.type == dtl::SES_ADD) {
                diff_text += "[color=green]+" + line.text.xml_escape() + "[/color]\n";
            } else if (line.type == dtl::SES_DELETE) {
                diff_text += "[color=red]-" + line.text.xml_escape() + "[/color]\n";
            } else {
                diff_text += " " + line.text.xml_escape() + "\n";
            }
        }
        
        print_line("DEBUG: Setting diff text for hunk: " + diff_text.substr(0, 100) + "...");
        diff_label->set_text(diff_text);
    }
}

String DiffViewer::get_final_content() {
    String new_content = "";
    Vector<String> original_lines = original_text.split("\n");
    int last_line = 0;

    for (int i = 0; i < hunks.size(); i++) {
        const DiffHunk &diff_hunk = hunks[i];
        
        // Get checkbox state from UI
        PanelContainer *panel = Object::cast_to<PanelContainer>(hunks_container->get_child(i));
        if (panel) {
            VBoxContainer *vb = Object::cast_to<VBoxContainer>(panel->get_child(0));
            if (vb) {
                CheckBox *checkbox = Object::cast_to<CheckBox>(vb->get_child(0));
                if (checkbox) {
                    // Apply the hunk if checkbox is checked
                    if (checkbox->is_pressed()) {
                        // Add lines before this hunk
                        for (int j = last_line; j < diff_hunk.hunk.a - 1; j++) {
                            if (j < original_lines.size()) {
                                new_content += original_lines[j] + "\n";
                            }
                        }
                        // Add modified lines (excluding deletions)
                        for (const auto &line : diff_hunk.lines) {
                            if (line.type != dtl::SES_DELETE) {
                                new_content += line.text + "\n";
                            }
                        }
                        last_line = diff_hunk.hunk.a + diff_hunk.hunk.b - 1;
                    } else {
                        // Keep original lines
                        for (int j = last_line; j < diff_hunk.hunk.a + diff_hunk.hunk.b - 1; j++) {
                            if (j < original_lines.size()) {
                                new_content += original_lines[j] + "\n";
                            }
                        }
                        last_line = diff_hunk.hunk.a + diff_hunk.hunk.b - 1;
                    }
                }
            }
        }
    }

    // Add remaining lines
    for (int i = last_line; i < original_lines.size(); i++) {
        new_content += original_lines[i] + "\n";
    }

    return new_content;
}

bool DiffViewer::has_script_open(const String &p_path) {
    ScriptEditor *script_editor = ScriptEditor::get_singleton();
    if (!script_editor) {
        return false;
    }
    
    // Check if the script is currently open
    for (int i = 0; i < script_editor->get_open_scripts().size(); i++) {
        if (script_editor->get_open_scripts()[i]->get_path() == p_path) {
            return true;
        }
    }
    return false;
}

void DiffViewer::apply_to_script_editor() {
    if (path.is_empty()) {
        return;
    }
    
    ScriptEditor *script_editor = ScriptEditor::get_singleton();
    if (!script_editor) {
        return;
    }
    
    // Get the final content based on selected hunks
    String final_content = get_final_content();
    
    // First, try to open the script if not already open
    Ref<Resource> resource = ResourceLoader::load(path);
    Ref<Script> script = resource;
    
    if (script.is_valid()) {
        // Open the script in the editor
        script_editor->edit(script);
        
        // Get the current script editor
        ScriptTextEditor *ste = Object::cast_to<ScriptTextEditor>(script_editor->get_current_editor());
        if (ste) {
            // Update the script source directly
            script->set_source_code(final_content);
            
            // Update the text editor content
            CodeTextEditor *code_editor = ste->get_code_editor();
            if (code_editor) {
                CodeEdit *text_editor = code_editor->get_text_editor();
                if (text_editor) {
                    // Save cursor position
                    int cursor_line = text_editor->get_caret_line();
                    int cursor_column = text_editor->get_caret_column();
                    
                    // Update the text
                    text_editor->set_text(final_content);
                    
                    // Restore cursor position (if still valid)
                    if (cursor_line < text_editor->get_line_count()) {
                        text_editor->set_caret_line(cursor_line);
                        int max_column = text_editor->get_line(cursor_line).length();
                        text_editor->set_caret_column(MIN(cursor_column, max_column));
                    }
                    
                    // Don't mark as saved since we want the user to save manually
                    // text_editor->tag_saved_version();
                    
                    // Validate the script to check for errors
                    ste->validate();
                    
                    // Switch to the Script editor
                    EditorInterface::get_singleton()->set_main_screen_editor("Script");
                    
                    print_line("Applied diff changes to script editor for: " + path);
                }
            }
        }
    }
}

void DiffViewer::_on_accept_pressed() {
    apply_to_script_editor();
    emit_signal("diff_accepted", path, get_final_content());
    hide();
}

void DiffViewer::_on_accept_all_pressed() {
    for (int i = 0; i < hunks_container->get_child_count(); i++) {
        PanelContainer *panel = Object::cast_to<PanelContainer>(hunks_container->get_child(i));
        if (panel) {
            VBoxContainer *vb = Object::cast_to<VBoxContainer>(panel->get_child(0));
            if (vb) {
                CheckBox *checkbox = Object::cast_to<CheckBox>(vb->get_child(0));
                if (checkbox) {
                    checkbox->set_pressed(true);
                }
            }
        }
    }
    apply_to_script_editor();
    hide();
}

void DiffViewer::_on_reject_all_pressed() {
    for (int i = 0; i < hunks_container->get_child_count(); i++) {
        PanelContainer *panel = Object::cast_to<PanelContainer>(hunks_container->get_child(i));
        VBoxContainer *vb = Object::cast_to<VBoxContainer>(panel->get_child(0));
        CheckBox *checkbox = Object::cast_to<CheckBox>(vb->get_child(0));
        checkbox->set_pressed(false);
    }
}

void DiffViewer::_on_reject_pressed() {
    for (int i = 0; i < hunks_container->get_child_count(); i++) {
        PanelContainer *panel = Object::cast_to<PanelContainer>(hunks_container->get_child(i));
        VBoxContainer *vb = Object::cast_to<VBoxContainer>(panel->get_child(0));
        CheckBox *checkbox = Object::cast_to<CheckBox>(vb->get_child(0));
        if (checkbox->is_pressed()) {
            checkbox->set_pressed(false);
        }
    }
    hide();
}
