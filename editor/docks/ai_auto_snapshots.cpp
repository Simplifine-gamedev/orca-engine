/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_auto_snapshots.h"

#include "ai_chat_dock.h"
#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/os/os.h"
#include "core/os/time.h"
#include "editor/file_system/editor_paths.h"
#include "editor/editor_string_names.h"

void AIAutoSnapshots::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_snapshot_item_selected"), &AIAutoSnapshots::_on_snapshot_item_selected);
}

AIAutoSnapshots::AIAutoSnapshots() {
}

AIAutoSnapshots::~AIAutoSnapshots() {
}

void AIAutoSnapshots::initialize(AIChatDock *p_chat_dock) {
	chat_dock = p_chat_dock;
	if (!chat_dock) {
		return;
	}
}

void AIAutoSnapshots::_ensure_dialog_created() {
	if (!chat_dock) {
		return;
	}

	if (auto_snapshots_window) {
		return;
	}

	auto_snapshots_window = memnew(Window);
	auto_snapshots_window->set_title("AI Auto Snapshots");
	auto_snapshots_window->set_min_size(Size2(960, 560));
	auto_snapshots_window->set_wrap_controls(false);

	VBoxContainer *main_vbox = memnew(VBoxContainer);
	auto_snapshots_window->add_child(main_vbox);

	// Header
	Label *header_label = memnew(Label);
	header_label->set_text("AI Automatic Checkpoints (per-user message snapshots)");
	header_label->add_theme_font_override("font", chat_dock->get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
	header_label->add_theme_font_size_override("font_size", 16);
	header_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	main_vbox->add_child(header_label);

	main_vbox->add_child(memnew(HSeparator));

	// Main content: left = list of checkpoints, right = folder tree/details.
	HBoxContainer *content_hbox = memnew(HBoxContainer);
	content_hbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	content_hbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	main_vbox->add_child(content_hbox);

	// Left side: checkpoint list.
	VBoxContainer *left_vbox = memnew(VBoxContainer);
	left_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	left_vbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	left_vbox->set_custom_minimum_size(Size2(320, 0));
	content_hbox->add_child(left_vbox);

	Label *list_label = memnew(Label);
	list_label->set_text("Automatic snapshots created before each AI response:");
	list_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	left_vbox->add_child(list_label);

	snapshot_list_tree = memnew(Tree);
	snapshot_list_tree->set_columns(3);
	snapshot_list_tree->set_column_title(0, "Checkpoint Tag");
	snapshot_list_tree->set_column_title(1, "Message Index");
	snapshot_list_tree->set_column_title(2, "Created");
	snapshot_list_tree->set_column_titles_visible(true);
	snapshot_list_tree->set_hide_root(true);
	snapshot_list_tree->set_custom_minimum_size(Size2(0, 320));
	snapshot_list_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	snapshot_list_tree->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	snapshot_list_tree->set_select_mode(Tree::SELECT_SINGLE);
	snapshot_list_tree->connect("item_selected", callable_mp(this, &AIAutoSnapshots::_on_snapshot_item_selected));
	left_vbox->add_child(snapshot_list_tree);

	// Right side: details + folder tree.
	VBoxContainer *right_vbox = memnew(VBoxContainer);
	right_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	right_vbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	right_vbox->set_custom_minimum_size(Size2(480, 0));
	content_hbox->add_child(right_vbox);

	details_label = memnew(Label);
	details_label->set_text("Select a snapshot on the left to explore the folder structure that AI saw when it started working on that message.");
	details_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	details_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	right_vbox->add_child(details_label);

	PanelContainer *folder_panel = memnew(PanelContainer);
	folder_panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	folder_panel->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	right_vbox->add_child(folder_panel);

	folder_tree = memnew(Tree);
	folder_tree->set_hide_root(true);
	folder_tree->set_column_titles_visible(false);
	folder_tree->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	folder_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	folder_panel->add_child(folder_tree);

	main_vbox->add_child(memnew(HSeparator));

	// Bottom buttons.
	HBoxContainer *buttons_hbox = memnew(HBoxContainer);
	buttons_hbox->set_alignment(BoxContainer::ALIGNMENT_END);
	main_vbox->add_child(buttons_hbox);

	restore_button = memnew(Button);
	restore_button->set_text("Restore Project to This Snapshot");
	restore_button->add_theme_icon_override("icon", chat_dock->get_theme_icon(SNAME("Reload"), SNAME("EditorIcons")));
	restore_button->add_theme_color_override("font_color", chat_dock->get_theme_color(SNAME("success_color"), SNAME("Editor")));
	restore_button->set_disabled(true);
	restore_button->connect("pressed", callable_mp(this, &AIAutoSnapshots::_on_restore_button_pressed));
	buttons_hbox->add_child(restore_button);

	Button *close_button = memnew(Button);
	close_button->set_text("Close");
	close_button->connect("pressed", callable_mp((Window *)auto_snapshots_window, &Window::hide));
	buttons_hbox->add_child(close_button);

	chat_dock->add_child(auto_snapshots_window);
}

void AIAutoSnapshots::show_auto_snapshots_dialog() {
	if (!chat_dock) {
		return;
	}

	_ensure_dialog_created();
	_refresh_auto_snapshots_list();

	if (auto_snapshots_window) {
		auto_snapshots_window->popup_centered();
	}
}

Vector<AIAutoSnapshots::AutoSnapshot> AIAutoSnapshots::_get_all_auto_snapshots() {
	Vector<AutoSnapshot> snapshots;

	String checkpoint_dir = _get_checkpoint_directory();
	if (checkpoint_dir.is_empty() || !DirAccess::exists(checkpoint_dir)) {
		return snapshots;
	}

	List<String> args;
	args.push_back("-C");
	args.push_back(checkpoint_dir);
	args.push_back("tag");
	args.push_back("--list");
	args.push_back("msg_*");
	args.push_back("--sort=-creatordate");
	args.push_back("--format=%(refname:short)|%(creatordate:unix)|%(contents:subject)");

	String output;
	int exitcode = 0;
	Error err = OS::get_singleton()->execute("git", args, &output, &exitcode, false, nullptr, false);

	if (err != OK || exitcode != 0 || output.strip_edges().is_empty()) {
		return snapshots;
	}

	PackedStringArray lines = output.strip_edges().split("\n");
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i].strip_edges();
		if (line.is_empty()) {
			continue;
		}

		PackedStringArray parts = line.split("|");
		if (parts.size() < 3) {
			continue;
		}

		AutoSnapshot snapshot;
		snapshot.tag_name = parts[0];
		snapshot.created_unix_time = parts[1].to_int();
		snapshot.description = parts[2];
		snapshot.message_index = _parse_message_index_from_tag(snapshot.tag_name);
		snapshot.created_timestamp = _format_timestamp_from_unix(snapshot.created_unix_time);

		snapshots.push_back(snapshot);
	}

	return snapshots;
}

void AIAutoSnapshots::_refresh_auto_snapshots_list() {
	if (!snapshot_list_tree) {
		return;
	}

	snapshot_list_tree->clear();
	TreeItem *root = snapshot_list_tree->create_item();

	Vector<AutoSnapshot> snapshots = _get_all_auto_snapshots();
	if (snapshots.is_empty()) {
		TreeItem *empty = snapshot_list_tree->create_item(root);
		empty->set_text(0, "No automatic snapshots yet");
		if (chat_dock) {
			empty->set_custom_color(0, chat_dock->get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.5));
		}
		empty->set_selectable(0, false);

		if (details_label) {
			details_label->set_text("Automatic checkpoints are created before each AI response. Once you start chatting, they will appear here.");
		}
		if (folder_tree) {
			folder_tree->clear();
		}
		return;
	}

	for (int i = 0; i < snapshots.size(); i++) {
		const AutoSnapshot &snap = snapshots[i];
		TreeItem *item = snapshot_list_tree->create_item(root);

		item->set_text(0, snap.tag_name);
		item->set_text(1, String::num_int64(snap.message_index));
		item->set_text(2, snap.created_timestamp);

		item->set_metadata(0, snap.tag_name);
		item->set_tooltip_text(0, snap.description);

		if (chat_dock) {
			item->set_icon(0, chat_dock->get_theme_icon(SNAME("History"), SNAME("EditorIcons")));
		}
	}

	if (details_label) {
		details_label->set_text("Select a snapshot on the left to explore the folder structure that AI saw when it started working on that message.");
	}
	if (folder_tree) {
		folder_tree->clear();
	}
}

void AIAutoSnapshots::_on_snapshot_item_selected() {
	if (!snapshot_list_tree || !folder_tree || !details_label) {
		return;
	}

	TreeItem *selected = snapshot_list_tree->get_selected();
	if (!selected) {
		return;
	}

	selected_tag_name = selected->get_metadata(0);
	String created = selected->get_text(2);
	String msg_index_str = selected->get_text(1);
	selected_message_index = msg_index_str.to_int();

	String summary = "Checkpoint tag: " + selected_tag_name + "\n";
	summary += "Message index: " + msg_index_str + "\n";
	summary += "Created: " + created + "\n\n";
	summary += "Folder view below is read‑only and shows the snapshot as it was when this checkpoint was taken.";

	details_label->set_text(summary);

	if (restore_button) {
		restore_button->set_disabled(false);
	}

	_populate_folder_tree(selected_tag_name);
}

void AIAutoSnapshots::_on_restore_button_pressed() {
	if (!chat_dock) {
		return;
	}
	if (selected_message_index < 0) {
		return;
	}

	String msg = "This will restore your ENTIRE project to the state it was in when this snapshot was taken.\n\n";
	msg += "Checkpoint tag: " + selected_tag_name + "\n";
	msg += "Message index: " + String::num_int64(selected_message_index) + "\n\n";
	msg += "All current uncommitted changes will be lost.\n";
	msg += "This action cannot be undone.\n\n";
	msg += "Do you want to continue?";

	ConfirmationDialog *confirm = memnew(ConfirmationDialog);
	confirm->set_title("Restore Project from AI Snapshot");
	confirm->set_text(msg);
	confirm->connect("confirmed", callable_mp(this, &AIAutoSnapshots::_on_restore_selected_snapshot_confirmed));
	confirm->connect("popup_hide", callable_mp((Node *)confirm, &Node::queue_free));

	chat_dock->add_child(confirm);
	confirm->popup_centered(Size2(520, 260));
}

void AIAutoSnapshots::_on_restore_selected_snapshot_confirmed() {
	if (!chat_dock) {
		return;
	}
	if (selected_message_index < 0) {
		return;
	}


	bool success = chat_dock->_restore_from_checkpoint((int)selected_message_index);

	if (success) {
		if (auto_snapshots_window) {
			auto_snapshots_window->hide();
		}
		chat_dock->call("_show_status_notification", "success", "Project restored to AI checkpoint", "✅", 4.0);
	} else {
		chat_dock->call("_show_status_notification", "connection_error", "Failed to restore from AI checkpoint", "❌", 3.0);
	}
}

void AIAutoSnapshots::_populate_folder_tree(const String &p_tag_name) {
	if (!folder_tree) {
		return;
	}

	folder_tree->clear();
	TreeItem *root = folder_tree->create_item();

	String checkpoint_dir = _get_checkpoint_directory();
	if (checkpoint_dir.is_empty() || !DirAccess::exists(checkpoint_dir)) {
		TreeItem *item = folder_tree->create_item(root);
		item->set_text(0, "Checkpoint directory not found");
		return;
	}

	List<String> args;
	args.push_back("-C");
	args.push_back(checkpoint_dir);
	args.push_back("ls-tree");
	args.push_back("-r");
	args.push_back("--name-only");
	args.push_back(p_tag_name);

	String output;
	int exitcode = 0;
	Error err = OS::get_singleton()->execute("git", args, &output, &exitcode, false, nullptr, false);

	if (err != OK || exitcode != 0) {
		TreeItem *item = folder_tree->create_item(root);
		item->set_text(0, "Failed to list files for snapshot: " + p_tag_name);
		return;
	}

	output = output.strip_edges();
	if (output.is_empty()) {
		TreeItem *item = folder_tree->create_item(root);
		item->set_text(0, "Snapshot has no files");
		return;
	}

	PackedStringArray lines = output.split("\n");
	Vector<String> paths;
	paths.resize(lines.size());
	for (int i = 0; i < lines.size(); i++) {
		paths.write[i] = lines[i].strip_edges();
	}

	_build_folder_tree_from_paths(paths);
}

void AIAutoSnapshots::_build_folder_tree_from_paths(const Vector<String> &p_paths) {
	if (!folder_tree) {
		return;
	}

	folder_tree->clear();
	TreeItem *root = folder_tree->create_item();

	// Map "dir path" -> TreeItem* so we can reuse existing nodes.
	HashMap<String, TreeItem *> dir_items;
	dir_items[""] = root;

	for (int i = 0; i < p_paths.size(); i++) {
		String path = p_paths[i];
		if (path.is_empty()) {
			continue;
		}

		PackedStringArray parts = path.split("/");
		String current_dir;
		TreeItem *parent = root;

		// Create directory items for all but the last part.
		for (int j = 0; j < parts.size() - 1; j++) {
			String part = parts[j];
			if (part.is_empty()) {
				continue;
			}

			if (!current_dir.is_empty()) {
				current_dir += "/";
			}
			current_dir += part;

			TreeItem **found = dir_items.getptr(current_dir);
			if (found) {
				parent = *found;
				continue;
			}

			TreeItem *dir_item = folder_tree->create_item(parent);
			dir_item->set_text(0, part + "/");
			if (chat_dock) {
				dir_item->set_icon(0, chat_dock->get_theme_icon(SNAME("Folder"), SNAME("EditorIcons")));
			}

			dir_items.insert(current_dir, dir_item);
			parent = dir_item;
		}

		// Last part is the file name.
		String file_name = parts[parts.size() - 1];
		TreeItem *file_item = folder_tree->create_item(parent);
		file_item->set_text(0, file_name);
		if (chat_dock) {
			file_item->set_icon(0, chat_dock->get_theme_icon(SNAME("File"), SNAME("EditorIcons")));
		}
	}
}

String AIAutoSnapshots::_get_checkpoint_directory() const {
	if (!ProjectSettings::get_singleton()) {
		return String();
	}

	// Keep AIAutoSnapshots in sync with AICheckpointManager: use the same
	// filesystem directory for checkpoints (OS path, not res://).
	if (EditorPaths::get_singleton() && EditorPaths::get_singleton()->are_paths_valid()) {
		String settings_dir_res = EditorPaths::get_singleton()->get_project_settings_dir();
		String settings_dir_fs = ProjectSettings::get_singleton()->globalize_path(settings_dir_res);
		return settings_dir_fs.path_join("ai_checkpoints");
	}

	// Fallback: legacy in‑project location on disk
	String project_root = ProjectSettings::get_singleton()->globalize_path("res://");
	return project_root.path_join(".ai-checkpoints");
}

String AIAutoSnapshots::_format_timestamp_from_unix(int64_t p_unix_time) const {
	Dictionary time_dict = Time::get_singleton()->get_datetime_dict_from_unix_time(p_unix_time);
	String ts = String::num_int64(time_dict["year"]) + "-" +
			String::num_int64(time_dict["month"]).pad_zeros(2) + "-" +
			String::num_int64(time_dict["day"]).pad_zeros(2) + " " +
			String::num_int64(time_dict["hour"]).pad_zeros(2) + ":" +
			String::num_int64(time_dict["minute"]).pad_zeros(2);
	return ts;
}

int64_t AIAutoSnapshots::_parse_message_index_from_tag(const String &p_tag_name) const {
	// Expected format: msg_<index>_<timestamp>
	if (!p_tag_name.begins_with("msg_")) {
		return -1;
	}

	int64_t underscore_pos = p_tag_name.find("_", 4); // after "msg_"
	if (underscore_pos == -1) {
		return -1;
	}

	String index_str = p_tag_name.substr(4, underscore_pos - 4);
	return index_str.to_int();
}


