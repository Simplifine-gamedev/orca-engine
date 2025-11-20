/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_manual_snapshots.h"
#include "ai_chat_dock.h"
#include "ai_checkpoint_manager.h"
#include "core/io/dir_access.h"
#include "core/os/os.h"
#include "core/os/time.h"
#include "core/config/project_settings.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"
#include "scene/resources/style_box_flat.h"
#include "editor/editor_string_names.h"
#include "editor/git/git_manager.h"

void AIManualSnapshots::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_create_snapshot_confirmed"), &AIManualSnapshots::_on_create_snapshot_confirmed);
	ClassDB::bind_method(D_METHOD("_on_create_snapshot_cancelled"), &AIManualSnapshots::_on_create_snapshot_cancelled);
	ClassDB::bind_method(D_METHOD("_on_snapshot_item_selected"), &AIManualSnapshots::_on_snapshot_item_selected);
	ClassDB::bind_method(D_METHOD("_on_restore_selected_snapshot"), &AIManualSnapshots::_on_restore_selected_snapshot);
	ClassDB::bind_method(D_METHOD("_on_delete_selected_snapshot"), &AIManualSnapshots::_on_delete_selected_snapshot);
	ClassDB::bind_method(D_METHOD("_on_snapshot_restore_requested", "snapshot_tag"), &AIManualSnapshots::_on_snapshot_restore_requested);
	ClassDB::bind_method(D_METHOD("_on_snapshot_delete_requested", "snapshot_tag"), &AIManualSnapshots::_on_snapshot_delete_requested);
}

AIManualSnapshots::AIManualSnapshots() {
}

AIManualSnapshots::~AIManualSnapshots() {
}

void AIManualSnapshots::initialize(AIChatDock *p_chat_dock) {
	chat_dock = p_chat_dock;
	
	if (!chat_dock) {
		return;
	}
	
}

void AIManualSnapshots::show_create_snapshot_dialog() {
	if (!chat_dock) return;
	
	
	// Create dialog if it doesn't exist
	if (!create_snapshot_dialog) {
		create_snapshot_dialog = memnew(ConfirmationDialog);
		create_snapshot_dialog->set_title("Save Project Snapshot");
		create_snapshot_dialog->set_min_size(Size2(500, 300));
		
		VBoxContainer *dialog_vbox = memnew(VBoxContainer);
		create_snapshot_dialog->add_child(dialog_vbox);
		
		// Header label
		Label *header_label = memnew(Label);
		header_label->set_text("Create a named snapshot of your entire project:");
		header_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
		dialog_vbox->add_child(header_label);
		
		dialog_vbox->add_child(memnew(HSeparator));
		
		// Snapshot name
		Label *name_label = memnew(Label);
		name_label->set_text("Snapshot Name:");
		dialog_vbox->add_child(name_label);
		
		snapshot_name_field = memnew(LineEdit);
		snapshot_name_field->set_placeholder("e.g. 'Working Combat System' or 'Before UI Redesign'");
		snapshot_name_field->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		dialog_vbox->add_child(snapshot_name_field);
		
		// Snapshot description
		Label *desc_label = memnew(Label);
		desc_label->set_text("Description (optional):");
		dialog_vbox->add_child(desc_label);
		
		snapshot_description_field = memnew(TextEdit);
		snapshot_description_field->set_placeholder("What's included in this snapshot? What were you working on?");
		snapshot_description_field->set_custom_minimum_size(Size2(0, 80));
		snapshot_description_field->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		dialog_vbox->add_child(snapshot_description_field);
		
		// Info label
		Label *info_label = memnew(Label);
		info_label->set_text("ℹ️  Snapshots capture ALL files: scenes, scripts, resources, and assets.");
		info_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
		info_label->set_modulate(Color(0.8, 0.8, 0.8));
		dialog_vbox->add_child(info_label);
		
		// Connect signals
		create_snapshot_dialog->connect("confirmed", callable_mp(this, &AIManualSnapshots::_on_create_snapshot_confirmed));
		create_snapshot_dialog->connect("canceled", callable_mp(this, &AIManualSnapshots::_on_create_snapshot_cancelled));
		
		// Add to chat dock's scene tree
		chat_dock->add_child(create_snapshot_dialog);
	}
	
	// Clear previous input
	if (snapshot_name_field) {
		snapshot_name_field->set_text("");
		snapshot_name_field->grab_focus();
	}
	if (snapshot_description_field) {
		snapshot_description_field->set_text("");
	}
	
	// Show dialog
	create_snapshot_dialog->popup_centered();
}

void AIManualSnapshots::show_snapshots_list_dialog() {
	if (!chat_dock) return;
	
	
	// Create dialog if it doesn't exist
	if (!snapshots_list_window) {
		snapshots_list_window = memnew(Window);
		snapshots_list_window->set_title("Project Snapshots");
    snapshots_list_window->set_min_size(Size2(800, 560));
    // Disable wrap_controls to avoid children overlapping; we'll manage sizes via size flags
    snapshots_list_window->set_wrap_controls(false);
		
		VBoxContainer *main_vbox = memnew(VBoxContainer);
		snapshots_list_window->add_child(main_vbox);
		
		// Header
    Label *header_label = memnew(Label);
		header_label->set_text("Your Saved Project Snapshots");
		header_label->add_theme_font_override("font", chat_dock->get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		header_label->add_theme_font_size_override("font_size", 16);
    header_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		main_vbox->add_child(header_label);
		
		main_vbox->add_child(memnew(HSeparator));
		
		// Snapshots tree
    snapshots_tree = memnew(Tree);
		snapshots_tree->set_columns(3);
		snapshots_tree->set_column_title(0, "Snapshot Name");
		snapshots_tree->set_column_title(1, "Created");
		snapshots_tree->set_column_title(2, "Description");
		snapshots_tree->set_column_titles_visible(true);
		snapshots_tree->set_hide_root(true);
    // Make the tree consume most of the window height and be clearly scrollable
    snapshots_tree->set_custom_minimum_size(Size2(0, 320));
    snapshots_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
    snapshots_tree->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		snapshots_tree->set_select_mode(Tree::SELECT_SINGLE);
		snapshots_tree->connect("item_selected", callable_mp(this, &AIManualSnapshots::_on_snapshot_item_selected));
		main_vbox->add_child(snapshots_tree);
		
		// Details panel (shows description of selected snapshot)
    PanelContainer *details_panel = memnew(PanelContainer);
    // Keep details compact so it doesn't cover the tree
    details_panel->set_custom_minimum_size(Size2(0, 96));
		main_vbox->add_child(details_panel);
		
		Ref<StyleBoxFlat> details_style = memnew(StyleBoxFlat);
		details_style->set_bg_color(chat_dock->get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
		details_style->set_content_margin_all(8);
		details_panel->add_theme_style_override("panel", details_style);
		
		ScrollContainer *details_scroll = memnew(ScrollContainer);
		details_panel->add_child(details_scroll);
		
		snapshot_details_label = memnew(Label);
		snapshot_details_label->set_text("Select a snapshot to see details");
		snapshot_details_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    snapshot_details_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    // Do not let details expand vertically too much
    snapshot_details_label->set_v_size_flags(Control::SIZE_FILL);
		details_scroll->add_child(snapshot_details_label);
		
		main_vbox->add_child(memnew(HSeparator));
		
		// Action buttons
    HBoxContainer *buttons_container = memnew(HBoxContainer);
    buttons_container->set_alignment(BoxContainer::ALIGNMENT_END);
		main_vbox->add_child(buttons_container);
		
		// Restore button
		restore_button = memnew(Button);
		restore_button->set_text("Restore Selected");
		restore_button->add_theme_icon_override("icon", chat_dock->get_theme_icon(SNAME("Reload"), SNAME("EditorIcons")));
		restore_button->add_theme_color_override("font_color", chat_dock->get_theme_color(SNAME("success_color"), SNAME("Editor")));
		restore_button->set_disabled(true); // Enabled when snapshot is selected
		restore_button->connect("pressed", callable_mp(this, &AIManualSnapshots::_on_restore_selected_snapshot));
		buttons_container->add_child(restore_button);
		
		// Delete button
		delete_button = memnew(Button);
		delete_button->set_text("Delete");
		delete_button->add_theme_icon_override("icon", chat_dock->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
		delete_button->add_theme_color_override("font_color", chat_dock->get_theme_color(SNAME("error_color"), SNAME("Editor")));
		delete_button->set_disabled(true); // Enabled when snapshot is selected
		delete_button->connect("pressed", callable_mp(this, &AIManualSnapshots::_on_delete_selected_snapshot));
		buttons_container->add_child(delete_button);
		
		// Close button
		Button *close_button = memnew(Button);
		close_button->set_text("Close");
		close_button->connect("pressed", callable_mp((Window *)snapshots_list_window, &Window::hide));
		buttons_container->add_child(close_button);
		
		// Add to chat dock's scene tree
		chat_dock->add_child(snapshots_list_window);
	}
	
	// Refresh the list
	_refresh_snapshots_list();
	
	// Show dialog
	snapshots_list_window->popup_centered();
}

bool AIManualSnapshots::create_manual_snapshot(const String &p_name, const String &p_description) {
	if (!chat_dock) {
		return false;
	}
	
	String name = p_name.strip_edges();
	if (name.is_empty()) {
		return false;
	}
	
	// Check if git is available
	if (!GitManager::is_git_available()) {
		print_line("Manual snapshots require Git. Please install Git for Windows to enable this feature.");
		return false;
	}
	
	
	// Get project root
	String project_root = ProjectSettings::get_singleton()->globalize_path("res://");
	
	// Create checkpoint using AICheckpointManager (same as auto-checkpoints)
	// But use a special message index (-1) to distinguish manual snapshots
	AICheckpointManager::CheckpointResult result = 
		AICheckpointManager::create_comprehensive_checkpoint(project_root, "Manual Snapshot: " + name, -1);
	
	if (!result.success) {
		return false;
	}
	
	// Now we need to rename the tag from msg_-1_timestamp to snapshot_timestamp
	// and add metadata for name and description
	String generated_tag = result.checkpoint_tag; // This will be msg_-1_timestamp
	String snapshot_tag = _generate_snapshot_tag_name(); // Generate snapshot_timestamp
	
	// Use git operations to rename tag and add metadata
	List<String> tag_args;
	tag_args.push_back("tag");
	tag_args.push_back("-f");
	tag_args.push_back(snapshot_tag);
	tag_args.push_back("-m");
	
	// Tag message includes name and description for easy retrieval
	String tag_message = "SNAPSHOT:" + name + "\nDESC:" + p_description;
	tag_args.push_back(tag_message);
	tag_args.push_back(generated_tag); // Tag the same commit
	
	GitManager::GitResult tag_result = GitManager::execute_git_command(project_root, tag_args);
	if (!tag_result.success) {
		return false;
	}
	
	// Delete the temporary msg_-1_* tag
	List<String> delete_temp_args;
	delete_temp_args.push_back("tag");
	delete_temp_args.push_back("-d");
	delete_temp_args.push_back(generated_tag);
	
	GitManager::execute_git_command(project_root, delete_temp_args); // Ignore result
	
	return true;
}

bool AIManualSnapshots::restore_to_snapshot(const String &p_snapshot_tag) {
	if (!chat_dock || p_snapshot_tag.is_empty()) {
		return false;
	}
	
	
	// Get project root
	String project_root = ProjectSettings::get_singleton()->globalize_path("res://");
	
	// Use similar restoration logic to AICheckpointManager, but directly to the tag
	// We can reuse most of the restore logic
	
	// Call the checkpoint manager's restore, but we'll pass -1 as message index
	// and patch the tag finding to use our snapshot tag
	AICheckpointManager::RestoreResult result = 
		AICheckpointManager::restore_to_checkpoint(project_root, -1);
	
	// Actually, we need a different approach - let's directly call git reset
	List<String> reset_args;
	reset_args.push_back("reset");
	reset_args.push_back("--hard");
	reset_args.push_back(p_snapshot_tag);
	
	GitManager::GitResult reset_result = GitManager::execute_git_command(project_root, reset_args);
	if (!reset_result.success) {
		return false;
	}
	
	
	// Trigger comprehensive editor refresh
	if (chat_dock) {
		chat_dock->call_deferred("_force_complete_editor_refresh");
	}
	
	return true;
}

bool AIManualSnapshots::delete_snapshot(const String &p_snapshot_tag) {
	if (p_snapshot_tag.is_empty()) {
		return false;
	}
	
	
	// Get project root
	String project_root = ProjectSettings::get_singleton()->globalize_path("res://");
	
	// Delete the git tag
	List<String> delete_args;
	delete_args.push_back("tag");
	delete_args.push_back("-d");
	delete_args.push_back(p_snapshot_tag);
	
	GitManager::GitResult delete_result = GitManager::execute_git_command(project_root, delete_args);
	if (!delete_result.success) {
		return false;
	}
	
	return true;
}

Vector<AIManualSnapshots::Snapshot> AIManualSnapshots::get_all_snapshots() {
	Vector<Snapshot> snapshots;
	
	if (!chat_dock) {
		return snapshots;
	}
	
	// Get project root
	String project_root = ProjectSettings::get_singleton()->globalize_path("res://");
	
	// List all snapshot tags (snapshot_* pattern)
	List<String> list_args;
	list_args.push_back("tag");
	list_args.push_back("--list");
	list_args.push_back("snapshot_*");
	list_args.push_back("--sort=-creatordate"); // Newest first
	list_args.push_back("--format=%(refname:short)|%(creatordate:unix)|%(contents:subject)|%(contents:body)");
	
	GitManager::GitResult list_result = GitManager::execute_git_command(project_root, list_args);
	if (!list_result.success || list_result.output.strip_edges().is_empty()) {
		return snapshots;
	}
	
	String output = list_result.output;
	
	// Parse the output
	PackedStringArray lines = output.strip_edges().split("\n");
	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i].strip_edges();
		if (line.is_empty()) continue;
		
		PackedStringArray parts = line.split("|");
		if (parts.size() < 4) continue;
		
		Snapshot snapshot;
		snapshot.tag_name = parts[0];
		snapshot.created_unix_time = parts[1].to_int();
		
		// Parse the tag message to extract name and description
		String subject = parts[2];
		String body = parts[3];
		
		// Subject format: "SNAPSHOT:name"
		if (subject.begins_with("SNAPSHOT:")) {
			snapshot.display_name = subject.substr(9); // Remove "SNAPSHOT:" prefix
		} else {
			snapshot.display_name = subject;
		}
		
		// Body format: "DESC:description"
		if (body.begins_with("DESC:")) {
			snapshot.description = body.substr(5); // Remove "DESC:" prefix
		} else {
			snapshot.description = body;
		}
		
		// Convert unix timestamp to readable format
		Dictionary time_dict = Time::get_singleton()->get_datetime_dict_from_unix_time(snapshot.created_unix_time);
		snapshot.created_timestamp = String::num_int64(time_dict["year"]) + "-" +
									  String::num_int64(time_dict["month"]).pad_zeros(2) + "-" +
									  String::num_int64(time_dict["day"]).pad_zeros(2) + " " +
									  String::num_int64(time_dict["hour"]).pad_zeros(2) + ":" +
									  String::num_int64(time_dict["minute"]).pad_zeros(2);
		
		snapshots.push_back(snapshot);
	}
	
	return snapshots;
}

// Private callback handlers

void AIManualSnapshots::_on_create_snapshot_confirmed() {
	if (!snapshot_name_field || !snapshot_description_field) return;
	
	String name = snapshot_name_field->get_text().strip_edges();
	String description = snapshot_description_field->get_text().strip_edges();
	
	if (name.is_empty()) {
		return;
	}
	
	bool success = create_manual_snapshot(name, description);
	
	if (success) {
		
		// Show success notification
		if (chat_dock) {
			chat_dock->call("_show_status_notification", "success", "Snapshot saved: " + name, "💾", 3.0);
		}
	} else {
		
		// Show error notification
		if (chat_dock) {
			chat_dock->call("_show_status_notification", "connection_error", "Failed to save snapshot", "❌", 3.0);
		}
	}
}

void AIManualSnapshots::_on_create_snapshot_cancelled() {
}

void AIManualSnapshots::_on_snapshot_item_selected() {
	if (!snapshots_tree || !snapshot_details_label || !restore_button || !delete_button) return;
	
	TreeItem *selected = snapshots_tree->get_selected();
	if (!selected) return;
	
	String tag_name = selected->get_metadata(0);
	String description = selected->get_text(2);
	
	// Update details label
	snapshot_details_label->set_text(description.is_empty() ? "(No description provided)" : description);
	
	// Enable action buttons
	restore_button->set_disabled(false);
	delete_button->set_disabled(false);
}

void AIManualSnapshots::_on_restore_selected_snapshot() {
	if (!snapshots_tree) return;
	
	TreeItem *selected = snapshots_tree->get_selected();
	if (!selected) return;
	
	String snapshot_tag = selected->get_metadata(0);
	String snapshot_name = selected->get_text(0);
	
	// Show confirmation dialog
	ConfirmationDialog *confirm = memnew(ConfirmationDialog);
	confirm->set_title("Restore Snapshot");
	confirm->set_text("This will restore your ENTIRE project to snapshot:\n\n\"" + snapshot_name + "\"\n\nAll current uncommitted changes will be lost.\nThis action cannot be undone.\n\nContinue?");
	
	// Store snapshot tag in metadata for the confirmation callback
	confirm->set_meta("snapshot_tag", snapshot_tag);
	confirm->connect("confirmed", callable_mp(this, &AIManualSnapshots::_on_snapshot_restore_requested).bind(snapshot_tag));
	
	if (chat_dock) {
		chat_dock->add_child(confirm);
	}
	
	confirm->popup_centered(Size2(450, 250));
	
	// Auto-cleanup after close
	confirm->connect("popup_hide", callable_mp((Node *)confirm, &Node::queue_free));
}

void AIManualSnapshots::_on_delete_selected_snapshot() {
	if (!snapshots_tree) return;
	
	TreeItem *selected = snapshots_tree->get_selected();
	if (!selected) return;
	
	String snapshot_tag = selected->get_metadata(0);
	String snapshot_name = selected->get_text(0);
	
	// Show confirmation dialog
	ConfirmationDialog *confirm = memnew(ConfirmationDialog);
	confirm->set_title("Delete Snapshot");
	confirm->set_text("Are you sure you want to delete snapshot:\n\n\"" + snapshot_name + "\"\n\nThis action cannot be undone.");
	
	confirm->connect("confirmed", callable_mp(this, &AIManualSnapshots::_on_snapshot_delete_requested).bind(snapshot_tag));
	
	if (chat_dock) {
		chat_dock->add_child(confirm);
	}
	
	confirm->popup_centered(Size2(400, 200));
	
	// Auto-cleanup after close
	confirm->connect("popup_hide", callable_mp((Node *)confirm, &Node::queue_free));
}

void AIManualSnapshots::_on_snapshot_restore_requested(const String &p_snapshot_tag) {
	
	bool success = restore_to_snapshot(p_snapshot_tag);
	
	if (success) {
		
		// Show success notification
		if (chat_dock) {
			chat_dock->call("_show_status_notification", "success", "Project restored to snapshot", "✅", 4.0);
		}
		
		// Close the snapshots window
		if (snapshots_list_window) {
			snapshots_list_window->hide();
		}
	} else {
		
		// Show error notification
		if (chat_dock) {
			chat_dock->call("_show_status_notification", "connection_error", "Failed to restore snapshot", "❌", 3.0);
		}
	}
}

void AIManualSnapshots::_on_snapshot_delete_requested(const String &p_snapshot_tag) {
	
	bool success = delete_snapshot(p_snapshot_tag);
	
	if (success) {
		
		// Refresh the snapshots list
		_refresh_snapshots_list();
		
		// Show success notification
		if (chat_dock) {
			chat_dock->call("_show_status_notification", "success", "Snapshot deleted", "🗑️", 2.0);
		}
	} else {
		
		// Show error notification
		if (chat_dock) {
			chat_dock->call("_show_status_notification", "connection_error", "Failed to delete snapshot", "❌", 3.0);
		}
	}
}

// Private helper methods

String AIManualSnapshots::_generate_snapshot_tag_name() {
	String timestamp = Time::get_singleton()->get_datetime_string_from_system();
	timestamp = timestamp.replace(":", "-").replace(" ", "_");
	return "snapshot_" + timestamp;
}

String AIManualSnapshots::_get_selected_snapshot_tag() {
	if (!snapshots_tree) return String();
	
	TreeItem *selected = snapshots_tree->get_selected();
	if (!selected) return String();
	
	return selected->get_metadata(0);
}

void AIManualSnapshots::_refresh_snapshots_list() {
	if (!snapshots_tree) return;
	
	
	// Clear tree
	snapshots_tree->clear();
	TreeItem *root = snapshots_tree->create_item();
	
	// Get all snapshots
	Vector<Snapshot> snapshots = get_all_snapshots();
	
	if (snapshots.is_empty()) {
		// Show empty state
		TreeItem *empty_item = snapshots_tree->create_item(root);
		empty_item->set_text(0, "No snapshots yet");
		empty_item->set_custom_color(0, chat_dock->get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.5));
		empty_item->set_selectable(0, false);
		
		// Update details
		if (snapshot_details_label) {
			snapshot_details_label->set_text("Create your first snapshot using the 'Save Snapshot' button!");
		}
		
		return;
	}
	
	// Populate tree with snapshots
	for (int i = 0; i < snapshots.size(); i++) {
		const Snapshot &snapshot = snapshots[i];
		
		TreeItem *item = snapshots_tree->create_item(root);
		item->set_text(0, snapshot.display_name);
		item->set_text(1, snapshot.created_timestamp);
		
		// Truncate description for column display
		String desc_preview = snapshot.description;
		if (desc_preview.length() > 50) {
			desc_preview = desc_preview.substr(0, 47) + "...";
		}
		item->set_text(2, desc_preview);
		
		// Store full tag name and description in metadata
		item->set_metadata(0, snapshot.tag_name);
		item->set_tooltip_text(2, snapshot.description); // Full description in tooltip
		
		// Add icon
		item->set_icon(0, chat_dock->get_theme_icon(SNAME("Favorites"), SNAME("EditorIcons")));
	}
	
}

