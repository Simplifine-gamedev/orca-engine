/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "scene/gui/dialogs.h"
#include "scene/gui/box_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/tree.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/scroll_container.h"
#include "core/string/ustring.h"

class AIChatDock;

/**
 * AIManualSnapshots - Handles user-created named snapshots separate from auto-checkpoints
 * 
 * This class provides:
 * - Dialog to create named snapshots with descriptions
 * - List view of all manual snapshots
 * - Restore functionality for manual snapshots
 * - Delete functionality for manual snapshots
 * 
 * Manual snapshots use Git tags prefixed with "snapshot_" to distinguish them
 * from auto-generated message checkpoints (prefixed with "msg_")
 */
class AIManualSnapshots : public RefCounted {
	GDCLASS(AIManualSnapshots, RefCounted);

public:
	struct Snapshot {
		String tag_name;           // Git tag name (snapshot_timestamp)
		String display_name;       // User-provided name
		String description;        // User-provided description
		String created_timestamp;  // Human-readable timestamp
		int64_t created_unix_time; // Unix timestamp for sorting
	};

	AIManualSnapshots();
	~AIManualSnapshots();

	// Initialize the snapshot system with reference to chat dock
	void initialize(AIChatDock *p_chat_dock);

	// Show the "Create Snapshot" dialog
	void show_create_snapshot_dialog();

	// Show the "View Snapshots" dialog with list
	void show_snapshots_list_dialog();

	// Create a manual snapshot with user-provided name and description
	bool create_manual_snapshot(const String &p_name, const String &p_description);

	// Restore to a manual snapshot
	bool restore_to_snapshot(const String &p_snapshot_tag);

	// Delete a manual snapshot
	bool delete_snapshot(const String &p_snapshot_tag);

	// Get all manual snapshots (sorted by creation date, newest first)
	Vector<Snapshot> get_all_snapshots();

private:
	static void _bind_methods();

	AIChatDock *chat_dock = nullptr;

	// Dialogs
	ConfirmationDialog *create_snapshot_dialog = nullptr;
	LineEdit *snapshot_name_field = nullptr;
	TextEdit *snapshot_description_field = nullptr;

	Window *snapshots_list_window = nullptr;
	Tree *snapshots_tree = nullptr;
	Button *restore_button = nullptr;
	Button *delete_button = nullptr;
	Label *snapshot_details_label = nullptr;

	// Callback handlers
	void _on_create_snapshot_confirmed();
	void _on_create_snapshot_cancelled();
	void _on_snapshot_item_selected();
	void _on_restore_selected_snapshot();
	void _on_delete_selected_snapshot();
	void _on_snapshot_restore_requested(const String &p_snapshot_tag);
	void _on_snapshot_delete_requested(const String &p_snapshot_tag);

	// Helper methods
	String _generate_snapshot_tag_name();
	String _get_selected_snapshot_tag();
	void _refresh_snapshots_list();
};

