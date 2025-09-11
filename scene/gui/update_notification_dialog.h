/*
© 2025 Simplifine Corp. Auto-update notification dialog for Orca Engine.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.
*/

#pragma once

#include "scene/gui/dialogs.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/progress_bar.h"
#include "core/variant/dictionary.h"

class UpdateNotificationDialog : public AcceptDialog {
	GDCLASS(UpdateNotificationDialog, AcceptDialog);

public:
	enum UpdateAction {
		UPDATE_ACTION_NONE,
		UPDATE_ACTION_INSTALL_NOW,
		UPDATE_ACTION_INSTALL_LATER,
		UPDATE_ACTION_SKIP_VERSION
	};

private:
	RichTextLabel *release_notes_label = nullptr;
	ProgressBar *download_progress = nullptr;
	Button *install_later_button = nullptr;
	Button *skip_version_button = nullptr;
	
	String current_version;
	String latest_version;
	String download_url;
	String release_notes;
	bool is_downloading = false;
	bool is_installing = false;
	
	UpdateAction selected_action = UPDATE_ACTION_NONE;
	
	void _on_install_now_pressed();
	void _on_install_later_pressed();
	void _on_skip_version_pressed();
	void _setup_ui();

protected:
	void _notification(int p_what);
	static void _bind_methods();
	virtual void ok_pressed() override;

public:
	UpdateNotificationDialog();
	~UpdateNotificationDialog();
	
	// Configuration
	void set_update_info(const Dictionary &update_info);
	void set_current_version(const String &version);
	void set_latest_version(const String &version);
	void set_release_notes(const String &notes);
	void set_download_url(const String &url);
	
	// Progress tracking
	void set_download_progress(float progress);
	void set_downloading(bool downloading);
	void set_installing(bool installing);
	
	// State
	UpdateAction get_selected_action() const;
	String get_current_version() const;
	String get_latest_version() const;
	String get_download_url() const;
	bool is_update_ready() const;
	
	// UI updates
	void update_status_text(const String &status);
	void show_error(const String &error_message);
	void show_success(const String &message);
	
	// Static convenience method
	static void show_update_notification(Node *parent, const Dictionary &update_info);
};

VARIANT_ENUM_CAST(UpdateNotificationDialog::UpdateAction)