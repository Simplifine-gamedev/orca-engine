/* AI Animation Export - Godot Template Export System
 * Handles exporting animations as ready-to-use Godot resources
 */

#ifndef AI_ANIMATION_EXPORT_H
#define AI_ANIMATION_EXPORT_H

#include "core/object/ref_counted.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/box_container.h"
#include "scene/gui/option_button.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/label.h"
#include "scene/gui/check_box.h"
#include "scene/main/http_request.h"
#include "editor/gui/editor_file_dialog.h"

class Control;

class AIAnimationExport : public RefCounted {
	GDCLASS(AIAnimationExport, RefCounted);

public:
	enum ExportMode {
		EXPORT_SIMPLE,      // Just PNG/GIF
		EXPORT_GODOT_TEMPLATE  // Full Godot resources
	};

	enum TemplateType {
		TEMPLATE_CHARACTER,      // CharacterBody2D with movement
		TEMPLATE_RPG_CHARACTER,  // Top-down RPG with 8-dir movement (auto-mirrors LEFT from RIGHT)
		TEMPLATE_EFFECT,         // One-shot effect
		TEMPLATE_PROP,           // Animated prop
		TEMPLATE_SIMPLE          // Minimal setup
	};

private:
	// UI Elements
	ConfirmationDialog *export_dialog = nullptr;
	EditorFileDialog *file_dialog = nullptr;
	HTTPRequest *http_request = nullptr;
	Control *parent_node = nullptr;
	
	// Options UI
	CheckBox *use_template_check = nullptr;  // Checkbox for template export (default: checked)
	OptionButton *mode_option = nullptr;
	OptionButton *template_type_option = nullptr;
	SpinBox *resolution_spin = nullptr;
	SpinBox *fps_spin = nullptr;
	LineEdit *resource_name_edit = nullptr;
	OptionButton *format_option = nullptr;
	VBoxContainer *template_options_container = nullptr;
	Label *info_label = nullptr;
	
	// State
	String pending_project_id;
	String pending_animation_id;
	String pending_export_path;
	ExportMode current_mode = EXPORT_GODOT_TEMPLATE;  // Default to template mode
	
	// Callbacks
	Callable on_export_complete;
	Callable on_export_error;
	
	// Internal methods
	void _create_dialog();
	void _on_mode_changed(int p_index);
	void _on_template_checkbox_toggled(bool p_pressed);
	void _on_dialog_confirmed();
	void _on_file_selected(const String &p_path);
	void _on_folder_selected(const String &p_path);
	void _on_http_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _save_template_files(const Dictionary &p_data, const String &p_folder);
	String _get_api_base_url();
	
protected:
	static void _bind_methods();

public:
	AIAnimationExport();
	~AIAnimationExport();
	
	// Initialize with parent node (for adding HTTP request as child)
	void initialize(Control *p_parent);
	
	// Show export dialog for an animation
	void show_export_dialog(const String &p_project_id, const String &p_animation_id);
	
	// Show export dialog for entire project (all animations)
	void show_project_export_dialog(const String &p_project_id, const Array &p_animation_ids);
	
	// Set callbacks
	void set_on_export_complete(const Callable &p_callback);
	void set_on_export_error(const Callable &p_callback);
	
	// Static helper to trigger quick export (no dialog)
	static void quick_export_template(
		Control *p_parent,
		const String &p_project_id,
		const Array &p_animation_ids,
		const String &p_folder_path,
		const String &p_resource_name,
		int p_template_type,
		int p_resolution,
		int p_fps
	);
};

#endif // AI_ANIMATION_EXPORT_H

