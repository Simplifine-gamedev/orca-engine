/**************************************************************************/
/*  project_dialog.cpp                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "project_dialog.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/zip_io.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "core/version.h"
#include "servers/display_server.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_icons.h"
#include "editor/themes/editor_scale.h"
#include "editor/version_control/editor_vcs_interface.h"
#include "scene/gui/check_box.h"
#include "scene/gui/check_button.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "scene/gui/separator.h"
#include "scene/gui/texture_rect.h"

void ProjectDialog::_set_message(const String &p_msg, MessageType p_type, InputType p_input_type) {
	msg->set_text(p_msg);
	get_ok_button()->set_disabled(p_type == MESSAGE_ERROR);

	Ref<Texture2D> new_icon;
	switch (p_type) {
		case MESSAGE_ERROR: {
			msg->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
			new_icon = get_editor_theme_icon(SNAME("StatusError"));
		} break;
		case MESSAGE_WARNING: {
			msg->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("warning_color"), EditorStringName(Editor)));
			new_icon = get_editor_theme_icon(SNAME("StatusWarning"));
		} break;
		case MESSAGE_SUCCESS: {
			msg->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("success_color"), EditorStringName(Editor)));
			new_icon = get_editor_theme_icon(SNAME("StatusSuccess"));
		} break;
	}

	if (p_input_type == PROJECT_PATH) {
		project_status_rect->set_texture(new_icon);
	} else if (p_input_type == INSTALL_PATH) {
		install_status_rect->set_texture(new_icon);
	}
}

static bool is_zip_file(Ref<DirAccess> p_d, const String &p_path) {
	return p_path.get_extension() == "zip" && p_d->file_exists(p_path);
}

void ProjectDialog::_validate_path() {
	_set_message("", MESSAGE_SUCCESS, PROJECT_PATH);
	_set_message("", MESSAGE_SUCCESS, INSTALL_PATH);

	if (project_name->get_text().strip_edges().is_empty()) {
		_set_message(TTRC("It would be a good idea to name your project."), MESSAGE_ERROR);
		return;
	}

	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	String path = project_path->get_text().simplify_path();

	String target_path = path;
	InputType target_path_input_type = PROJECT_PATH;

	if (mode == MODE_IMPORT) {
		if (path.get_file().strip_edges() == "project.godot") {
			path = path.get_base_dir();
			project_path->set_text(path);
		}

		if (is_zip_file(d, path)) {
			zip_path = path;
		} else if (is_zip_file(d, path.strip_edges())) {
			zip_path = path.strip_edges();
		} else {
			zip_path = "";
		}

		if (!zip_path.is_empty()) {
			target_path = install_path->get_text().simplify_path();
			target_path_input_type = INSTALL_PATH;

			create_dir->show();
			install_path_container->show();

			Ref<FileAccess> io_fa;
			zlib_filefunc_def io = zipio_create_io(&io_fa);

			unzFile pkg = unzOpen2(zip_path.utf8().get_data(), &io);
			if (!pkg) {
				_set_message(TTRC("Invalid \".zip\" project file; it is not in ZIP format."), MESSAGE_ERROR);
				unzClose(pkg);
				return;
			}

			int ret = unzGoToFirstFile(pkg);
			while (ret == UNZ_OK) {
				unz_file_info info;
				char fname[16384];
				ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);
				ERR_FAIL_COND_MSG(ret != UNZ_OK, "Failed to get current file info.");

				String name = String::utf8(fname);

				// Skip the __MACOSX directory created by macOS's built-in file zipper.
				if (name.begins_with("__MACOSX")) {
					ret = unzGoToNextFile(pkg);
					continue;
				}

				if (name.get_file() == "project.godot") {
					break; // ret == UNZ_OK.
				}

				ret = unzGoToNextFile(pkg);
			}

			if (ret == UNZ_END_OF_LIST_OF_FILE) {
				_set_message(TTRC("Invalid \".zip\" project file; it doesn't contain a \"project.godot\" file."), MESSAGE_ERROR);
				unzClose(pkg);
				return;
			}

			unzClose(pkg);
		} else if (d->dir_exists(path) && d->file_exists(path.path_join("project.godot"))) {
			zip_path = "";

			create_dir->hide();
			install_path_container->hide();

			_set_message(TTRC("Valid project found at path."), MESSAGE_SUCCESS);
		} else {
			create_dir->hide();
			install_path_container->hide();

			_set_message(TTRC("Please choose a \"project.godot\", a directory with one, or a \".zip\" file."), MESSAGE_ERROR);
			return;
		}
	}

	if (target_path.is_relative_path()) {
		_set_message(TTRC("The path specified is invalid."), MESSAGE_ERROR, target_path_input_type);
		return;
	}

	if (target_path.get_file() != OS::get_singleton()->get_safe_dir_name(target_path.get_file())) {
		_set_message(TTRC("The directory name specified contains invalid characters or trailing whitespace."), MESSAGE_ERROR, target_path_input_type);
		return;
	}

	String working_dir = d->get_current_dir();
	String executable_dir = OS::get_singleton()->get_executable_path().get_base_dir();
	if (target_path == working_dir || target_path == executable_dir) {
		_set_message(TTRC("Creating a project at the engine's working directory or executable directory is not allowed, as it would prevent the project manager from starting."), MESSAGE_ERROR, target_path_input_type);
		return;
	}

	// TODO: The following 5 lines could be simplified if OS.get_user_home_dir() or SYSTEM_DIR_HOME is implemented. See: https://github.com/godotengine/godot-proposals/issues/4851.
#ifdef WINDOWS_ENABLED
	String home_dir = OS::get_singleton()->get_environment("USERPROFILE");
#else
	String home_dir = OS::get_singleton()->get_environment("HOME");
#endif
	String documents_dir = OS::get_singleton()->get_system_dir(OS::SYSTEM_DIR_DOCUMENTS);
	if (target_path == home_dir || target_path == documents_dir) {
		_set_message(TTRC("You cannot save a project at the selected path. Please create a subfolder or choose a new path."), MESSAGE_ERROR, target_path_input_type);
		return;
	}

	is_folder_empty = true;
	if (mode == MODE_NEW || mode == MODE_INSTALL || mode == MODE_DUPLICATE || (mode == MODE_IMPORT && target_path_input_type == InputType::INSTALL_PATH)) {
		if (create_dir->is_pressed()) {
			if (!d->dir_exists(target_path.get_base_dir())) {
				_set_message(TTRC("The parent directory of the path specified doesn't exist."), MESSAGE_ERROR, target_path_input_type);
				return;
			}

			if (d->dir_exists(target_path)) {
				// The path is not necessarily empty here, but we will update the message later if it isn't.
				_set_message(TTRC("The project folder already exists and is empty."), MESSAGE_SUCCESS, target_path_input_type);
			} else {
				_set_message(TTRC("The project folder will be automatically created."), MESSAGE_SUCCESS, target_path_input_type);
			}
		} else {
			if (!d->dir_exists(target_path)) {
				_set_message(TTRC("The path specified doesn't exist."), MESSAGE_ERROR, target_path_input_type);
				return;
			}

			// The path is not necessarily empty here, but we will update the message later if it isn't.
			_set_message(TTRC("The project folder exists and is empty."), MESSAGE_SUCCESS, target_path_input_type);
		}

		// Check if the directory is empty. Not an error, but we want to warn the user.
		if (d->change_dir(target_path) == OK) {
			d->list_dir_begin();
			String n = d->get_next();
			while (!n.is_empty()) {
				if (n[0] != '.') {
					// Allow `.`, `..` (reserved current/parent folder names)
					// and hidden files/folders to be present.
					// For instance, this lets users initialize a Git repository
					// and still be able to create a project in the directory afterwards.
					is_folder_empty = false;
					break;
				}
				n = d->get_next();
			}
			d->list_dir_end();

			if (!is_folder_empty) {
				_set_message(TTRC("The selected path is not empty. Choosing an empty folder is highly recommended."), MESSAGE_WARNING, target_path_input_type);
			}
		}
	}
}

String ProjectDialog::_get_target_path() {
	if (mode == MODE_NEW || mode == MODE_INSTALL || mode == MODE_DUPLICATE) {
		return project_path->get_text();
	} else if (mode == MODE_IMPORT) {
		return install_path->get_text();
	} else {
		ERR_FAIL_V("");
	}
}
void ProjectDialog::_set_target_path(const String &p_text) {
	if (mode == MODE_NEW || mode == MODE_INSTALL || mode == MODE_DUPLICATE) {
		project_path->set_text(p_text);
	} else if (mode == MODE_IMPORT) {
		install_path->set_text(p_text);
	} else {
		ERR_FAIL();
	}
}

void ProjectDialog::_update_target_auto_dir() {
	String new_auto_dir;
	if (mode == MODE_NEW || mode == MODE_INSTALL || mode == MODE_DUPLICATE) {
		new_auto_dir = project_name->get_text();
	} else if (mode == MODE_IMPORT) {
		new_auto_dir = project_path->get_text().get_file().get_basename();
	}
	int naming_convention = (int)EDITOR_GET("project_manager/directory_naming_convention");
	switch (naming_convention) {
		case 0: // No convention
			break;
		case 1: // kebab-case
			new_auto_dir = new_auto_dir.to_kebab_case();
			break;
		case 2: // snake_case
			new_auto_dir = new_auto_dir.to_snake_case();
			break;
		case 3: // camelCase
			new_auto_dir = new_auto_dir.to_camel_case();
			break;
		case 4: // PascalCase
			new_auto_dir = new_auto_dir.to_pascal_case();
			break;
		case 5: // Title Case
			new_auto_dir = new_auto_dir.capitalize();
			break;
		default:
			ERR_FAIL_MSG("Invalid directory naming convention.");
			break;
	}
	new_auto_dir = OS::get_singleton()->get_safe_dir_name(new_auto_dir);

	if (create_dir->is_pressed()) {
		String target_path = _get_target_path();

		if (target_path.get_file() == auto_dir) {
			// Update target dir name to new project name / ZIP name.
			target_path = target_path.get_base_dir().path_join(new_auto_dir);
		}

		_set_target_path(target_path);
	}

	auto_dir = new_auto_dir;
}

void ProjectDialog::_create_dir_toggled(bool p_pressed) {
	String target_path = _get_target_path();

	if (create_dir->is_pressed()) {
		// (Re-)append target dir name.
		if (last_custom_target_dir.is_empty()) {
			target_path = target_path.path_join(auto_dir);
		} else {
			target_path = target_path.path_join(last_custom_target_dir);
		}
	} else {
		// Strip any trailing slash.
		target_path = target_path.rstrip("/\\");
		// Save and remove target dir name.
		if (target_path.get_file() == auto_dir) {
			last_custom_target_dir = "";
		} else {
			last_custom_target_dir = target_path.get_file();
		}
		target_path = target_path.get_base_dir();
	}

	_set_target_path(target_path);
	_validate_path();
}

void ProjectDialog::_project_name_changed() {
	if (mode == MODE_NEW || mode == MODE_INSTALL || mode == MODE_DUPLICATE) {
		_update_target_auto_dir();
	}

	_validate_path();
}

void ProjectDialog::_project_path_changed() {
	if (mode == MODE_IMPORT) {
		_update_target_auto_dir();
	}

	_validate_path();
}

void ProjectDialog::_install_path_changed() {
	_validate_path();
}

void ProjectDialog::_browse_project_path() {
	String path = project_path->get_text();
	if (path.is_relative_path()) {
		path = EDITOR_GET("filesystem/directories/default_project_path");
	}
	if (mode == MODE_IMPORT && install_path->is_visible_in_tree()) {
		// Select last ZIP file.
		fdialog_project->set_current_path(path);
	} else if ((mode == MODE_NEW || mode == MODE_INSTALL || mode == MODE_DUPLICATE) && create_dir->is_pressed()) {
		// Select parent directory of project path.
		fdialog_project->set_current_dir(path.get_base_dir());
	} else {
		// Select project path.
		fdialog_project->set_current_dir(path);
	}

	if (mode == MODE_IMPORT) {
		fdialog_project->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_ANY);
		fdialog_project->clear_filters();
		fdialog_project->add_filter("project.godot", vformat("%s %s", GODOT_VERSION_NAME, TTR("Project")));
		fdialog_project->add_filter("*.zip", TTR("ZIP File"));
	} else {
		fdialog_project->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_DIR);
	}

	hide();
	fdialog_project->popup_file_dialog();
}

void ProjectDialog::_browse_install_path() {
	ERR_FAIL_COND_MSG(mode != MODE_IMPORT, "Install path is only used for MODE_IMPORT.");

	String path = install_path->get_text();
	if (path.is_relative_path() || !DirAccess::dir_exists_absolute(path)) {
		path = EDITOR_GET("filesystem/directories/default_project_path");
	}
	if (create_dir->is_pressed()) {
		// Select parent directory of install path.
		fdialog_install->set_current_dir(path.get_base_dir());
	} else {
		// Select install path.
		fdialog_install->set_current_dir(path);
	}

	fdialog_install->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_DIR);
	fdialog_install->popup_file_dialog();
}

void ProjectDialog::_project_path_selected(const String &p_path) {
	show_dialog(false);

	if (create_dir->is_pressed() && (mode == MODE_NEW || mode == MODE_INSTALL || mode == MODE_DUPLICATE)) {
		// Replace parent directory, but keep target dir name.
		project_path->set_text(p_path.path_join(project_path->get_text().get_file()));
	} else {
		project_path->set_text(p_path);
	}

	_project_path_changed();

	if (install_path->is_visible_in_tree()) {
		// ZIP is selected; focus install path.
		install_path->grab_focus();
	} else {
		get_ok_button()->grab_focus();
	}
}

void ProjectDialog::_install_path_selected(const String &p_path) {
	ERR_FAIL_COND_MSG(mode != MODE_IMPORT, "Install path is only used for MODE_IMPORT.");

	if (create_dir->is_pressed()) {
		// Replace parent directory, but keep target dir name.
		install_path->set_text(p_path.path_join(install_path->get_text().get_file()));
	} else {
		install_path->set_text(p_path);
	}

	_install_path_changed();

	get_ok_button()->grab_focus();
}

void ProjectDialog::_reset_name() {
	project_name->set_text(TTR("New Game Project"));
}

void ProjectDialog::_renderer_selected() {
	ERR_FAIL_NULL(renderer_button_group->get_pressed_button());

	String renderer_type = renderer_button_group->get_pressed_button()->get_meta(SNAME("rendering_method"));

	bool rd_error = false;
	
	// ORCA ENGINE FIX: On Windows with RD_ENABLED, allow Forward+ and Mobile even if runtime test failed
	// Match the same logic used when creating the renderer buttons
	#ifdef WINDOWS_ENABLED
	#ifdef RD_ENABLED
	bool allow_rd_renderers = true;
	#else
	bool allow_rd_renderers = rendering_device_supported;
	#endif
	#else
	bool allow_rd_renderers = rendering_device_supported;
	#endif

	if (renderer_type == "forward_plus") {
		renderer_info->set_text(
				String::utf8("•  ") + TTR("Supports desktop platforms only.") +
				String::utf8("\n•  ") + TTR("Advanced 3D graphics available.") +
				String::utf8("\n•  ") + TTR("Can scale to large complex scenes.") +
				String::utf8("\n•  ") + TTR("Uses RenderingDevice backend.") +
				String::utf8("\n•  ") + TTR("Slower rendering of simple scenes."));
		rd_error = !allow_rd_renderers;
	} else if (renderer_type == "mobile") {
		renderer_info->set_text(
				String::utf8("•  ") + TTR("Supports desktop + mobile platforms.") +
				String::utf8("\n•  ") + TTR("Less advanced 3D graphics.") +
				String::utf8("\n•  ") + TTR("Less scalable for complex scenes.") +
				String::utf8("\n•  ") + TTR("Uses RenderingDevice backend.") +
				String::utf8("\n•  ") + TTR("Fast rendering of simple scenes."));
		rd_error = !allow_rd_renderers;
	} else if (renderer_type == "gl_compatibility") {
		renderer_info->set_text(
				String::utf8("•  ") + TTR("Supports desktop, mobile + web platforms.") +
				String::utf8("\n•  ") + TTR("Least advanced 3D graphics.") +
				String::utf8("\n•  ") + TTR("Intended for low-end/older devices.") +
				String::utf8("\n•  ") + TTR("Uses OpenGL 3 backend (OpenGL 3.3/ES 3.0/WebGL2).") +
				String::utf8("\n•  ") + TTR("Fastest rendering of simple scenes."));
	} else {
		WARN_PRINT("Unknown renderer type. Please report this as a bug on GitHub.");
	}

	rd_not_supported->set_visible(rd_error);
	get_ok_button()->set_disabled(rd_error);
	if (rd_error) {
		// Needs to be set here since theme colors aren't available at startup.
		rd_not_supported->add_theme_color_override(SceneStringName(font_color), get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
	}
}

void ProjectDialog::_nonempty_confirmation_ok_pressed() {
	is_folder_empty = true;
	ok_pressed();
}

void ProjectDialog::ok_pressed() {
	// Before we create a project, check that the target folder is empty.
	// If not, we need to ask the user if they're sure they want to do this.
	if (!is_folder_empty) {
		if (!nonempty_confirmation) {
			nonempty_confirmation = memnew(ConfirmationDialog);
			nonempty_confirmation->set_title(TTRC("Warning: This folder is not empty"));
			nonempty_confirmation->set_text(TTRC("You are about to create a Godot project in a non-empty folder.\nThe entire contents of this folder will be imported as project resources!\n\nAre you sure you wish to continue?"));
			nonempty_confirmation->get_ok_button()->connect(SceneStringName(pressed), callable_mp(this, &ProjectDialog::_nonempty_confirmation_ok_pressed));
			add_child(nonempty_confirmation);
		}
		nonempty_confirmation->popup_centered();
		return;
	}

	String path = project_path->get_text();

	if (mode == MODE_NEW) {
		// Check if this is a template project
		bool is_template = has_meta("is_template") && get_meta("is_template");
		if (is_template) {
			String repo_url = get_meta("template_repo_url", "");
			String subdir = get_meta("template_subdir", "");
			
			if (!repo_url.is_empty()) {
				// Check if already cloning
				if (is_cloning) {
					_set_message(TTRC("Template clone already in progress. Please wait."), MESSAGE_ERROR);
					return;
				}
				
				// Disable OK button during operation
				get_ok_button()->set_disabled(true);
				is_cloning = true;
				
				// Show progress message
				_show_progress(TTR("Cloning template repository... This may take a moment."));
				
				// Prepare clone task
				clone_task = memnew(CloneTaskData);
				clone_task->dialog = this;
				clone_task->repo_url = repo_url;
				clone_task->temp_clone_dir = OS::get_singleton()->get_cache_path().path_join("template_clone_" + String::num_int64(OS::get_singleton()->get_ticks_msec()));
				clone_task->project_path = path;
				clone_task->create_dir = create_dir->is_pressed();
				
				// Get subdir
				Variant subdir_variant = get_meta("template_subdir", "");
				if (subdir_variant.get_type() != Variant::NIL) {
					String subdir_str = String(subdir_variant);
					if (!subdir_str.is_empty() && 
					    subdir_str != "null" && 
					    subdir_str != "<null>" && 
					    subdir_str != "None" && 
					    subdir_str != "nil") {
						clone_task->subdir = subdir_str;
					}
				}
				
				// Find git executable
				clone_task->git_executable = "git";
#ifdef WINDOWS_ENABLED
				clone_task->git_executable = "git.exe";
#endif
				
				// Prepare clone args
				clone_task->clone_args.push_back("clone");
				clone_task->clone_args.push_back("--depth");
				clone_task->clone_args.push_back("1");
				clone_task->clone_args.push_back(repo_url);
				clone_task->clone_args.push_back(clone_task->temp_clone_dir);
				
				// Start clone in background thread
				clone_thread = memnew(Thread);
				Thread::ID thread_id = clone_thread->start(_clone_thread_func, clone_task);
				
				if (thread_id == Thread::UNASSIGNED_ID || !clone_thread->is_started()) {
					print_line("ERROR: Failed to start clone thread");
					_hide_progress();
					get_ok_button()->set_disabled(false);
					is_cloning = false;
					_set_message(TTRC("Failed to start clone operation. Please try again."), MESSAGE_ERROR);
					memdelete(clone_task);
					clone_task = nullptr;
					memdelete(clone_thread);
					clone_thread = nullptr;
					return;
				}
				
				print_line("DEBUG: Clone thread started successfully with ID: " + itos(thread_id));
				
				// Return early - completion will be handled in _on_clone_complete
				return;
			}
		} else {
			if (create_dir->is_pressed()) {
				Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
				if (!d->dir_exists(path) && d->make_dir(path) != OK) {
					_set_message(TTRC("Couldn't create project directory, check permissions."), MESSAGE_ERROR);
					return;
				}
			}
		}

		// Check if project.godot already exists (from template)
		bool project_godot_exists = FileAccess::exists(path.path_join("project.godot"));
		
		if (!project_godot_exists) {
			PackedStringArray project_features = ProjectSettings::get_required_features();
			ProjectSettings::CustomMap initial_settings;

			// Be sure to change this code if/when renderers are changed.
			// Default values are "forward_plus" for the main setting, "mobile" for the mobile override,
			// and "gl_compatibility" for the web override.
			String renderer_type = renderer_button_group->get_pressed_button()->get_meta(SNAME("rendering_method"));
			initial_settings["rendering/renderer/rendering_method"] = renderer_type;

			EditorSettings::get_singleton()->set("project_manager/default_renderer", renderer_type);
			EditorSettings::get_singleton()->save();

			if (renderer_type == "forward_plus") {
				project_features.push_back("Forward Plus");
			} else if (renderer_type == "mobile") {
				project_features.push_back("Mobile");
			} else if (renderer_type == "gl_compatibility") {
				project_features.push_back("GL Compatibility");
				// Also change the default rendering method for the mobile override.
				initial_settings["rendering/renderer/rendering_method.mobile"] = "gl_compatibility";
			} else {
				WARN_PRINT("Unknown renderer type. Please report this as a bug on GitHub.");
			}

			project_features.sort();
			initial_settings["application/config/features"] = project_features;
			initial_settings["application/config/name"] = project_name->get_text().strip_edges();
			initial_settings["application/config/icon"] = "res://icon.svg";

			Error err = ProjectSettings::get_singleton()->save_custom(path.path_join("project.godot"), initial_settings, Vector<String>(), false);
			if (err != OK) {
				_set_message(TTRC("Couldn't create project.godot in project path."), MESSAGE_ERROR);
				return;
			}
		} else {
			// Update project name in existing project.godot if it's from a template
			Ref<ConfigFile> config = memnew(ConfigFile);
			Error err = config->load(path.path_join("project.godot"));
			if (err == OK) {
				config->set_value("application", "config/name", project_name->get_text().strip_edges());
				config->save(path.path_join("project.godot"));
			}
		}

		// Copy Orca logo to project as icon
		String icon_source = "res://orcabranding/Logo.svg";
		Error err;
		Ref<FileAccess> fa_icon_src = FileAccess::open(icon_source, FileAccess::READ, &err);
		if (err == OK) {
			String svg_content = fa_icon_src->get_as_utf8_string();
			Ref<FileAccess> fa_icon = FileAccess::open(path.path_join("icon.svg"), FileAccess::WRITE, &err);
			if (err == OK) {
				fa_icon->store_string(svg_content);
			}
		} else {
			// Fallback to embedded icon if Logo.svg not found
			Ref<FileAccess> fa_icon = FileAccess::open(path.path_join("icon.svg"), FileAccess::WRITE, &err);
			if (err != OK) {
				_set_message(TTRC("Couldn't create icon.svg in project path."), MESSAGE_ERROR);
				return;
			}
			fa_icon->store_string(get_default_project_icon());
		}

		EditorVCSInterface::create_vcs_metadata_files(EditorVCSInterface::VCSMetadata(vcs_metadata_selection->get_selected()), path);

		// Ensures external editors and IDEs use UTF-8 encoding.
		const String editor_config_path = path.path_join(".editorconfig");
		Ref<FileAccess> f = FileAccess::open(editor_config_path, FileAccess::WRITE);
		if (f.is_null()) {
			// .editorconfig isn't so critical.
			ERR_PRINT("Couldn't create .editorconfig in project path.");
		} else {
			f->store_line("root = true");
			f->store_line("");
			f->store_line("[*]");
			f->store_line("charset = utf-8");
			f->close();
			FileAccess::set_hidden_attribute(editor_config_path, true);
		}
	}

	// Two cases for importing a ZIP.
	switch (mode) {
		case MODE_IMPORT: {
			if (zip_path.is_empty()) {
				break;
			}

			path = install_path->get_text().simplify_path();
			[[fallthrough]];
		}
		case MODE_INSTALL: {
			ERR_FAIL_COND(zip_path.is_empty());

			Ref<FileAccess> io_fa;
			zlib_filefunc_def io = zipio_create_io(&io_fa);

			unzFile pkg = unzOpen2(zip_path.utf8().get_data(), &io);
			if (!pkg) {
				dialog_error->set_text(TTRC("Error opening package file, not in ZIP format."));
				dialog_error->popup_centered();
				return;
			}

			// Find the first directory with a "project.godot".
			String zip_root;
			int ret = unzGoToFirstFile(pkg);
			while (ret == UNZ_OK) {
				unz_file_info info;
				char fname[16384];
				unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);
				ERR_FAIL_COND_MSG(ret != UNZ_OK, "Failed to get current file info.");

				String name = String::utf8(fname);

				// Skip the __MACOSX directory created by macOS's built-in file zipper.
				if (name.begins_with("__MACOSX")) {
					ret = unzGoToNextFile(pkg);
					continue;
				}

				if (name.get_file() == "project.godot") {
					zip_root = name.get_base_dir();
					break;
				}

				ret = unzGoToNextFile(pkg);
			}

			if (ret == UNZ_END_OF_LIST_OF_FILE) {
				_set_message(TTRC("Invalid \".zip\" project file; it doesn't contain a \"project.godot\" file."), MESSAGE_ERROR);
				unzClose(pkg);
				return;
			}

			if (create_dir->is_pressed()) {
				Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
				if (!d->dir_exists(path) && d->make_dir(path) != OK) {
					_set_message(TTRC("Couldn't create project directory, check permissions."), MESSAGE_ERROR);
					return;
				}
			}

			ret = unzGoToFirstFile(pkg);

			Vector<String> failed_files;
			while (ret == UNZ_OK) {
				//get filename
				unz_file_info info;
				char fname[16384];
				ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);
				ERR_FAIL_COND_MSG(ret != UNZ_OK, "Failed to get current file info.");

				String name = String::utf8(fname);

				// Skip the __MACOSX directory created by macOS's built-in file zipper.
				if (name.begins_with("__MACOSX")) {
					ret = unzGoToNextFile(pkg);
					continue;
				}

				String rel_path = name.trim_prefix(zip_root);
				if (rel_path.is_empty()) { // Root.
				} else if (rel_path.ends_with("/")) { // Directory.
					Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
					da->make_dir(path.path_join(rel_path));
				} else { // File.
					Vector<uint8_t> uncomp_data;
					uncomp_data.resize(info.uncompressed_size);

					unzOpenCurrentFile(pkg);
					ret = unzReadCurrentFile(pkg, uncomp_data.ptrw(), uncomp_data.size());
					ERR_BREAK_MSG(ret < 0, vformat("An error occurred while attempting to read from file: %s. This file will not be used.", rel_path));
					unzCloseCurrentFile(pkg);

					Ref<FileAccess> f = FileAccess::open(path.path_join(rel_path), FileAccess::WRITE);
					if (f.is_valid()) {
						f->store_buffer(uncomp_data.ptr(), uncomp_data.size());
					} else {
						failed_files.push_back(rel_path);
					}
				}

				ret = unzGoToNextFile(pkg);
			}

			unzClose(pkg);

			if (failed_files.size()) {
				String err_msg = TTR("The following files failed extraction from package:") + "\n\n";
				for (int i = 0; i < failed_files.size(); i++) {
					if (i > 15) {
						err_msg += "\nAnd " + itos(failed_files.size() - i) + " more files.";
						break;
					}
					err_msg += failed_files[i] + "\n";
				}

				dialog_error->set_text(err_msg);
				dialog_error->popup_centered();
				return;
			}
		} break;
		default: {
		} break;
	}

	if (mode == MODE_DUPLICATE) {
		Ref<DirAccess> dir = DirAccess::open(original_project_path);
		Error err = FAILED;
		if (dir.is_valid()) {
			err = dir->copy_dir(".", path, -1, true);
		}
		if (err != OK) {
			dialog_error->set_text(vformat(TTR("Couldn't duplicate project (error %d)."), err));
			dialog_error->popup_centered();
			return;
		}
	}

	if (mode == MODE_RENAME || mode == MODE_INSTALL || mode == MODE_DUPLICATE) {
		// Load project.godot as ConfigFile to set the new name.
		ConfigFile cfg;
		String project_godot = path.path_join("project.godot");
		Error err = cfg.load(project_godot);
		if (err != OK) {
			dialog_error->set_text(vformat(TTR("Couldn't load project at '%s' (error %d). It may be missing or corrupted."), project_godot, err));
			dialog_error->popup_centered();
			return;
		}
		cfg.set_value("application", "config/name", project_name->get_text().strip_edges());
		err = cfg.save(project_godot);
		if (err != OK) {
			dialog_error->set_text(vformat(TTR("Couldn't save project at '%s' (error %d)."), project_godot, err));
			dialog_error->popup_centered();
			return;
		}
	}

	hide();
	if (mode == MODE_NEW || mode == MODE_IMPORT || mode == MODE_INSTALL) {
#ifdef ANDROID_ENABLED
		// Android 11 has some issues with nomedia files, so it's disabled there. See GH-106479, GH-105399 for details.
		String sdk_version = OS::get_singleton()->get_version().get_slicec('.', 0);
		if (sdk_version != "30") {
			// Create a .nomedia file to hide assets from media apps on Android.
			const String nomedia_file_path = path.path_join(".nomedia");
			Ref<FileAccess> f2 = FileAccess::open(nomedia_file_path, FileAccess::WRITE);
			if (f2.is_null()) {
				// .nomedia isn't so critical.
				ERR_PRINT("Couldn't create .nomedia in project path.");
			} else {
				f2->close();
			}
		}
#endif
		emit_signal(SNAME("project_created"), path, edit_check_box->is_pressed());
	} else if (mode == MODE_DUPLICATE) {
		emit_signal(SNAME("project_duplicated"), original_project_path, path, edit_check_box->is_visible() && edit_check_box->is_pressed());
	} else if (mode == MODE_RENAME) {
		emit_signal(SNAME("projects_updated"));
	}
}

void ProjectDialog::set_zip_path(const String &p_path) {
	zip_path = p_path;
}

void ProjectDialog::set_zip_title(const String &p_title) {
	zip_title = p_title;
}

void ProjectDialog::set_original_project_path(const String &p_path) {
	original_project_path = p_path;
}

void ProjectDialog::set_duplicate_can_edit(bool p_duplicate_can_edit) {
	duplicate_can_edit = p_duplicate_can_edit;
}

void ProjectDialog::set_mode(Mode p_mode) {
	mode = p_mode;
}

void ProjectDialog::set_project_name(const String &p_name) {
	project_name->set_text(p_name);
}

void ProjectDialog::set_project_path(const String &p_path) {
	project_path->set_text(p_path);
}

void ProjectDialog::_show_progress(const String &p_message) {
	if (progress_label) {
		progress_label->set_text(p_message);
		progress_label->show();
		msg->hide(); // Hide error/validation messages while showing progress
		// Force UI update immediately
		progress_label->queue_redraw();
		// Process all pending events
		DisplayServer::get_singleton()->process_events();
		DisplayServer::get_singleton()->force_process_and_drop_events();
	}
}

void ProjectDialog::_hide_progress() {
	if (progress_label) {
		progress_label->hide();
		msg->show(); // Show messages again
	}
}

void ProjectDialog::_clone_thread_func(void *p_userdata) {
	CloneTaskData *task = static_cast<CloneTaskData *>(p_userdata);
	if (!task || !task->dialog) {
		print_line("ERROR: CloneTaskData is null or dialog is null");
		return;
	}
	
	print_line("DEBUG: Starting git clone in thread...");
	print_line("DEBUG: Git executable: " + task->git_executable);
	print_line("DEBUG: Repo URL: " + task->repo_url);
	print_line("DEBUG: Temp dir: " + task->temp_clone_dir);
	
	// Execute git clone in background thread
	task->result = OS::get_singleton()->execute(
		task->git_executable,
		task->clone_args,
		&task->output,
		&task->exit_code,
		true,
		nullptr,
		false
	);
	
	print_line("DEBUG: Git clone completed. Result: " + itos(task->result) + ", Exit code: " + itos(task->exit_code));
	if (!task->output.is_empty()) {
		print_line("DEBUG: Git output: " + task->output.substr(0, 200));
	}
	
	task->done = true;
	
	// Signal completion on main thread using callable_mp
	print_line("DEBUG: Calling _on_clone_complete deferred...");
	callable_mp(task->dialog, &ProjectDialog::_on_clone_complete).call_deferred();
}

void ProjectDialog::_on_clone_complete() {
	print_line("DEBUG: _on_clone_complete called");
	
	if (!clone_task) {
		print_line("ERROR: clone_task is null in _on_clone_complete");
		return;
	}
	
	if (!is_cloning) {
		print_line("ERROR: is_cloning is false in _on_clone_complete");
		return;
	}
	
	is_cloning = false;
	print_line("DEBUG: Processing clone completion...");
	
	// Check clone result
	if (clone_task->result != OK || clone_task->exit_code != 0) {
		_hide_progress();
		get_ok_button()->set_disabled(false);
		String error_msg = TTRC("Failed to clone template repository. Make sure Git is installed and the repository URL is valid.");
		error_msg += "\n" + clone_task->output;
		_set_message(error_msg, MESSAGE_ERROR);
		
		// Clean up thread
		if (clone_thread) {
			clone_thread->wait_to_finish();
			memdelete(clone_thread);
			clone_thread = nullptr;
		}
		memdelete(clone_task);
		clone_task = nullptr;
		return;
	}
	
	_show_progress(TTR("Copying template files..."));
	DisplayServer::get_singleton()->process_events();
	
	// Determine source path (with or without subdir)
	String source_path = clone_task->temp_clone_dir;
	if (!clone_task->subdir.is_empty()) {
		source_path = clone_task->temp_clone_dir.path_join(clone_task->subdir);
		if (!DirAccess::dir_exists_absolute(source_path)) {
			_hide_progress();
			get_ok_button()->set_disabled(false);
			_set_message(TTRC("Template subdirectory not found: ") + clone_task->subdir, MESSAGE_ERROR);
			// Clean up temp directory
			Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
			da->remove(clone_task->temp_clone_dir);
			// Clean up thread
			if (clone_thread) {
				clone_thread->wait_to_finish();
				memdelete(clone_thread);
				clone_thread = nullptr;
			}
			memdelete(clone_task);
			clone_task = nullptr;
			return;
		}
	}
	
	// Create target directory
	if (clone_task->create_dir) {
		Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (!d->dir_exists(clone_task->project_path) && d->make_dir(clone_task->project_path) != OK) {
			_hide_progress();
			get_ok_button()->set_disabled(false);
			_set_message(TTRC("Couldn't create project directory, check permissions."), MESSAGE_ERROR);
			// Clean up temp directory
			Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
			da->remove(clone_task->temp_clone_dir);
			// Clean up thread
			if (clone_thread) {
				clone_thread->wait_to_finish();
				memdelete(clone_thread);
				clone_thread = nullptr;
			}
			memdelete(clone_task);
			clone_task = nullptr;
			return;
		}
	}
	
	// Copy files from cloned repo to project directory
	Ref<DirAccess> copy_da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error copy_err = copy_da->copy_dir(source_path, clone_task->project_path);
	
	if (copy_err != OK) {
		_hide_progress();
		get_ok_button()->set_disabled(false);
		_set_message(TTRC("Failed to copy template files to project directory."), MESSAGE_ERROR);
		// Clean up temp directory
		Ref<DirAccess> cleanup_da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		cleanup_da->remove(clone_task->temp_clone_dir);
		// Clean up thread
		if (clone_thread) {
			clone_thread->wait_to_finish();
			memdelete(clone_thread);
			clone_thread = nullptr;
		}
		memdelete(clone_task);
		clone_task = nullptr;
		return;
	}
	
	// Remove .git directory if it was copied
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	String git_dir = clone_task->project_path.path_join(".git");
	if (da->dir_exists(git_dir)) {
		da->remove(git_dir);
	}
	
	// Clean up temp directory
	Ref<DirAccess> cleanup_da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	cleanup_da->remove(clone_task->temp_clone_dir);
	
	_hide_progress();
	get_ok_button()->set_disabled(false);
	
	// Clean up thread
	if (clone_thread) {
		clone_thread->wait_to_finish();
		memdelete(clone_thread);
		clone_thread = nullptr;
	}
	
	// Remove template metadata
	remove_meta("is_template");
	remove_meta("template_repo_url");
	remove_meta("template_subdir");
	
	String path = clone_task->project_path;
	
	// Continue with normal project creation flow
	// Check if project.godot already exists (from template)
	Ref<DirAccess> da_check = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	bool project_godot_exists = da_check->file_exists(path.path_join("project.godot"));
	
	if (!project_godot_exists) {
		// Create project.godot if template didn't include it
		PackedStringArray project_features = ProjectSettings::get_required_features();
		ProjectSettings::CustomMap initial_settings;
		
		// Get renderer selection
		String renderer_type = "forward_plus";
		if (renderer_button_group.is_valid() && renderer_button_group->get_pressed_button()) {
			renderer_type = renderer_button_group->get_pressed_button()->get_meta(SNAME("rendering_method"));
		}
		
		initial_settings["rendering/renderer/rendering_method"] = renderer_type;
		
		if (renderer_type == "forward_plus") {
			project_features.push_back("Forward Plus");
		} else if (renderer_type == "mobile") {
			project_features.push_back("Mobile");
		} else if (renderer_type == "gl_compatibility") {
			project_features.push_back("GL Compatibility");
			initial_settings["rendering/renderer/rendering_method.mobile"] = "gl_compatibility";
		}
		
		project_features.sort();
		initial_settings["application/config/features"] = project_features;
		initial_settings["application/config/name"] = project_name->get_text().strip_edges();
		initial_settings["application/config/icon"] = "res://icon.svg";
		
		Error err = ProjectSettings::get_singleton()->save_custom(path.path_join("project.godot"), initial_settings, Vector<String>(), false);
		if (err != OK) {
			_set_message(TTRC("Couldn't create project.godot in project path."), MESSAGE_ERROR);
			memdelete(clone_task);
			clone_task = nullptr;
			return;
		}
	} else {
		// Update project name in existing project.godot
		Ref<ConfigFile> config = memnew(ConfigFile);
		Error err = config->load(path.path_join("project.godot"));
		if (err == OK) {
			config->set_value("application", "config/name", project_name->get_text().strip_edges());
			config->save(path.path_join("project.godot"));
		}
	}
	
	// Copy Orca logo to project as icon if not exists
	if (!da_check->file_exists(path.path_join("icon.svg"))) {
		String icon_source = "res://orcabranding/Logo.svg";
		Error err;
		Ref<FileAccess> fa_icon_src = FileAccess::open(icon_source, FileAccess::READ, &err);
		if (err == OK) {
			String svg_content = fa_icon_src->get_as_utf8_string();
			Ref<FileAccess> fa_icon = FileAccess::open(path.path_join("icon.svg"), FileAccess::WRITE, &err);
			if (err == OK) {
				fa_icon->store_string(svg_content);
			}
		} else {
			// Fallback to embedded icon
			Ref<FileAccess> fa_icon = FileAccess::open(path.path_join("icon.svg"), FileAccess::WRITE, &err);
			if (err == OK) {
				fa_icon->store_string(get_default_project_icon());
			}
		}
	}
	
	// Create VCS metadata
	EditorVCSInterface::create_vcs_metadata_files(EditorVCSInterface::VCSMetadata(vcs_metadata_selection->get_selected()), path);
	
	// Create .editorconfig
	const String editor_config_path = path.path_join(".editorconfig");
	if (!da_check->file_exists(editor_config_path)) {
		Ref<FileAccess> f = FileAccess::open(editor_config_path, FileAccess::WRITE);
		if (f.is_valid()) {
			f->store_line("root = true");
			f->store_line("");
			f->store_line("[*]");
			f->store_line("charset = utf-8");
			f->close();
			FileAccess::set_hidden_attribute(editor_config_path, true);
		}
	}
	
	// Clean up task
	memdelete(clone_task);
	clone_task = nullptr;
	
	// Emit signal to open project
	emit_signal(SNAME("project_created"), path, edit_check_box->is_pressed());
	hide();
}

void ProjectDialog::ask_for_path_and_show() {
	_reset_name();
	_browse_project_path();
}

void ProjectDialog::show_dialog(bool p_reset_name) {
	if (mode == MODE_RENAME) {
		// Name and path are set in `ProjectManager::_rename_project`.
		project_path->set_editable(false);

		set_title(TTRC("Rename Project"));
		set_ok_button_text(TTRC("Rename"));

		create_dir->hide();
		project_status_rect->hide();
		project_browse->hide();
		edit_check_box->hide();

		name_container->show();
		install_path_container->hide();
		renderer_container->hide();
		default_files_container->hide();

		callable_mp((Control *)project_name, &Control::grab_focus).call_deferred();
		callable_mp(project_name, &LineEdit::select_all).call_deferred();
	} else {
		if (p_reset_name) {
			_reset_name();
		}
		project_path->set_editable(true);

		if (mode == MODE_DUPLICATE) {
			String original_dir = original_project_path.get_base_dir();
			project_path->set_text(original_dir);
			install_path->set_text(original_dir);
			fdialog_project->set_current_dir(original_dir);
		} else {
			String fav_dir = EDITOR_GET("filesystem/directories/default_project_path");
			fav_dir = fav_dir.simplify_path();
			if (!fav_dir.is_empty()) {
				project_path->set_text(fav_dir);
				install_path->set_text(fav_dir);
				fdialog_project->set_current_dir(fav_dir);
			} else {
				Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
				project_path->set_text(d->get_current_dir());
				install_path->set_text(d->get_current_dir());
				fdialog_project->set_current_dir(d->get_current_dir());
			}
		}

		create_dir->show();
		project_status_rect->show();
		project_browse->show();
		edit_check_box->show();

		if (mode == MODE_IMPORT) {
			set_title(TTRC("Import Existing Project"));
			set_ok_button_text(TTRC("Import"));

			name_container->hide();
			install_path_container->hide();
			renderer_container->hide();
			default_files_container->hide();

			// Project path dialog is also opened; no need to change focus.
		} else if (mode == MODE_NEW) {
			set_title(TTRC("Create New Project"));
			set_ok_button_text(TTRC("Create"));

			name_container->show();
			install_path_container->hide();
			renderer_container->show();
			default_files_container->show();

			callable_mp((Control *)project_name, &Control::grab_focus).call_deferred();
			callable_mp(project_name, &LineEdit::select_all).call_deferred();
		} else if (mode == MODE_INSTALL) {
			set_title(TTR("Install Project:") + " " + zip_title);
			set_ok_button_text(TTRC("Install"));

			project_name->set_text(zip_title);

			name_container->show();
			install_path_container->hide();
			renderer_container->hide();
			default_files_container->hide();

			callable_mp((Control *)project_path, &Control::grab_focus).call_deferred();
		} else if (mode == MODE_DUPLICATE) {
			set_title(TTRC("Duplicate Project"));
			set_ok_button_text(TTRC("Duplicate"));

			name_container->show();
			install_path_container->hide();
			renderer_container->hide();
			default_files_container->hide();
			if (!duplicate_can_edit) {
				edit_check_box->hide();
			}

			callable_mp((Control *)project_name, &Control::grab_focus).call_deferred();
			callable_mp(project_name, &LineEdit::select_all).call_deferred();
		}

		auto_dir = "";
		last_custom_target_dir = "";
		_update_target_auto_dir();
		if (create_dir->is_pressed()) {
			// Append `auto_dir` to target path.
			_create_dir_toggled(true);
		}
	}

	_validate_path();

	popup_centered(Size2(500, 0) * EDSCALE);
}

void ProjectDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
			_renderer_selected();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			create_dir->set_button_icon(get_editor_theme_icon(SNAME("FolderCreate")));
			project_browse->set_button_icon(get_editor_theme_icon(SNAME("FolderBrowse")));
			install_browse->set_button_icon(get_editor_theme_icon(SNAME("FolderBrowse")));
		} break;
		case NOTIFICATION_READY: {
			fdialog_project = memnew(EditorFileDialog);
			fdialog_project->set_previews_enabled(false); // Crucial, otherwise the engine crashes.
			fdialog_project->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
			fdialog_project->connect("dir_selected", callable_mp(this, &ProjectDialog::_project_path_selected));
			fdialog_project->connect("file_selected", callable_mp(this, &ProjectDialog::_project_path_selected));
			fdialog_project->connect("canceled", callable_mp(this, &ProjectDialog::show_dialog).bind(false), CONNECT_DEFERRED);
			callable_mp((Node *)this, &Node::add_sibling).call_deferred(fdialog_project, false);
		} break;
	}
}

void ProjectDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("project_created"));
	ADD_SIGNAL(MethodInfo("project_duplicated"));
	ADD_SIGNAL(MethodInfo("projects_updated"));
}

ProjectDialog::ProjectDialog() {
	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	name_container = memnew(VBoxContainer);
	vb->add_child(name_container);

	Label *l = memnew(Label);
	l->set_text(TTRC("Project Name:"));
	name_container->add_child(l);

	project_name = memnew(LineEdit);
	project_name->set_virtual_keyboard_show_on_focus(false);
	project_name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	name_container->add_child(project_name);

	project_path_container = memnew(VBoxContainer);
	vb->add_child(project_path_container);

	HBoxContainer *pphb_label = memnew(HBoxContainer);
	project_path_container->add_child(pphb_label);

	l = memnew(Label);
	l->set_text(TTRC("Project Path:"));
	l->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	pphb_label->add_child(l);

	create_dir = memnew(CheckButton);
	create_dir->set_text(TTRC("Create Folder"));
	create_dir->set_pressed(true);
	pphb_label->add_child(create_dir);
	create_dir->connect(SceneStringName(toggled), callable_mp(this, &ProjectDialog::_create_dir_toggled));

	HBoxContainer *pphb = memnew(HBoxContainer);
	project_path_container->add_child(pphb);

	project_path = memnew(LineEdit);
	project_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	project_path->set_accessibility_name(TTRC("Project Path:"));
	project_path->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_FILE);
	pphb->add_child(project_path);

	install_path_container = memnew(VBoxContainer);
	vb->add_child(install_path_container);

	l = memnew(Label);
	l->set_text(TTRC("Project Installation Path:"));
	install_path_container->add_child(l);

	HBoxContainer *iphb = memnew(HBoxContainer);
	install_path_container->add_child(iphb);

	install_path = memnew(LineEdit);
	install_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	install_path->set_accessibility_name(TTRC("Project Installation Path:"));
	install_path->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_FILE);
	iphb->add_child(install_path);

	// status icon
	project_status_rect = memnew(TextureRect);
	project_status_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	pphb->add_child(project_status_rect);

	project_browse = memnew(Button);
	project_browse->set_text(TTRC("Browse"));
	project_browse->connect(SceneStringName(pressed), callable_mp(this, &ProjectDialog::_browse_project_path));
	pphb->add_child(project_browse);

	// install status icon
	install_status_rect = memnew(TextureRect);
	install_status_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	iphb->add_child(install_status_rect);

	install_browse = memnew(Button);
	install_browse->set_text(TTRC("Browse"));
	install_browse->connect(SceneStringName(pressed), callable_mp(this, &ProjectDialog::_browse_install_path));
	iphb->add_child(install_browse);

	msg = memnew(Label);
	msg->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	msg->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	msg->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	msg->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	vb->add_child(msg);
	
	// Progress label for template cloning
	progress_label = memnew(Label);
	progress_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	progress_label->set_custom_minimum_size(Size2(200, 30) * EDSCALE);
	progress_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	progress_label->add_theme_color_override("font_color", Color(0.5, 0.8, 1.0));
	progress_label->hide();
	vb->add_child(progress_label);

	// Renderer selection.
	renderer_container = memnew(VBoxContainer);
	vb->add_child(renderer_container);
	l = memnew(Label);
	l->set_text(TTRC("Renderer:"));
	renderer_container->add_child(l);
	HBoxContainer *rshc = memnew(HBoxContainer);
	renderer_container->add_child(rshc);
	renderer_button_group.instantiate();

	// Left hand side, used for checkboxes to select renderer.
	Container *rvb = memnew(VBoxContainer);
	rshc->add_child(rvb);

	String default_renderer_type = "forward_plus";
	if (EditorSettings::get_singleton()->has_setting("project_manager/default_renderer")) {
		default_renderer_type = EditorSettings::get_singleton()->get_setting("project_manager/default_renderer");
	}

	rendering_device_supported = DisplayServer::is_rendering_device_supported();
	
	// ORCA ENGINE FIX: On Windows, if RD_ENABLED is compiled in, always show Forward+ and Mobile options
	// The runtime test can fail in virtualization environments (Parallels, VMWare) even when drivers work
	// Users should be able to try Forward+ and Mobile - if they don't work, they can switch to Compatibility
	#ifdef WINDOWS_ENABLED
	#ifdef RD_ENABLED
	// If RenderingDevice is compiled in, allow Forward+ and Mobile even if runtime test failed
	// This is more user-friendly - let users try the modern renderers
	bool allow_rd_renderers = true;
	#else
	bool allow_rd_renderers = rendering_device_supported;
	#endif
	#else
	bool allow_rd_renderers = rendering_device_supported;
	#endif

	if (!rendering_device_supported && !allow_rd_renderers) {
		default_renderer_type = "gl_compatibility";
	}

	Button *rs_button = memnew(CheckBox);
	rs_button->set_button_group(renderer_button_group);
	rs_button->set_text(TTRC("Forward+"));
#ifndef RD_ENABLED
	rs_button->set_disabled(true);
#else
	// On Windows with RD_ENABLED, always enable Forward+ even if runtime test failed
	// User can try it and switch to Compatibility if it doesn't work
	#ifdef WINDOWS_ENABLED
	if (!allow_rd_renderers) {
		rs_button->set_disabled(true);
	} else {
		rs_button->set_disabled(false); // Explicitly enable on Windows with RD_ENABLED
	}
	#else
	if (!rendering_device_supported) {
		rs_button->set_disabled(true);
	} else {
		rs_button->set_disabled(false);
	}
	#endif
#endif
	rs_button->set_meta(SNAME("rendering_method"), "forward_plus");
	rs_button->connect(SceneStringName(pressed), callable_mp(this, &ProjectDialog::_renderer_selected));
	rvb->add_child(rs_button);
	if (default_renderer_type == "forward_plus") {
		rs_button->set_pressed(true);
	}
	rs_button = memnew(CheckBox);
	rs_button->set_button_group(renderer_button_group);
	rs_button->set_text(TTRC("Mobile"));
#ifndef RD_ENABLED
	rs_button->set_disabled(true);
#else
	// On Windows with RD_ENABLED, always enable Mobile even if runtime test failed
	#ifdef WINDOWS_ENABLED
	if (!allow_rd_renderers) {
		rs_button->set_disabled(true);
	} else {
		rs_button->set_disabled(false); // Explicitly enable on Windows with RD_ENABLED
	}
	#else
	if (!rendering_device_supported) {
		rs_button->set_disabled(true);
	} else {
		rs_button->set_disabled(false);
	}
	#endif
#endif
	rs_button->set_meta(SNAME("rendering_method"), "mobile");
	rs_button->connect(SceneStringName(pressed), callable_mp(this, &ProjectDialog::_renderer_selected));
	rvb->add_child(rs_button);
	if (default_renderer_type == "mobile") {
		rs_button->set_pressed(true);
	}
	rs_button = memnew(CheckBox);
	rs_button->set_button_group(renderer_button_group);
	rs_button->set_text(TTRC("Compatibility"));
#if !defined(GLES3_ENABLED)
	rs_button->set_disabled(true);
#endif
	rs_button->set_meta(SNAME("rendering_method"), "gl_compatibility");
	rs_button->connect(SceneStringName(pressed), callable_mp(this, &ProjectDialog::_renderer_selected));
	rvb->add_child(rs_button);
#if defined(GLES3_ENABLED)
	if (default_renderer_type == "gl_compatibility") {
		rs_button->set_pressed(true);
	}
#endif
	rshc->add_child(memnew(VSeparator));

	// Right hand side, used for text explaining each choice.
	rvb = memnew(VBoxContainer);
	rvb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	rshc->add_child(rvb);
	renderer_info = memnew(Label);
	renderer_info->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	renderer_info->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	renderer_info->set_modulate(Color(1, 1, 1, 0.7));
	rvb->add_child(renderer_info);

	rd_not_supported = memnew(Label);
	rd_not_supported->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	rd_not_supported->set_text(vformat(TTRC("RenderingDevice-based methods not available on this GPU:\n%s\nPlease use the Compatibility renderer."), RenderingServer::get_singleton()->get_video_adapter_name()));
	rd_not_supported->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	rd_not_supported->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	rd_not_supported->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	rd_not_supported->set_visible(false);
	renderer_container->add_child(rd_not_supported);

	_renderer_selected();

	l = memnew(Label);
	l->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	l->set_text(TTRC("The renderer can be changed later, but scenes may need to be adjusted."));
	// Add some extra spacing to separate it from the list above and the buttons below.
	l->set_custom_minimum_size(Size2(0, 40) * EDSCALE);
	l->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	l->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	l->set_modulate(Color(1, 1, 1, 0.7));
	renderer_container->add_child(l);

	default_files_container = memnew(HBoxContainer);
	vb->add_child(default_files_container);
	l = memnew(Label);
	l->set_text(TTRC("Version Control Metadata:"));
	default_files_container->add_child(l);
	vcs_metadata_selection = memnew(OptionButton);
	vcs_metadata_selection->set_custom_minimum_size(Size2(100, 20));
	vcs_metadata_selection->add_item(TTRC("None"), (int)EditorVCSInterface::VCSMetadata::NONE);
	vcs_metadata_selection->add_item(TTRC("Git"), (int)EditorVCSInterface::VCSMetadata::GIT);
	vcs_metadata_selection->select((int)EditorVCSInterface::VCSMetadata::GIT);
	vcs_metadata_selection->set_accessibility_name(TTRC("Version Control Metadata:"));
	default_files_container->add_child(vcs_metadata_selection);
	Control *spacer = memnew(Control);
	spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	default_files_container->add_child(spacer);
	fdialog_install = memnew(EditorFileDialog);
	fdialog_install->set_previews_enabled(false); //Crucial, otherwise the engine crashes.
	fdialog_install->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	add_child(fdialog_install);

	Control *spacer2 = memnew(Control);
	spacer2->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vb->add_child(spacer2);

	edit_check_box = memnew(CheckBox);
	edit_check_box->set_text(TTRC("Edit Now"));
	edit_check_box->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	edit_check_box->set_pressed(true);
	vb->add_child(edit_check_box);

	project_name->connect(SceneStringName(text_changed), callable_mp(this, &ProjectDialog::_project_name_changed).unbind(1));
	project_name->connect(SceneStringName(text_submitted), callable_mp(this, &ProjectDialog::ok_pressed).unbind(1));

	project_path->connect(SceneStringName(text_changed), callable_mp(this, &ProjectDialog::_project_path_changed).unbind(1));
	project_path->connect(SceneStringName(text_submitted), callable_mp(this, &ProjectDialog::ok_pressed).unbind(1));

	install_path->connect(SceneStringName(text_changed), callable_mp(this, &ProjectDialog::_install_path_changed).unbind(1));
	install_path->connect(SceneStringName(text_submitted), callable_mp(this, &ProjectDialog::ok_pressed).unbind(1));

	fdialog_install->connect("dir_selected", callable_mp(this, &ProjectDialog::_install_path_selected));
	fdialog_install->connect("file_selected", callable_mp(this, &ProjectDialog::_install_path_selected));

	set_hide_on_ok(false);

	dialog_error = memnew(AcceptDialog);
	add_child(dialog_error);
}
