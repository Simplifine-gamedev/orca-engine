/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "core/variant/dictionary.h"

class AIChatDock;

class AIChatDockAsset {
public:
	// Asset Library callback implementations
	static void on_asset_install_requested(AIChatDock *p_dock, const String &p_asset_id, const String &p_asset_name);
	static void on_asset_browse_requested(const String &p_url);
	static void on_asset_folder_open_requested(AIChatDock *p_dock, const String &p_path);
	static void on_asset_plugin_settings_requested(AIChatDock *p_dock);
	
	// Cloud mode asset installation callbacks
	static void on_cloud_asset_install_requested(AIChatDock *p_dock, const String &p_asset_data, const String &p_install_path, const String &p_asset_name);
	static void on_cloud_asset_manual_download(AIChatDock *p_dock, const String &p_asset_data, const String &p_asset_name);
	static void on_manual_asset_save_location_selected(AIChatDock *p_dock, const String &p_file_path);
};


