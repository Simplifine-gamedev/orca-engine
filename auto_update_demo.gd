# Auto-Update System Demo for Orca Engine
# This script demonstrates how to use the auto-update system

extends Node

func _ready():
	print("=== Orca Engine Auto-Update System Demo ===")
	
	# Get the AutoUpdateManager singleton
	var update_manager = AutoUpdateManager
	if not update_manager:
		print("ERROR: AutoUpdateManager not available!")
		return
	
	# Get the AutoUpdateService
	var update_service = AutoUpdateService.get_singleton()
	if not update_service:
		print("ERROR: AutoUpdateService not available!")
		return
	
	print("Current version: ", update_manager.get_current_version())
	print("Backend URL: ", update_manager.get_backend_url())
	
	# Connect to update signals
	update_service.connect("update_check_completed", _on_update_check_completed)
	update_service.connect("update_notification_shown", _on_update_notification_shown)
	update_service.connect("update_install_started", _on_update_install_started)
	
	# Check for updates manually
	print("Checking for updates...")
	update_service.check_for_updates_now()

func _on_update_check_completed(update_info: Dictionary):
	print("Update check completed:")
	print("  Update available: ", update_info.get("update_available", false))
	print("  Latest version: ", update_info.get("latest_version", "unknown"))
	print("  Current version: ", update_info.get("current_version", "unknown"))
	
	if update_info.has("error"):
		print("  Error: ", update_info["error"])
	
	if update_info.get("update_available", false):
		print("  Release notes: ", update_info.get("release_notes", "No release notes"))
		print("  Download URL: ", update_info.get("download_url", "No download URL"))

func _on_update_notification_shown(version: String):
	print("Update notification shown for version: ", version)

func _on_update_install_started():
	print("User requested update installation!")

# Manual test functions you can call from the console
func test_check_updates():
	var update_service = AutoUpdateService.get_singleton()
	if update_service:
		update_service.check_for_updates_now()

func test_show_dialog():
	var update_service = AutoUpdateService.get_singleton()
	if update_service:
		update_service.show_update_dialog_if_available()

func test_backend_connection():
	var update_manager = AutoUpdateManager
	if update_manager:
		print("Testing backend connection...")
		update_manager.check_for_updates()