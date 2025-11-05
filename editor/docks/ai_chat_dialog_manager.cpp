/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_chat_dialog_manager.h"
#include "ai_chat_dock.h"

void AIChatDialogManager::setup_all_dialogs(AIChatDock *p_chat_dock) {
	if (!p_chat_dock) {
		return;
	}

	// Create image warning dialog
	AcceptDialog *image_warning = create_image_warning_dialog(p_chat_dock);
	p_chat_dock->add_child(image_warning);
	p_chat_dock->image_warning_dialog = image_warning;

	// Create checkpoint restore dialog
	ConfirmationDialog *restore_dialog = create_restore_checkpoint_dialog(p_chat_dock);
	p_chat_dock->add_child(restore_dialog);
	p_chat_dock->restore_checkpoint_dialog = restore_dialog;

	// Create add models dialog
	AcceptDialog *add_models = create_add_models_dialog(p_chat_dock);
	p_chat_dock->add_child(add_models);
	p_chat_dock->add_models_dialog = add_models;
}

AcceptDialog *AIChatDialogManager::create_image_warning_dialog(AIChatDock *p_chat_dock) {
	AcceptDialog *dialog = memnew(AcceptDialog);
	dialog->set_title("Image Processing Warning");
	_setup_image_warning_content(dialog, p_chat_dock);
	return dialog;
}

ConfirmationDialog *AIChatDialogManager::create_restore_checkpoint_dialog(AIChatDock *p_chat_dock) {
	ConfirmationDialog *dialog = memnew(ConfirmationDialog);
	dialog->set_title("Restore Checkpoint");
	_setup_restore_checkpoint_content(dialog, p_chat_dock);
	return dialog;
}

AcceptDialog *AIChatDialogManager::create_add_models_dialog(AIChatDock *p_chat_dock) {
	AcceptDialog *dialog = memnew(AcceptDialog);
	dialog->set_title("Add AI Models");
	dialog->set_size(Size2(600, 400));
	
	Tree *models_tree = _create_models_tree(p_chat_dock);
	dialog->add_child(models_tree);
	p_chat_dock->add_models_tree = models_tree;
	
	return dialog;
}

void AIChatDialogManager::_setup_image_warning_content(AcceptDialog *p_dialog, AIChatDock *p_chat_dock) {
	if (!p_dialog) {
		return;
	}
	
	p_dialog->set_text("Large images will be automatically downsampled to optimize API usage.\nOriginal images are preserved.");
}

void AIChatDialogManager::_setup_restore_checkpoint_content(ConfirmationDialog *p_dialog, AIChatDock *p_chat_dock) {
	if (!p_dialog) {
		return;
	}
	
	p_dialog->set_text("This will restore your project to the state at the selected message.\nAny changes made after that point will be lost.\n\nAre you sure you want to continue?");
	p_dialog->get_ok_button()->set_text("Restore");
	p_dialog->get_cancel_button()->set_text("Cancel");
}

Tree *AIChatDialogManager::_create_models_tree(AIChatDock *p_chat_dock) {
	Tree *tree = memnew(Tree);
	tree->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tree->set_columns(2);
	tree->set_column_title(0, "Model");
	tree->set_column_title(1, "Provider");
	tree->set_column_titles_visible(true);
	return tree;
}
