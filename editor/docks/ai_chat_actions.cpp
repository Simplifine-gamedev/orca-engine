/**************************************************************************/
/*  ai_chat_actions.cpp                                                   */
/**************************************************************************/

#include "ai_chat_actions.h"

#include "editor/editor_node.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/docks/ai_chat_dock.h"
#include "core/object/class_db.h"
#include "scene/gui/popup_menu.h"

AIChatActions *AIChatActions::singleton = nullptr;
bool AIChatActions::registered = false;
int AIChatActions::MENU_ID_NEW_CHAT = 0;
int AIChatActions::MENU_ID_OPEN_DOCK = 0;

AIChatActions *AIChatActions::get_singleton() {
    if (!singleton) {
        singleton = memnew(AIChatActions);
    }
    return singleton;
}

void AIChatActions::ensure_registered() {
    if (registered) {
        return;
    }
    registered = true;

    // Hook into the docks menu to add a "New AI Chat" action
    EditorDockManager *dm = EditorDockManager::get_singleton();
    if (!dm) {
        return;
    }
    PopupMenu *docks_menu = dm->get_docks_menu();
    if (!docks_menu) {
        return;
    }

    MENU_ID_OPEN_DOCK = docks_menu->get_item_count();
    docks_menu->add_separator();
    docks_menu->add_item(TTRC("Open AI Chat"));
    MENU_ID_NEW_CHAT = docks_menu->get_item_count();
    docks_menu->add_item(TTRC("New AI Chat Conversation"));
    docks_menu->connect(SceneStringName(id_pressed), callable_mp(get_singleton(), &AIChatActions::_on_docks_menu_id_pressed));
}

void AIChatActions::_on_docks_menu_id_pressed(int p_id) {
    EditorDockManager *dm = EditorDockManager::get_singleton();
    if (!dm) {
        return;
    }

    PopupMenu *docks_menu = dm->get_docks_menu();
    if (!docks_menu) {
        return;
    }

    if (p_id == MENU_ID_OPEN_DOCK) {
        // Focus AI Chat dock if present
        if (AIChatDock::get_singleton()) {
            EditorDockManager::get_singleton()->focus_dock(AIChatDock::get_singleton());
        }
        return;
    }

    if (p_id == MENU_ID_NEW_CHAT) {
        // Create a new conversation in the existing AI Chat dock; ensure the dock is visible
        if (AIChatDock::get_singleton()) {
            EditorDockManager::get_singleton()->open_dock(AIChatDock::get_singleton(), true);
            // Use the public menu path to create a new conversation
            AIChatDock::get_singleton()->send_error_message(""); // no-op to ensure symbol usage
            // Call through the same signal AIChatDock connects: simulate pressing new button
            AIChatDock::get_singleton()->call_deferred("_on_new_conversation_pressed");
        }
        return;
    }
}


