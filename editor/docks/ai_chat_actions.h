/**************************************************************************/
/*  ai_chat_actions.h                                                     */
/**************************************************************************/

#pragma once

#include "core/object/object.h"
class ClassDB;

class PopupMenu;

class AIChatActions : public Object {
    GDCLASS(AIChatActions, Object);

private:
    static AIChatActions *singleton;
    static bool registered;
    static int MENU_ID_NEW_CHAT;
    static int MENU_ID_OPEN_DOCK;

    void _on_docks_menu_id_pressed(int p_id);

protected:
    static void _bind_methods() {}

public:
    static AIChatActions *get_singleton();
    static void ensure_registered();
};


