/**************************************************************************/
/*  ai_chat_model_manager.cpp                                             */
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
/* included in all copies or substantial portions of the Software.         */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "ai_chat_model_manager.h"
#include "scene/gui/option_button.h"
#include "core/templates/hash_set.h"

Array AIChatModelManager::get_default_model_names() {
	Array default_models;
	default_models.push_back("claude-4");
	default_models.push_back("claude-4 (thinking)");
	default_models.push_back("gemini-2.5");
	default_models.push_back("gemini-2.5 (thinking)");
	default_models.push_back("gpt-5");
	default_models.push_back("gpt-5 (thinking)");
	return default_models;
}

Array AIChatModelManager::filter_default_models(const Array &p_all_models) {
	Array default_model_names = get_default_model_names();
	HashSet<String> default_set;
	for (int i = 0; i < default_model_names.size(); i++) {
		default_set.insert(default_model_names[i]);
	}
	
	Array filtered_models;
	for (int i = 0; i < p_all_models.size(); i++) {
		Dictionary model_info = p_all_models[i];
		String model_name = model_info.get("name", "");
		if (default_set.has(model_name)) {
			filtered_models.push_back(model_info);
		}
	}
	
	return filtered_models;
}

int AIChatModelManager::populate_dropdown(OptionButton *p_dropdown, const Array &p_all_models, const Array &p_user_selected_models) {
	if (!p_dropdown) {
		return 0;
	}
	
	// Clear existing items
	p_dropdown->clear();
	
	// Define allowed models (default + user-selected)
	HashSet<String> allowed_models;
	Array default_model_names = get_default_model_names();
	for (int i = 0; i < default_model_names.size(); i++) {
		allowed_models.insert(default_model_names[i]);
	}
	
	// Add user-selected models
	for (int i = 0; i < p_user_selected_models.size(); i++) {
		String user_model = p_user_selected_models[i];
		allowed_models.insert(user_model);
	}
	
	// Organize models by type
	Array base_models;
	Array thinking_models;
	
	for (int i = 0; i < p_all_models.size(); i++) {
		Dictionary model_info = p_all_models[i];
		String model_name = model_info.get("name", "");
		bool is_thinking = model_info.get("is_thinking_variant", false);
		
		// Only include allowed models
		if (!allowed_models.has(model_name)) {
			continue;
		}
		
		if (is_thinking) {
			thinking_models.push_back(model_info);
		} else {
			base_models.push_back(model_info);
		}
	}
	
	int added_models = 0;
	
	// Add base models in priority order: claude-4, gpt-5, gemini-2.5
	for (int i = 0; i < base_models.size(); i++) {
		Dictionary model_info = base_models[i];
		String model_name = model_info.get("name", "");
		if (model_name == "claude-4") {
			p_dropdown->add_item(model_name);
			added_models++;
			break;
		}
	}
	for (int i = 0; i < base_models.size(); i++) {
		Dictionary model_info = base_models[i];
		String model_name = model_info.get("name", "");
		if (model_name == "gpt-5") {
			p_dropdown->add_item(model_name);
			added_models++;
			break;
		}
	}
	for (int i = 0; i < base_models.size(); i++) {
		Dictionary model_info = base_models[i];
		String model_name = model_info.get("name", "");
		if (model_name == "gemini-2.5") {
			p_dropdown->add_item(model_name);
			added_models++;
			break;
		}
	}
	// Add any other user-selected base models
	for (int i = 0; i < base_models.size(); i++) {
		Dictionary model_info = base_models[i];
		String model_name = model_info.get("name", "");
		if (model_name != "claude-4" && model_name != "gpt-5" && model_name != "gemini-2.5") {
			p_dropdown->add_item(model_name);
			added_models++;
		}
	}
	
	// Add thinking variants in priority order: claude-4 (thinking), gpt-5 (thinking), gemini-2.5 (thinking)
	for (int i = 0; i < thinking_models.size(); i++) {
		Dictionary model_info = thinking_models[i];
		String model_name = model_info.get("name", "");
		if (model_name == "claude-4 (thinking)") {
			p_dropdown->add_item(model_name);
			added_models++;
			break;
		}
	}
	for (int i = 0; i < thinking_models.size(); i++) {
		Dictionary model_info = thinking_models[i];
		String model_name = model_info.get("name", "");
		if (model_name == "gpt-5 (thinking)") {
			p_dropdown->add_item(model_name);
			added_models++;
			break;
		}
	}
	for (int i = 0; i < thinking_models.size(); i++) {
		Dictionary model_info = thinking_models[i];
		String model_name = model_info.get("name", "");
		if (model_name == "gemini-2.5 (thinking)") {
			p_dropdown->add_item(model_name);
			added_models++;
			break;
		}
	}
	// Add any other user-selected thinking models
	for (int i = 0; i < thinking_models.size(); i++) {
		Dictionary model_info = thinking_models[i];
		String model_name = model_info.get("name", "");
		if (model_name != "claude-4 (thinking)" && model_name != "gpt-5 (thinking)" && model_name != "gemini-2.5 (thinking)") {
			p_dropdown->add_item(model_name);
			added_models++;
		}
	}
	
	// Add separator and "Add Models..." option at the bottom
	p_dropdown->add_separator();
	p_dropdown->add_item("Add Models...");
	
	return added_models;
}


