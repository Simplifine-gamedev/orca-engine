/**************************************************************************/
/*  pricing_dialog.cpp                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             ORCA ENGINE                                */
/**************************************************************************/

#include "pricing_dialog.h"
#include "core/config/engine.h"
#include "core/io/json.h"
#include "core/os/os.h"
#include "scene/gui/separator.h"
#include "scene/gui/margin_container.h"
#include "editor/settings/editor_settings.h"

void PricingDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_upgrade_pressed"), &PricingDialog::_on_upgrade_pressed);
	ClassDB::bind_method(D_METHOD("_on_pricing_response"), &PricingDialog::_on_pricing_response);
	ClassDB::bind_method(D_METHOD("_on_checkout_response"), &PricingDialog::_on_checkout_response);
}

PricingDialog::PricingDialog() {
	set_title("Orca Engine Pricing");
	set_min_size(Size2(600, 500));
	
	_setup_ui();
	_load_pricing_tiers();
}

PricingDialog::~PricingDialog() {
}

void PricingDialog::_setup_ui() {
	// Main container
	main_container = memnew(VBoxContainer);
	add_child(main_container);
	
	// Header
	header_label = memnew(RichTextLabel);
	header_label->set_custom_minimum_size(Size2(0, 80));
	header_label->set_fit_content(true);
	header_label->set_use_bbcode(true);
	header_label->set_text("[center][font_size=18][b]Upgrade Your Orca Engine Plan[/b][/font_size][/center]\n[center]Choose the plan that fits your needs[/center]");
	main_container->add_child(header_label);
	
	// Separator
	HSeparator *separator = memnew(HSeparator);
	main_container->add_child(separator);
	
	// Tiers container
	tiers_container = memnew(VBoxContainer);
	tiers_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	main_container->add_child(tiers_container);
	
	// HTTP request for pricing data
	pricing_http_request = memnew(HTTPRequest);
	add_child(pricing_http_request);
	pricing_http_request->connect("request_completed", callable_mp(this, &PricingDialog::_on_pricing_response));
	
	// HTTP request for checkout
	checkout_http_request = memnew(HTTPRequest);
	add_child(checkout_http_request);
	checkout_http_request->connect("request_completed", callable_mp(this, &PricingDialog::_on_checkout_response));
}

void PricingDialog::_create_tier_card(const PricingTier &tier) {
	// Card panel
	PanelContainer *card = memnew(PanelContainer);
	card->set_custom_minimum_size(Size2(0, 120));
	tiers_container->add_child(card);
	
	// Card margin
	MarginContainer *margin = memnew(MarginContainer);
	margin->add_theme_constant_override("margin_left", 15);
	margin->add_theme_constant_override("margin_right", 15);
	margin->add_theme_constant_override("margin_top", 10);
	margin->add_theme_constant_override("margin_bottom", 10);
	card->add_child(margin);
	
	// Card content
	HBoxContainer *card_content = memnew(HBoxContainer);
	margin->add_child(card_content);
	
	// Left side - info
	VBoxContainer *info_container = memnew(VBoxContainer);
	info_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	card_content->add_child(info_container);
	
	// Title and price
	HBoxContainer *title_container = memnew(HBoxContainer);
	info_container->add_child(title_container);
	
	Label *name_label = memnew(Label);
	name_label->set_text(tier.name);
	name_label->add_theme_font_size_override("font_size", 18);
	name_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	title_container->add_child(name_label);
	
	Label *price_label = memnew(Label);
	if (tier.price == 0) {
		price_label->set_text("Free");
	} else {
		price_label->set_text("$" + String::num(tier.price) + "/month");
	}
	price_label->add_theme_font_size_override("font_size", 16);
	title_container->add_child(price_label);
	
	// Requests info
	Label *requests_label = memnew(Label);
	requests_label->set_text(String::num(tier.requests_per_month) + " AI requests per month");
	info_container->add_child(requests_label);
	
	// Features
	String features_text = "Features: ";
	for (int i = 0; i < tier.features.size(); i++) {
		features_text += tier.features[i];
		if (i < tier.features.size() - 1) {
			features_text += ", ";
		}
	}
	Label *features_label = memnew(Label);
	features_label->set_text(features_text);
	features_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	info_container->add_child(features_label);
	
	// Right side - button
	VBoxContainer *button_container = memnew(VBoxContainer);
	card_content->add_child(button_container);
	
	Button *upgrade_button = memnew(Button);
	if (tier.price == 0) {
		upgrade_button->set_text("Current Plan");
		upgrade_button->set_disabled(true);
	} else {
		upgrade_button->set_text("Upgrade");
		upgrade_button->connect("pressed", callable_mp(this, &PricingDialog::_on_upgrade_pressed).bind(tier.product_id));
	}
	upgrade_button->set_custom_minimum_size(Size2(100, 40));
	button_container->add_child(upgrade_button);
}

void PricingDialog::_load_pricing_tiers() {
	// Load pricing tiers from backend
	String backend_url = OS::get_singleton()->get_environment("BACKEND_URL");
	if (backend_url.is_empty()) {
		backend_url = "http://127.0.0.1:8080";
	}
	
	String url = backend_url + "/pricing/tiers";
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	pricing_http_request->request(url, headers, HTTPClient::METHOD_GET);
}

void PricingDialog::_on_upgrade_pressed(const String &product_id) {
	selected_product_id = product_id;
	
	// Create checkout request
	String backend_url = OS::get_singleton()->get_environment("BACKEND_URL");
	if (backend_url.is_empty()) {
		backend_url = "http://127.0.0.1:8080";
	}
	
	String url = backend_url + "/pricing/checkout";
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	Dictionary request_data;
	request_data["product_id"] = product_id;
	
	String json_string = JSON::stringify(request_data);
	checkout_http_request->request(url, headers, HTTPClient::METHOD_POST, json_string);
}

void PricingDialog::_on_pricing_response(int result, int response_code, const PackedStringArray &headers, const PackedByteArray &body) {
	if (response_code != 200) {
		print_line("Failed to load pricing tiers: " + String::num(response_code));
		return;
	}
	
	String response_text = String::utf8((const char *)body.ptr(), body.size());
	JSON json;
	Error parse_result = json.parse(response_text);
	
	if (parse_result != OK) {
		print_line("Failed to parse pricing response");
		return;
	}
	
	Dictionary response_data = json.get_data();
	if (!response_data.has("success") || !response_data.get("success", false)) {
		print_line("Pricing API returned error");
		return;
	}
	
	Dictionary tiers_data = response_data.get("tiers", Dictionary());
	
	// Clear existing tiers
	for (int i = tiers_container->get_child_count() - 1; i >= 0; i--) {
		Node *child = tiers_container->get_child(i);
		tiers_container->remove_child(child);
		child->queue_free();
	}
	pricing_tiers.clear();
	
	// Parse and create tier cards
	Array tier_keys = tiers_data.keys();
	for (int i = 0; i < tier_keys.size(); i++) {
		String tier_key = tier_keys[i];
		Dictionary tier_dict = tiers_data[tier_key];
		
		PricingTier tier;
		tier.product_id = tier_dict.get("product_id", "");
		tier.name = tier_dict.get("name", "");
		tier.price = tier_dict.get("price", 0);
		tier.requests_per_month = tier_dict.get("requests_per_month", 0);
		
		Array features_array = tier_dict.get("features", Array());
		for (int j = 0; j < features_array.size(); j++) {
			tier.features.push_back(features_array[j]);
		}
		
		pricing_tiers.push_back(tier);
		_create_tier_card(tier);
	}
}

void PricingDialog::_on_checkout_response(int result, int response_code, const PackedStringArray &headers, const PackedByteArray &body) {
	String response_text = String::utf8((const char *)body.ptr(), body.size());
	
	if (response_code != 200) {
		print_line("Checkout request failed: " + String::num(response_code) + " - " + response_text);
		return;
	}
	
	JSON json;
	Error parse_result = json.parse(response_text);
	
	if (parse_result != OK) {
		print_line("Failed to parse checkout response");
		return;
	}
	
	Dictionary response_data = json.get_data();
	if (!response_data.has("success") || !response_data.get("success", false)) {
		String error_msg = response_data.get("error", "Unknown error");
		print_line("Checkout API returned error: " + error_msg);
		return;
	}
	
	Dictionary checkout_data = response_data.get("checkout", Dictionary());
	
	if (checkout_data.has("checkout_url")) {
		// Open checkout URL in browser
		String checkout_url = checkout_data.get("checkout_url", "");
		OS::get_singleton()->shell_open(checkout_url);
		hide();
	} else {
		print_line("No checkout URL in response");
	}
}

void PricingDialog::show_dialog() {
	popup_centered();
	_load_pricing_tiers();
}

void PricingDialog::show_rate_limit_dialog(const Dictionary &rate_limit_info) {
	// Update header to show rate limit message
	String message = "[center][font_size=18][b]Request Limit Exceeded[/b][/font_size][/center]\n";
	message += "[center]You've reached your monthly request limit.[/center]\n";
	message += "[center]Upgrade your plan to continue using Orca Engine.[/center]";
	
	header_label->set_text(message);
	popup_centered();
	_load_pricing_tiers();
}

void PricingDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			// Initialization when ready
		} break;
	}
}
