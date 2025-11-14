/**************************************************************************/
/*  pricing_dialog.h                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             ORCA ENGINE                                */
/**************************************************************************/

#ifndef PRICING_DIALOG_H
#define PRICING_DIALOG_H

#include "scene/gui/dialogs.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/button.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/rich_text_label.h"
#include "scene/main/http_request.h"
#include "core/variant/dictionary.h"

class PricingDialog : public AcceptDialog {
	GDCLASS(PricingDialog, AcceptDialog);

private:
	struct PricingTier {
		String product_id;
		String name;
		int price;
		int requests_per_month;
		Vector<String> features;
	};

	VBoxContainer *main_container = nullptr;
	VBoxContainer *tiers_container = nullptr;
	HTTPRequest *pricing_http_request = nullptr;
	HTTPRequest *checkout_http_request = nullptr;
	
	RichTextLabel *header_label = nullptr;
	Button *close_button = nullptr;
	
	Vector<PricingTier> pricing_tiers;
	String selected_product_id;
	
	void _setup_ui();
	void _create_tier_card(const PricingTier &tier);
	void _load_pricing_tiers();
	
	void _on_upgrade_pressed(const String &product_id);
	void _on_pricing_response(int result, int response_code, const PackedStringArray &headers, const PackedByteArray &body);
	void _on_checkout_response(int result, int response_code, const PackedStringArray &headers, const PackedByteArray &body);
	
protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	PricingDialog();
	~PricingDialog();
	
	void show_dialog();
	void show_rate_limit_dialog(const Dictionary &rate_limit_info);
};

#endif // PRICING_DIALOG_H
