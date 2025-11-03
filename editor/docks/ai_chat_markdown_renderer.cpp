/**************************************************************************/
/*  ai_chat_markdown_renderer.cpp                                         */
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

#include "ai_chat_markdown_renderer.h"

String AIChatMarkdownRenderer::process_inline_markdown(String p_line) {
	String line = p_line;

	// Bold (**text** or __text__)
	while (true) {
		int start = line.find("**");
		if (start == -1) {
			start = line.find("__");
			if (start == -1) {
				break;
			}
		}

		String marker = line.substr(start, 2);
		int end = line.find(marker, start + 2);
		if (end == -1) {
			break;
		}

		String before = line.substr(0, start);
		String bold_text = line.substr(start + 2, end - start - 2);
		String after = line.substr(end + 2);

		line = before + "[b]" + bold_text + "[/b]" + after;
	}

	// Italic (*text* or _text_) - but not inside ** or __
	int pos = 0;
	while (pos < line.length()) {
		int star_pos = line.find("*", pos);
		int underscore_pos = line.find("_", pos);

		int start = -1;
		String marker;

		if (star_pos != -1 && (underscore_pos == -1 || star_pos < underscore_pos)) {
			// Check it's not part of **
			if ((star_pos > 0 && line[star_pos - 1] == '*') || (star_pos < line.length() - 1 && line[star_pos + 1] == '*')) {
				pos = star_pos + 1;
				continue;
			}
			start = star_pos;
			marker = "*";
		} else if (underscore_pos != -1) {
			// Check it's not part of __
			if ((underscore_pos > 0 && line[underscore_pos - 1] == '_') || (underscore_pos < line.length() - 1 && line[underscore_pos + 1] == '_')) {
				pos = underscore_pos + 1;
				continue;
			}
			start = underscore_pos;
			marker = "_";
		}

		if (start == -1) {
			break;
		}

		int end = line.find(marker, start + 1);
		if (end == -1) {
			pos = start + 1;
			continue;
		}

		String before = line.substr(0, start);
		String italic_text = line.substr(start + 1, end - start - 1);
		String after = line.substr(end + 1);

		line = before + "[i]" + italic_text + "[/i]" + after;
		pos = before.length() + 3 + italic_text.length() + 4; // Skip past [i]...[/i]
	}

	// Inline code (`text`)
	while (true) {
		int start = line.find("`");
		if (start == -1) {
			break;
		}

		int end = line.find("`", start + 1);
		if (end == -1) {
			break;
		}

		String before = line.substr(0, start);
		String code_text = line.substr(start + 1, end - start - 1);
		String after = line.substr(end + 1);

		line = before + "[code]" + code_text.xml_escape() + "[/code]" + after;
	}

	return line;
}

String AIChatMarkdownRenderer::markdown_to_bbcode(const String &p_markdown) {
	if (p_markdown.is_empty()) {
		return "";
	}

	Vector<String> lines = p_markdown.split("\n");
	String result = "";
	bool in_code_block = false;
	String code_block_lang = "";

	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];

		// Code blocks (```)
		if (line.strip_edges().begins_with("```")) {
			if (in_code_block) {
				result += "[/code]";
				in_code_block = false;
				code_block_lang = "";
			} else {
				code_block_lang = line.strip_edges().substr(3).strip_edges();
				result += "[code]";
			}
		} else if (in_code_block) {
			result += line.xml_escape(); // Escape to prevent BBCode parsing inside code blocks.
		} else {
			if (line.strip_edges().is_empty()) {
				// Preserve blank lines between paragraphs.
				result += "";
			} else {
				String processed_line = line;
				String trimmed_line = processed_line.lstrip(" \t");

				// Headers - ENHANCED sizing (bigger and more prominent)
				if (trimmed_line.begins_with("#")) {
					int header_level = 0;
					while (header_level < trimmed_line.length() && trimmed_line[header_level] == '#') {
						header_level++;
					}
					String header_content = trimmed_line.substr(header_level).strip_edges();
					if (!header_content.is_empty()) {
						// Larger font sizes: # = 48, ## = 42, ### = 36, #### = 30, etc.
						// Increased from previous sizes to make titles more prominent
						int font_size = 48 - (header_level * 6);
						if (font_size < 24) {
							font_size = 24; // Minimum size
						}
						processed_line = "[font_size=" + String::num_int64(font_size) + "][b]" + process_inline_markdown(header_content) + "[/b][/font_size]";
					}
					// Lists
				} else if (trimmed_line.begins_with("- ") || trimmed_line.begins_with("* ")) {
					String item_content = trimmed_line.substr(trimmed_line.find(" ") + 1);
					processed_line = "[indent]* " + process_inline_markdown(item_content) + "[/indent]";
				} else {
					// Regular paragraph
					processed_line = process_inline_markdown(processed_line);
				}
				result += processed_line;
			}
		}

		if (i < lines.size() - 1) {
			result += "\n";
		}
	}

	return result;
}


