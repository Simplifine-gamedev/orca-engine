#!/usr/bin/env python3
import argparse
import base64
import datetime as dt
import os
import sys
import xml.etree.ElementTree as ET


def parse_args():
    p = argparse.ArgumentParser(description="Update Sparkle/WinSparkle appcast feed")
    p.add_argument("--feed", required=True, help="Path to appcast XML file to update")
    p.add_argument("--version", required=True, help="Version string (e.g., v1.2.3)")
    p.add_argument("--url", required=True, help="Download URL for the asset")
    p.add_argument("--size", required=True, help="Asset size in bytes")
    p.add_argument("--os", choices=["mac", "windows"], required=True, help="Target OS")
    p.add_argument("--signature", default="", help="Sparkle Ed25519 signature (mac only)")
    return p.parse_args()


def ensure_namespaces(root):
    nsmap = {
        "sparkle": "http://www.andymatuschak.org/xml-namespaces/sparkle",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    for prefix, uri in nsmap.items():
        ET.register_namespace(prefix, uri)


def update_feed(feed_path: str, version: str, url: str, size: str, os_name: str, signature: str):
    if not os.path.exists(feed_path):
        raise SystemExit(f"Feed file not found: {feed_path}")

    tree = ET.parse(feed_path)
    root = tree.getroot()
    ensure_namespaces(root)

    channel = root.find("channel")
    if channel is None:
        # Namespaced variants
        for child in root:
            if child.tag.endswith("channel"):
                channel = child
                break
    if channel is None:
        raise SystemExit("Invalid appcast: missing <channel>")

    # Create an <item>
    item = ET.Element("item")
    title = ET.SubElement(item, "title")
    title.text = version

    rn = ET.SubElement(item, "{http://www.andymatuschak.org/xml-namespaces/sparkle}releaseNotesLink")
    rn.text = f"https://github.com/{os.environ.get('GITHUB_REPOSITORY','')}/releases/tag/{version}"

    pub = ET.SubElement(item, "pubDate")
    pub.text = dt.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")

    enc = ET.SubElement(item, "enclosure")
    enc.set("url", url)
    enc.set("length", str(size))
    enc.set("type", "application/octet-stream")
    enc.set("{http://www.andymatuschak.org/xml-namespaces/sparkle}version", version.lstrip("v"))

    if os_name == "mac" and signature:
        enc.set("{http://www.andymatuschak.org/xml-namespaces/sparkle}edSignature", signature.strip())

    # Prepend the latest item at the top of the channel
    # Insert after any <title>/<link>/<description>/<language>
    insert_index = 0
    for i, child in enumerate(list(channel)):
        if child.tag in ("title", "link", "description", "language") or child.tag.endswith(
            ("title", "link", "description", "language")
        ):
            insert_index = i + 1
        else:
            break
    channel.insert(insert_index, item)

    tree.write(feed_path, encoding="utf-8", xml_declaration=True)


def main():
    args = parse_args()
    update_feed(args.feed, args.version, args.url, args.size, args.os, args.signature)


if __name__ == "__main__":
    main()


