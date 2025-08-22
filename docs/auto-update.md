# Auto-update: GitHub Releases + Appcast + Sparkle

This repo auto-generates an update feed (appcast) from GitHub Releases. Your macOS app can read that feed with Sparkle to show “Update available” and install the update.

## What these mean (very short)
- Appcast: An RSS XML file (`docs/appcast.xml`) that says “latest version is X, download it here”.
- Sparkle: The updater used by many macOS apps; it checks the appcast and handles download/install.
- GitHub Actions: CI that runs when you publish a Release and rewrites the appcast.
- YAML (`.yml`): Human-readable config format used by GitHub Actions to describe steps.

## Files in this repo
- `docs/appcast.xml`: The appcast feed, served by GitHub Pages.
- `.github/workflows/appcast.yml`: The workflow that updates `appcast.xml` whenever you publish a Release.

## One-time setup
1) Keys
- Create Ed25519 keys (keep the private key out of git). You’ll put the private key into a repo secret named `SPARKLE_ED25519_PRIV_PEM`. The public key (base64, no PEM headers) goes into your app’s `Info.plist` as `SUPublicEDKey`.

2) GitHub Pages
- Enable Pages: Settings → Pages → Source: Deploy from a branch → Branch: `main`, Folder: `/docs`.
- Your feed URL will be: `https://simplifine-gamedev.github.io/orca-engine/appcast.xml`.

3) App configuration
- Add to your built app’s `Info.plist`:
  - `SUFeedURL` = the feed URL above
  - `SUPublicEDKey` = your base64-encoded public key (no header/footer lines)

## Shipping a new version (every release)
1) Build your `.app`.
2) Zip it correctly (preserves attrs):
```bash
ditto -c -k --sequesterRsrc --keepParent "/path/to/YourApp.app" "YourApp_1.2.3.zip"
```
3) Create a GitHub Release with tag `v1.2.3` and upload that zip (or a dmg) as an asset.
4) The workflow updates `docs/appcast.xml` to point to the uploaded asset (and signs it if the secret exists). Pages serves the updated feed.

## What users see in the app
- On launch (or periodically), Sparkle checks the feed.
- If a newer version exists, a small dialog/banner appears: “Update available”.
- Clicking Install downloads the update, verifies the signature, replaces the app, and relaunches into the new version.

## Troubleshooting (quick)
- No prompt: make sure `SUFeedURL` and `SUPublicEDKey` are set in `Info.plist`, and that your tag (e.g. `v1.2.3`) matches your app’s version.
- Feed 404: confirm GitHub Pages is enabled for `main` → `/docs`.
- Signature error: ensure the repo secret `SPARKLE_ED25519_PRIV_PEM` is set and publish a new Release.

## Windows updates (WinSparkle)
- WinSparkle is the Windows equivalent of Sparkle and supports the same appcast format.
- Use a separate feed for Windows: `https://simplifine-gamedev.github.io/orca-engine/appcast-windows.xml`.

One-time (Windows):
- Integrate WinSparkle in your app and set the appcast URL:
  - C/C++ (example):
    ```c
    #include <winsparkle.h>
    int main() {
        win_sparkle_set_appcast_url("https://simplifine-gamedev.github.io/orca-engine/appcast-windows.xml");
        win_sparkle_init();
        // your app loop...
        win_sparkle_cleanup();
        return 0;
    }
    ```
  - To check at startup: call `win_sparkle_check_update_with_ui();` after init if desired.

Per release (Windows):
1) Build your installer (`.exe` or `.msi`).
2) Create a GitHub Release (tag `vX.Y.Z`) and upload the installer as an asset.
3) The workflow writes `docs/appcast-windows.xml` pointing to that asset (and adds a signature if configured).
4) WinSparkle shows “Update available” and will download/run the installer.
