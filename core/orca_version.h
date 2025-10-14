// Orca Engine version information
// Version is embedded at build time from GitHub releases

#pragma once

// Orca version string - embedded during build by generate_orca_version.py
extern const char *const ORCA_VERSION_STRING;
extern const char *const ORCA_VERSION_FULL;

// Helper function to get Orca version at runtime
const char *get_orca_version();

