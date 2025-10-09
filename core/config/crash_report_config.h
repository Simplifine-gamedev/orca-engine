/**************************************************************************/
/*  crash_report_config.h                                                 */
/* © 2025 Simplifine Corp. This file is an original contribution to Orca Engine (based on Godot Engine). Licensed for free personal/non-commercial use under the Company Non‑Commercial License. See LICENSES/COMPANY-NONCOMMERCIAL.md. Commercial use requires a separate license from Simplifine. */
/**************************************************************************/

#ifndef CRASH_REPORT_CONFIG_H
#define CRASH_REPORT_CONFIG_H

// CENTRALIZED CRASH REPORTING CONFIGURATION
// This URL is used by ALL crash handlers across all platforms
// to send crash reports to the backend for investigation

// Production crash report endpoint
#define CRASH_REPORT_URL "https://api.orcaengine.ai/crash_report"

// Development/local crash report endpoint (used when DEV_MODE=true)
#define CRASH_REPORT_URL_DEV "http://localhost:5050/crash_report"

#endif // CRASH_REPORT_CONFIG_H

