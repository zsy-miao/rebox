#pragma once

#include <windows.h>

#include "console_compile.h"

/* Console programs quarantined across runs (quarantine.c): REBOX_QUARANTINE=<file>. */

/* At a dispatch's or draw stage's compile (ReboxConsoleCompile): TRUE to skip it, its program
 * being quarantined. *program is the program's record, for quarantine_ran; NULL when off. */
BOOL quarantine_skips(const struct rebox_compile_in *in, void **program);
/* After the compile: it got a module, so it may run. */
void quarantine_ran(void *program);
/* The device is lost: a few programs that ran just before join the file. Once. */
void quarantine_device_lost(void);
