#pragma once

#include <windows.h>

/* Program names and REBOX_FAST's detail sacrifices; see fast.c. */

/* The entry point name a console container carries in its PSV0 part (for example
 * "PS_DeferredTextured_c5fd36bb_Wrapped"), copied into name; FALSE if it has none. */
BOOL fast_program_name(const void *container, SIZE_T size, char *name, SIZE_T name_size);

/* Whether REBOX_FAST (or REBOX_SKIP_PROGRAMS) leaves this program out: the list entry's reason, or
 * NULL to keep it. kind names the stage for the log ("cs", "vs", "ps", "ms", "as"). */
const char *fast_skip_reason(const void *container, SIZE_T size, const char *kind);

/* Names a census or trace id (a graphics stage id, a compute program's code hash) for the logs. */
void fast_name_id(UINT64 id, const void *container, SIZE_T size);
/* The name fast_name_id gave this id, or NULL. */
const char *fast_id_name(UINT64 id);
