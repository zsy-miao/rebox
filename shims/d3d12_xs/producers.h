#pragma once

#include <windows.h>

/* Indirect-argument producers (producers.c): the compute dispatches that write the argument records
 * mesh ExecuteIndirects read (the title's GPU culling and compaction). */

struct producer;

/* proxy.c, at a mesh ExecuteIndirect: where its records ([args, args + bytes)) and count (0: none)
 * are read. *groups is set to REBOX_INDIRECT_FALLBACK's group count when the ExecuteIndirect is to
 * be replaced by a mesh dispatch of that many groups (substitute: the signature allows it), else 0. */
void producers_mesh_indirect(UINT64 args, UINT64 bytes, UINT64 count, BOOL substitute, UINT *groups);
/* Whether any argument ranges are known: before the first mesh ExecuteIndirect, nothing to match. */
BOOL producers_watching(void);
/* The argument ranges [base, base + bytes) touches, a bit each (0: none). */
UINT64 producers_overlap(UINT64 base, UINT64 bytes);

/* A compute program's record by its key (console_compile.c's), or NULL if it isn't a producer. */
struct producer *producers_find(UINT64 key);
/* Makes program key (id: the id the logs name it by) a producer of the ranges in mask, found by
 * how ("its bindings", "its user data"); state is its compile state then. Logged once. */
struct producer *producers_add(UINT64 key, UINT64 id, UINT64 mask, const char *how, const char *state);
/* One call of a producer: it ran, or was skipped (why: the outcome's name); quarantined: it is in
 * REBOX_QUARANTINE's file and ran anyway. error: AnyPS5's message when it failed, or NULL. */
void producers_call(struct producer *p, BOOL ran, const char *why, BOOL quarantined, const char *error);
UINT64 producers_id(const struct producer *p);
/* Whether a non-producer's user data is due a scan this frame (once a frame a program). */
BOOL producers_scan_due(UINT64 key);
/* A present (console_frame_presented). */
void producers_frame(LONG frame);
/* With the census (console_compile.c's outcomes_log). */
void producers_census_log(void);
