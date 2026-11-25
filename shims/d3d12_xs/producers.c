/* Indirect-argument producers.
 *
 * The world's meshes are mesh ExecuteIndirects whose group counts a compute pass of the title's
 * writes first (GPU culling and compaction). REBOX_DEBUG_VIEW_PROBE found nearly all of those
 * records 0 0 0 at the draw: the pass that fills them didn't run. Here the two are matched up:
 *
 * - Each mesh ExecuteIndirect's argument records (and count) are noted as ranges of GPU addresses
 *   (proxy.c), neighbours within RANGE_GAP merged, at most RANGES of them.
 * - Each console compute call is matched against them (console_compile.c): the guest buffers its
 *   bindings resolve to when it compiles, else the buffers its user data points at (root views, and
 *   V#s in the tables it points at; once a frame a program). A compute program that touches the
 *   argument ranges is a producer from then on, logged once with its compile state.
 * - A producer's calls are essential: their compiles go ahead of the rest and are waited for, and
 *   REBOX_QUARANTINE doesn't skip them (console_compile.c); REBOX_FAST keeps programs named as
 *   producers (fast.c). One that failed logs AnyPS5's error prominently.
 * - Whether a range's producer was skipped after it last ran (this frame or the one before) is
 *   kept, so a mesh ExecuteIndirect can tell its arguments weren't written.
 *
 * REBOX_INDIRECT_FALLBACK=<groups> (default 0, off): a mesh ExecuteIndirect whose producer was
 * skipped is replaced by a mesh dispatch of that many groups (at most 4096), so its mesh stage runs
 * with whatever per-group data is there, to get something on screen. Only for signatures whose
 * records are the group counts alone (no root arguments). REBOX_INDIRECT_FALLBACK_UNKNOWN=1 does so
 * too where no producer of its arguments was seen to run in this frame or the one before (a
 * producer that never compiled can't be matched by its bindings).
 *
 * Why it can't fault where today's path doesn't: patch 0017 already lets argument records of up to
 * 4M groups run whatever the buffer held (stale records the skipped pass never wrote), so a few
 * thousand groups is within what mesh stages already get. A console mesh stage's reads go through
 * guest buffer table entries cut to mapped GPU memory, unmapped addresses read as empty through
 * AnyPS5's page table, its descriptors are fixed at compile time (AnyPS5 refuses runtime-indexed
 * ones), and its vertex and primitive counts are clamped to its maximum (AnyPS5's mesh path). */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/rebox_log.h"
#include "fast.h"
#include "producers.h"

#define RANGES 64
#define RANGE_GAP 0x1000ull
#define PRODUCERS 256
#define PRODUCER_KEYS 1024
#define SCANS 16384 /* a power of two */
#define FALLBACK_MAX 4096u

struct range
{
    UINT64 lo, hi;
    volatile LONG64 skip_seq, run_seq; /* the last skipped and run producer calls (seq) */
    volatile LONG skip_frame, run_frame;
};

struct producer
{
    UINT64 id;
    volatile LONG64 mask;
    const char *how;
    volatile LONG calls, ran, skipped;              /* since the start */
    volatile LONG census_ran, census_skipped;       /* since the last census */
    const char *volatile last_skip;                 /* the last skip's reason (a static string) */
    volatile LONG error_logged, quarantine_logged;
};

static struct range ranges[RANGES];
static volatile LONG range_count;
static SRWLOCK ranges_lock = SRWLOCK_INIT;

static struct producer producers[PRODUCERS];
static volatile LONG producer_count;
static SRWLOCK producers_lock = SRWLOCK_INIT;

static volatile LONG64 scans[SCANS][2]; /* key, the frame it was last scanned + 1 */
static volatile LONG64 seq;
static volatile LONG frame = 1;

/* Mesh ExecuteIndirects since the last census: in all, with their producer skipped, with no
 * producer seen running, replaced by REBOX_INDIRECT_FALLBACK. */
static volatile LONG eis, eis_skipped, eis_unproduced, eis_substituted;

BOOL producers_watching(void)
{
    return range_count != 0;
}

/* Under ranges_lock (exclusive): the range holding or near [lo, hi), grown to hold it; a new one; or,
 * when full, the nearest grown to hold it. */
static int range_add(UINT64 lo, UINT64 hi)
{
    int n = range_count, best = -1;
    UINT64 best_gap = ~0ull;

    for (int i = 0; i < n; i++)
    {
        struct range *r = &ranges[i];
        UINT64 gap = lo > r->hi ? lo - r->hi : r->lo > hi ? r->lo - hi : 0;
        if (gap < best_gap)
            best_gap = gap, best = i;
    }
    if (best < 0 || (best_gap > RANGE_GAP && n < RANGES))
    {
        struct range *r = &ranges[n];
        r->lo = lo;
        r->hi = hi;
        r->skip_seq = r->run_seq = 0;
        r->skip_frame = r->run_frame = -100;
        InterlockedExchange(&range_count, n + 1);
        if (n + 1 == RANGES)
            rebox_log("d3d12_xs: indirect producers: %d argument ranges noted; later ones grow the nearest", RANGES);
        return n;
    }
    ranges[best].lo = min(ranges[best].lo, lo);
    ranges[best].hi = max(ranges[best].hi, hi);
    return best;
}

/* The range holding [lo, hi), added or grown if none does. */
static int range_of(UINT64 lo, UINT64 hi)
{
    int found = -1;

    AcquireSRWLockShared(&ranges_lock);
    for (int i = 0; i < range_count && found < 0; i++)
        if (ranges[i].lo <= lo && hi <= ranges[i].hi)
            found = i;
    ReleaseSRWLockShared(&ranges_lock);
    if (found >= 0)
        return found;
    AcquireSRWLockExclusive(&ranges_lock);
    found = range_add(lo, hi);
    ReleaseSRWLockExclusive(&ranges_lock);
    return found;
}

UINT64 producers_overlap(UINT64 base, UINT64 bytes)
{
    UINT64 mask = 0, end = base + bytes;

    if (!range_count || !base || !bytes || end < base)
        return 0;
    AcquireSRWLockShared(&ranges_lock);
    for (int i = 0; i < range_count; i++)
        if (base < ranges[i].hi && ranges[i].lo < end)
            mask |= 1ull << i;
    ReleaseSRWLockShared(&ranges_lock);
    return mask;
}

static LONG env_long(const char *name, LONG fallback)
{
    char v[16];
    DWORD n = GetEnvironmentVariableA(name, v, sizeof(v));
    return n && n < sizeof(v) ? atol(v) : fallback;
}

static UINT fallback_groups(BOOL *unknown_too)
{
    static LONG groups = -1, unknown = -1;
    if (groups < 0)
    {
        unknown = env_long("REBOX_INDIRECT_FALLBACK_UNKNOWN", 0) != 0;
        groups = min(max(env_long("REBOX_INDIRECT_FALLBACK", 0), 0L), (LONG)FALLBACK_MAX);
        if (groups)
            rebox_log("d3d12_xs: REBOX_INDIRECT_FALLBACK: a mesh ExecuteIndirect whose producer was skipped%s is "
                      "replaced by a mesh dispatch of %ld groups", unknown ? ", or none was seen running" : "", groups);
    }
    *unknown_too = unknown != 0;
    return (UINT)groups;
}

/* The state of range i for a draw now: its producer skipped after it last ran (this frame or the
 * one before), or none seen running in that time. */
static void range_state(int i, BOOL *skipped, BOOL *unproduced)
{
    const struct range *r = &ranges[i];
    LONG now = frame;
    *skipped = r->skip_seq > r->run_seq && r->skip_frame >= now - 1;
    *unproduced = r->run_frame < now - 1;
}

void producers_mesh_indirect(UINT64 args, UINT64 bytes, UINT64 count, BOOL substitute, UINT *groups)
{
    BOOL skipped = FALSE, unproduced = FALSE, unknown_too;
    UINT fallback = fallback_groups(&unknown_too);
    int a, c = -1;

    *groups = 0;
    if (!args || !bytes)
        return;
    a = range_of(args, args + bytes);
    if (count)
        c = range_of(count, count + 4);
    InterlockedIncrement(&eis);
    range_state(a, &skipped, &unproduced);
    if (c >= 0)
    {
        BOOL s, u;
        range_state(c, &s, &u);
        skipped |= s;
        unproduced &= u;
    }
    if (skipped)
        InterlockedIncrement(&eis_skipped);
    else if (unproduced)
        InterlockedIncrement(&eis_unproduced);
    if (fallback && substitute && (skipped || (unknown_too && unproduced)))
    {
        static LONG logged;
        LONG n = InterlockedIncrement(&eis_substituted);
        *groups = fallback;
        if (InterlockedIncrement(&logged) <= 4)
            rebox_log("d3d12_xs: REBOX_INDIRECT_FALLBACK: mesh ExecuteIndirect of arguments at %#llx (%s) replaced by "
                      "%u groups (%ld so far this census)", (unsigned long long)args,
                      skipped ? "its producer was skipped" : "no producer seen running", fallback, n);
    }
}

/* Keys (each a deferred blob's: one program comes in several PSOs) and the producer each is. */
static struct { volatile UINT64 key; struct producer *producer; } keys[PRODUCER_KEYS];
static volatile LONG key_count;

struct producer *producers_find(UINT64 key)
{
    LONG n = key_count;
    for (LONG i = 0; i < n; i++)
        if (keys[i].key == key)
            return keys[i].producer;
    return NULL;
}

UINT64 producers_id(const struct producer *p)
{
    return p ? p->id : 0;
}

/* The argument ranges of a mask, as text. */
static void mask_text(UINT64 mask, char *out, size_t size)
{
    int at = 0, shown = 0;
    out[0] = 0;
    AcquireSRWLockShared(&ranges_lock);
    for (int i = 0; i < range_count && at < (int)size - 48; i++)
        if (mask & (1ull << i))
        {
            if (shown++ == 4)
            {
                at += snprintf(out + at, size - at, " ...");
                break;
            }
            at += snprintf(out + at, size - at, "%s%#llx-%#llx", at ? ", " : "", (unsigned long long)ranges[i].lo,
                           (unsigned long long)ranges[i].hi);
        }
    ReleaseSRWLockShared(&ranges_lock);
}

struct producer *producers_add(UINT64 key, UINT64 id, UINT64 mask, const char *how, const char *state)
{
    struct producer *p = NULL;
    const char *name;
    char where[256];
    BOOL fresh = FALSE;

    AcquireSRWLockExclusive(&producers_lock);
    if ((p = producers_find(key)))
    {
        InterlockedOr64(&p->mask, (LONG64)mask);
        ReleaseSRWLockExclusive(&producers_lock);
        return p;
    }
    if (key_count >= PRODUCER_KEYS)
    {
        ReleaseSRWLockExclusive(&producers_lock);
        return NULL;
    }
    for (LONG i = 0; i < producer_count && !p; i++)
        if (producers[i].id == id)
            p = &producers[i];
    if (p)
        InterlockedOr64(&p->mask, (LONG64)mask);
    else if (producer_count < PRODUCERS)
    {
        p = &producers[producer_count];
        memset(p, 0, sizeof(*p));
        p->id = id;
        p->mask = (LONG64)mask;
        p->how = how;
        InterlockedIncrement(&producer_count);
        fresh = TRUE;
    }
    if (p)
    {
        keys[key_count].producer = p;
        keys[key_count].key = key;
        InterlockedIncrement(&key_count);
    }
    ReleaseSRWLockExclusive(&producers_lock);
    if (!fresh)
        return p;

    name = fast_id_name(id);
    mask_text(mask, where, sizeof(where));
    rebox_log("d3d12_xs: indirect producer: compute program %016llx%s%s%s writes mesh ExecuteIndirect arguments "
              "(found by %s; arguments at %s); compile state: %s. Its calls are essential from now on",
              (unsigned long long)id, name ? " (" : "", name ? name : "", name ? ")" : "", how, where, state);
    return p;
}

void producers_call(struct producer *p, BOOL ran, const char *why, BOOL quarantined, const char *error)
{
    UINT64 mask;
    LONG64 s;
    LONG now = frame;

    if (!p)
        return;
    InterlockedIncrement(&p->calls);
    mask = (UINT64)p->mask;
    s = InterlockedIncrement64(&seq);
    for (int i = 0; i < RANGES && mask; i++, mask >>= 1)
    {
        if (!(mask & 1))
            continue;
        if (ran)
        {
            InterlockedExchange64(&ranges[i].run_seq, s);
            InterlockedExchange(&ranges[i].run_frame, now);
        }
        else
        {
            InterlockedExchange64(&ranges[i].skip_seq, s);
            InterlockedExchange(&ranges[i].skip_frame, now);
        }
    }
    if (ran)
    {
        InterlockedIncrement(&p->ran);
        InterlockedIncrement(&p->census_ran);
    }
    else
    {
        InterlockedIncrement(&p->skipped);
        InterlockedIncrement(&p->census_skipped);
        p->last_skip = why;
    }
    if (quarantined && !InterlockedExchange(&p->quarantine_logged, 1))
        rebox_log("d3d12_xs: indirect producer %016llx (%s) is quarantined (REBOX_QUARANTINE) but writes indirect "
                  "arguments: it runs anyway", (unsigned long long)p->id, fast_id_name(p->id) ? fast_id_name(p->id) : "?");
    if (error && !InterlockedExchange(&p->error_logged, 1))
        rebox_log("d3d12_xs: *** INDIRECT PRODUCER FAILED: compute program %016llx (%s) writes mesh ExecuteIndirect "
                  "arguments and can't be compiled, so the meshes it culls for draw nothing. AnyPS5: %s",
                  (unsigned long long)p->id, fast_id_name(p->id) ? fast_id_name(p->id) : "?", error);
}

BOOL producers_scan_due(UINT64 key)
{
    LONG now = frame;
    UINT at = (UINT)(key ^ key >> 32) & (SCANS - 1);

    if (!key)
        return FALSE;
    for (UINT n = 0; n < 16; n++, at = (at + 1) & (SCANS - 1))
    {
        LONG64 k = scans[at][0];
        if (!k && !(k = InterlockedCompareExchange64(&scans[at][0], (LONG64)key, 0)))
            k = (LONG64)key; /* claimed */
        if ((UINT64)k != key)
            continue;
        return InterlockedExchange64(&scans[at][1], now + 1) != now + 1;
    }
    return FALSE; /* crowded: not scanned */
}

void producers_frame(LONG f)
{
    InterlockedExchange(&frame, f + 1);
}

void producers_census_log(void)
{
    LONG n = producer_count, ready = 0, skipped = 0, shown = 0;
    LONG e = InterlockedExchange(&eis, 0), es = InterlockedExchange(&eis_skipped, 0);
    LONG eu = InterlockedExchange(&eis_unproduced, 0), ef = InterlockedExchange(&eis_substituted, 0);
    char line[1024];
    int at = 0;

    if (!n && !e)
        return;
    for (LONG i = 0; i < n; i++)
    {
        struct producer *p = &producers[i];
        ready += p->ran != 0;
        skipped += p->census_skipped != 0;
    }
    rebox_log("d3d12_xs:   indirect producers: %ld seen, %ld of them ready (ran at least once), %ld skipped since the "
              "last census; mesh ExecuteIndirects since the last census: %ld, %ld of them with their producer skipped, "
              "%ld with no producer seen running", n, ready, skipped, e, es, eu);
    if (ef)
        rebox_log("d3d12_xs:   REBOX_INDIRECT_FALLBACK replaced %ld of those mesh ExecuteIndirects", ef);
    line[0] = 0;
    for (LONG i = 0; i < n && at < (int)sizeof(line) - 160; i++)
    {
        struct producer *p = &producers[i];
        LONG r = InterlockedExchange(&p->census_ran, 0), s = InterlockedExchange(&p->census_skipped, 0);
        const char *name = fast_id_name(p->id);
        if (!r && !s)
            continue;
        if (shown++ >= 12)
        {
            at += snprintf(line + at, sizeof(line) - at, " ...");
            break;
        }
        at += snprintf(line + at, sizeof(line) - at, " %016llx%s%.40s%s ran %ld, skipped %ld%s%s%s;",
                       (unsigned long long)p->id, name ? " (" : "", name ? name : "", name ? ")" : "", r, s,
                       s && p->last_skip ? " (" : "", s && p->last_skip ? p->last_skip : "", s && p->last_skip ? ")" : "");
    }
    if (shown)
        rebox_log("d3d12_xs:   indirect producers' calls since the last census:%s", line);
}

/* For tests: notes [va, va + bytes) as a mesh ExecuteIndirect's arguments are noted (0: nothing),
 * and returns how many producers have been found. */
UINT WINAPI ReboxIndirectProducers(UINT64 va, UINT64 bytes)
{
    UINT groups;
    if (va && bytes)
        producers_mesh_indirect(va, bytes, 0, FALSE, &groups);
    return (UINT)producer_count;
}
