/* Textures a title placed at its own GPU addresses (CreatePlacedResourceX, slot 44).
 *
 * On the console the texture is the memory at that address, in a Gfx10 tiled layout, and the
 * title fills it there: DirectStorage writes texture pools into it (already tiled, docs/d3d12x.md
 * "Texture tiling"). Here the texture has its own memory, because vkd3d-proton can't place an
 * image in the buffer-only GPU memory chunks. So each such texture is registered with its
 * address and swizzle mode, writes the shims see (ReboxTitleMemoryWritten: DirectStorage
 * requests) mark it dirty, and before a queue runs the title's command lists every dirty texture
 * is detiled from title memory (detile.cpp, addrlib) into an upload buffer and copied in. A new
 * texture starts dirty, for data written before it was created. Writes nothing reports (the CPU,
 * or GPU copies to the address) are caught by sampling: each upload keeps a fingerprint of a few
 * bytes spread over the texture's memory, and before each submission a slice of the textures
 * (WATCH_BUDGET, round robin) is fingerprinted again; a change makes it dirty. Textures up to
 * FULL_HASH_MAX are hashed whole, within FULL_HASH_BUDGET bytes a submission, so a small edit is
 * seen too (Unreal's font atlas gains glyphs as text is first drawn); larger ones are sampled,
 * which sees a texture rewritten (a streaming slot refilled), not a small edit between samples.
 * REBOX_TEXTURE_WATCH=0 turns the sampling off. Textures wholly in write-watched title memory
 * (xmem reserves it with MEM_WRITE_WATCH) aren't sampled: before each submission the pages written
 * since the last one are asked for (GetWriteWatch, reset), and the textures on them made dirty, as
 * xcl does (ivall/xcl, GPL-2.0-only, title_textures.rs). Textures wholly in physical-page views
 * are write-tracked instead (xmem.c, "Write tracking"): xmem makes their pages read-only
 * (ReboxTrackPages, at registration and before each upload reads them), notes the first write to
 * each 64 KiB unit (and every page newly mapped), and before each submission the ranges noted
 * (ReboxTrackedDirty) make the textures there dirty. They aren't sampled either, but for a slow
 * safety net: each is fingerprinted at most every SAFETY_MS. REBOX_TEXTURE_TRACKING=0 turns it
 * off. GPU memory chunks and title memory adopted as GPU memory can't be watched or tracked;
 * their textures are sampled as before.
 *
 * The swizzle mode: a console Layout from 0x100 is 0x100 plus the AMD swizzle mode (0x100 linear,
 * 0x105 SW_4KB_S, 0x10A SW_64KB_D; xcl and TuxBox, xcl checked against xg_xs.dll on 29,700 descs).
 * Layout 2, or desc flag 0x08000000 (TuxBox), is SW_64KB_D; Layout 0 (the driver's choice) is
 * SW_4KB_S for a plain single 2D texture whatever its size (the same two). REBOX_TEXTURE_SWIZZLE picks: "auto" (default: those; ROW_MAJOR descs linear;
 * cubemaps and volumes SW_4KB_S, other arrays SW_64KB_D, what the texture pools measured show),
 * "64KB_D", "4KB_S", "linear",
 * or "off" (no uploads). Each distinct console layout is logged once with what it got. */

#define COBJMACROS
#define WIDL_C_INLINE_WRAPPERS
#include <windows.h>
#include <d3d12.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/rebox_log.h"
#include "detile.h"
#include "gpu_memory.h"
#include "title_textures.h"
#include "console_compile.h"
#include "proxy.h"

struct title_texture
{
    ID3D12Resource *resource;   /* not a reference: removed when it is destroyed */
    UINT64 address, size;       /* the title's (GPU) address */
    const BYTE *cpu;            /* where the CPU reads it */
    struct detile_surface surface;
    D3D12_RESOURCE_STATES state;
    UINT subresources;
    BOOL dirty;
    BOOL unreadable;            /* memory not committed yet: rechecked with backoff */
    BOOL chunk;                 /* in our GPU memory chunks, which stay mapped */
    UINT check_at, backoff;     /* in submissions */
    UINT64 fingerprint;
    UINT uploads;
    BOOL gpu_written;           /* a GPU copy wrote it: title memory isn't its contents (font atlases) */
    BOOL target;                /* a render target kept in step anyway (small_cpu_filled_target) */
    BOOL watched;               /* wholly in write-watched memory: written pages find its changes */
    BOOL tracked;               /* wholly in physical-page views armed by xmem: write faults find its changes */
    ULONGLONG checked_at;       /* tracked: when last fingerprinted or uploaded (GetTickCount64) */
    UINT callback;
    struct title_texture *next;
    struct title_texture *rnext;  /* in by_resource */
    struct page_node *pages;      /* in by_page: each 64 KiB page of its GPU and CPU ranges */
    UINT page_count;
};

/* Indexes over the textures, under the lock: by resource, and by each 64 KiB page their GPU and
 * CPU ranges touch. ReboxTitleMemoryWritten (every DirectStorage request) and title_texture_info
 * (bindings) scanned every texture, tens of thousands once a title's world streams in, the first
 * under the exclusive lock the DirectStorage threads then queued on. */
#define INDEX_BUCKETS 65536
struct page_node { UINT64 page; struct title_texture *t; struct page_node *next; };
static struct title_texture *by_resource[INDEX_BUCKETS];
static struct page_node *by_page[INDEX_BUCKETS];
static UINT unpaged; /* textures by_page doesn't have (no memory for their pages): title_texture_at walks */

static SIZE_T resource_bucket(const ID3D12Resource *r)
{
    return (SIZE_T)(((UINT64)(ULONG_PTR)r >> 4) * 0x9e3779b97f4a7c15ull >> 48);
}

static SIZE_T page_bucket(UINT64 page)
{
    return (SIZE_T)(page * 0x9e3779b97f4a7c15ull >> 48);
}

/* Under the lock, exclusive. */
static void index_add(struct title_texture *t)
{
    UINT64 g0 = t->address >> 16, g1 = (t->address + t->size - 1) >> 16;
    UINT64 c0 = (UINT64)(ULONG_PTR)t->cpu >> 16, c1 = ((UINT64)(ULONG_PTR)t->cpu + t->size - 1) >> 16;
    BOOL cpu = t->cpu && (c0 != g0 || c1 != g1);
    UINT n = (UINT)(g1 - g0 + 1) + (cpu ? (UINT)(c1 - c0 + 1) : 0), k = 0;
    SIZE_T b = resource_bucket(t->resource);

    t->rnext = by_resource[b];
    by_resource[b] = t;
    console_bindings_changed(TRUE); /* title_texture_at finds it */
    if (!t->size || !(t->pages = calloc(n, sizeof(*t->pages))))
    {
        unpaged++;
        return;
    }
    for (UINT64 page = g0; page <= g1; page++)
        t->pages[k++].page = page;
    for (UINT64 page = c0; cpu && page <= c1; page++)
        t->pages[k++].page = page;
    for (k = 0; k < n; k++)
    {
        SIZE_T pb = page_bucket(t->pages[k].page);
        t->pages[k].t = t;
        t->pages[k].next = by_page[pb];
        by_page[pb] = &t->pages[k];
    }
    t->page_count = n;
}

/* Under the lock, exclusive. */
static void index_remove(struct title_texture *t)
{
    struct title_texture **r;
    for (r = &by_resource[resource_bucket(t->resource)]; *r; r = &(*r)->rnext)
        if (*r == t)
        {
            *r = t->rnext;
            break;
        }
    if (!t->pages)
        unpaged--;
    console_bindings_changed(TRUE);
    for (UINT k = 0; k < t->page_count; k++)
    {
        struct page_node **at;
        for (at = &by_page[page_bucket(t->pages[k].page)]; *at; at = &(*at)->next)
            if (*at == &t->pages[k])
            {
                *at = t->pages[k].next;
                break;
            }
    }
    free(t->pages);
    t->pages = NULL;
    t->page_count = 0;
}

/* Under the lock. */
static struct title_texture *texture_of(const ID3D12Resource *resource)
{
    struct title_texture *t;
    for (t = by_resource[resource_bucket(resource)]; t && t->resource != resource; t = t->rnext)
        ;
    return t;
}

static SRWLOCK lock = SRWLOCK_INIT;
static struct title_texture *textures;
static volatile LONG dirty_count;
static UINT watch_cursor;
static UINT submissions;

#define WATCH_SAMPLES 32
#define WATCH_BUDGET 512                   /* textures fingerprinted per submission */
#define FULL_HASH_MAX (4u << 20)           /* textures this size or smaller are hashed whole */
#define FULL_HASH_BUDGET (4u << 20)        /* bytes hashed whole per submission */
#define HOT_BUDGET (32u << 20)             /* bytes of rewritten textures checked every submission (a movie's planes; 8 MiB left some out) */
#define BYTE_HOT_BUDGET (8u << 20)         /* and of 1-byte presumed-hot ones (A8 glyph atlases, R8 planes), in turn */
#define OTHER_HOT_BUDGET (4u << 20)        /* and of the other presumed-hot ones, in turn */
#define SAFETY_MS 4000                     /* a write-tracked texture is fingerprinted at most this often */

static BOOL full_hash(const struct title_texture *t)
{
    return t->size <= FULL_HASH_MAX;
}

/* Small textures: a multiply-xor hash over every 8 bytes. Others: FNV-1a over 8 bytes at each of
 * WATCH_SAMPLES points spread over the texture, the first and last included. */
/* Bytes of title memory, or FALSE if they can't be read now: the title can unmap a texture's
 * physical pages at any moment (a read between a readability check and the unmap faulted). */
static BOOL read_title(const BYTE *from, void *to, SIZE_T size)
{
    SIZE_T done = 0;
    return ReadProcessMemory(GetCurrentProcess(), from, to, size, &done) && done == size;
}

static UINT64 fingerprint(const struct title_texture *t)
{
    UINT64 h = 0xcbf29ce484222325ull, step = t->size > 8 ? (t->size - 8) / (WATCH_SAMPLES - 1) : 0;

    if (full_hash(t))
    {
        BYTE *chunk = malloc(64u << 10);
        if (!chunk)
            return t->fingerprint;
        for (UINT64 at = 0; at < t->size; at += 64u << 10)
        {
            SIZE_T n = (SIZE_T)min(t->size - at, 64u << 10), i;
            UINT64 v;
            if (!read_title(t->cpu + at, chunk, n))
            {
                free(chunk);
                return t->fingerprint; /* gone now: unchanged, nothing to upload */
            }
            for (i = 0; i + 8 <= n; i += 8)
            {
                memcpy(&v, chunk + i, 8);
                h = (h ^ v) * 0x9e3779b97f4a7c15ull;
                h ^= h >> 29;
            }
            for (; i < n; i++)
                h = (h ^ chunk[i]) * 0x100000001b3ull;
        }
        free(chunk);
        return h;
    }

    for (UINT i = 0; i < WATCH_SAMPLES; i++)
    {
        UINT64 v = 0;
        if (!read_title(t->cpu + (t->size > 8 ? i * step : 0), &v, t->size >= 8 ? 8 : t->size))
            return t->fingerprint;
        h = (h ^ v) * 0x100000001b3ull;
    }
    return h;
}

/* The same hash over a copy of the texture's bytes (what an upload read). */
static UINT64 fingerprint_bytes(const struct title_texture *t, const BYTE *bytes)
{
    UINT64 h = 0xcbf29ce484222325ull, step = t->size > 8 ? (t->size - 8) / (WATCH_SAMPLES - 1) : 0, v;

    if (full_hash(t))
    {
        UINT64 i;
        for (i = 0; i + 8 <= t->size; i += 8)
        {
            memcpy(&v, bytes + i, 8);
            h = (h ^ v) * 0x9e3779b97f4a7c15ull;
            h ^= h >> 29;
        }
        for (; i < t->size; i++)
            h = (h ^ bytes[i]) * 0x100000001b3ull;
        return h;
    }
    for (UINT i = 0; i < WATCH_SAMPLES; i++)
    {
        v = 0;
        memcpy(&v, bytes + (t->size > 8 ? i * step : 0), t->size >= 8 ? 8 : (SIZE_T)t->size);
        h = (h ^ v) * 0x100000001b3ull;
    }
    return h;
}

static BOOL tracking_enabled(void)
{
    static int enabled = -1;
    if (enabled == -1)
    {
        char *v = rebox_getenv("REBOX_TEXTURE_TRACKING");
        enabled = !v || strcmp(v, "0");
        free(v);
    }
    return enabled;
}

BOOL title_textures_tracking(void)
{
    return tracking_enabled();
}

/* xmem's write tracking for physical-page views: arming a range, and taking the ranges noted
 * dirty since. */
static BOOL (WINAPI *track_pages)(ULONGLONG, ULONGLONG);
static UINT (WINAPI *tracked_dirty)(ULONGLONG *, UINT, BOOL *);

static BOOL CALLBACK find_tracking(INIT_ONCE *once, void *param, void **ctx)
{
    HMODULE xmem = tracking_enabled() ? GetModuleHandleA("xmem.dll") : NULL;
    track_pages = xmem ? (void *)GetProcAddress(xmem, "ReboxTrackPages") : NULL;
    tracked_dirty = xmem ? (void *)GetProcAddress(xmem, "ReboxTrackedDirty") : NULL;
    if (!track_pages || !tracked_dirty)
        track_pages = NULL, tracked_dirty = NULL;
    return TRUE;
}

static BOOL in_chunk(UINT64 start, UINT64 end);
static struct { LONG64 ranges, textures, safety_checks, safety_changes, arm_ticks; } track_stats;
static LONG64 stat_ticks(void);

/* Under the lock: arms the texture's memory if it can be tracked; TRUE if it is now. Its memory is
 * read-only after this, so a write it doesn't see faults and is noted. Not memory made GPU memory
 * (the driver's mapping) or write-watched. */
static BOOL track(const struct title_texture *t)
{
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    LONG64 start;
    BOOL ok;

    InitOnceExecuteOnce(&once, find_tracking, NULL, NULL);
    if (!track_pages || t->chunk || !t->cpu || t->watched || t->gpu_written
            || in_chunk((UINT64)(ULONG_PTR)t->cpu, (UINT64)(ULONG_PTR)t->cpu + t->size))
        return FALSE;
    start = stat_ticks();
    ok = track_pages((UINT64)(ULONG_PTR)t->cpu, t->size);
    track_stats.arm_ticks += stat_ticks() - start;
    return ok;
}

static BOOL watch_enabled(void)
{
    static int enabled = -1;
    if (enabled == -1)
    {
        char *v = rebox_getenv("REBOX_TEXTURE_WATCH");
        enabled = !v || strcmp(v, "0");
        free(v);
    }
    return enabled;
}

static BOOL readable(UINT64 address, UINT64 size);

/* Called with the lock held: a texture whose memory wasn't readable is checked again after 1, 2,
 * 4, ... up to 1024 submissions (TomBox backs off the same way), and uploaded once it is. */
static void recheck(struct title_texture *t)
{
    if (submissions < t->check_at)
        return;
    if (!readable((UINT64)(ULONG_PTR)t->cpu, t->size))
    {
        t->backoff = t->backoff ? min(t->backoff * 2, 1024) : 1;
        t->check_at = submissions + t->backoff;
        return;
    }
    t->unreadable = FALSE;
    t->dirty = TRUE;
    InterlockedIncrement(&dirty_count);
}

/* Called with the lock held: memory outside our chunks can be decommitted or remapped after the
 * texture was registered (a UE5 title remaps title pages), so it is checked before it is read; gone,
 * the texture waits for it again like one placed on uncommitted memory. */
static BOOL still_readable(struct title_texture *t)
{
    if (t->chunk || readable((UINT64)(ULONG_PTR)t->cpu, t->size))
        return TRUE;
    t->unreadable = TRUE;
    t->check_at = submissions + 1;
    t->backoff = 1;
    if (t->dirty)
    {
        t->dirty = FALSE;
        InterlockedDecrement(&dirty_count);
    }
    return FALSE;
}

/* Rewritten by the CPU since it was made, or likely to be: uploaded twice, or once if it has one
 * mip and an uncompressed format. A video's planes (a Bink decoder's R8 luma and chroma, rewritten
 * every frame and never reported) look like that; the round robin over a world's tens of thousands
 * of textures came back to them so rarely that a cutscene showed its first, empty frame. */
static BOOL hot(const struct title_texture *t)
{
    UINT block = 0;
    return t->uploads >= 2
            || (t->uploads == 1 && t->surface.mips == 1 && detile_element(t->surface.format, &block) && block == 1);
}

/* Hot textures by how sure: rewritten (uploaded again since made), or presumed so (one upload,
 * one mip, uncompressed) with 1-byte texels (glyph atlases, video planes) or others. 0: not hot. */
enum { HOT_REWRITTEN = 1, HOT_BYTE, HOT_OTHER };

static int hot_tier(const struct title_texture *t)
{
    UINT block = 0;
    if (!hot(t))
        return 0;
    if (t->uploads >= 2)
        return HOT_REWRITTEN;
    return detile_element(t->surface.format, &block) == 1 ? HOT_BYTE : HOT_OTHER;
}

/* Where each presumed-hot tier's turn got to (its textures counted in list order), and how many
 * submissions its last full turn took (for the stats log). */
static UINT hot_cursor[HOT_OTHER + 1], hot_turn[HOT_OTHER + 1], hot_turn_start[HOT_OTHER + 1];

/* Write-watched title memory, under the lock: xmem's watched reservations (ReboxWatchedMemory),
 * each with the span its watched textures lie in (only that is asked about), and the GPU memory
 * chunks, which can't be watched: memory adopted there is the Vulkan driver's mapping now, which
 * the kernel's write watch skips without a word. Both reread when they change. */
struct watched_range { UINT64 start, end, lo, hi; BOOL failed; };
static struct watched_range *watched_ranges;
static UINT watched_count;
static struct gpu_memory_region *chunk_regions;
static UINT chunk_count;
static LONG watched_generation = -1, chunks_generation = -1;

static BOOL in_chunk(UINT64 start, UINT64 end)
{
    for (UINT i = 0; i < chunk_count; i++)
        if (start < (UINT64)(ULONG_PTR)chunk_regions[i].cpu + chunk_regions[i].size
                && (UINT64)(ULONG_PTR)chunk_regions[i].cpu < end)
            return TRUE;
    return FALSE;
}

static void set_watched(struct title_texture *t)
{
    UINT64 start = (UINT64)(ULONG_PTR)t->cpu, end = start + t->size;

    t->watched = FALSE;
    if (t->chunk || !t->cpu || in_chunk(start, end))
        return;
    for (UINT i = 0; i < watched_count; i++)
    {
        struct watched_range *r = &watched_ranges[i];
        if (!r->failed && start >= r->start && end <= r->end)
        {
            t->watched = TRUE;
            r->lo = r->hi > r->lo ? min(r->lo, start) : start;
            r->hi = max(r->hi, end);
            return;
        }
    }
}

static void refresh_watched(void)
{
    static UINT (WINAPI *ranges_of)(ULONGLONG *, UINT, LONG *);
    static BOOL resolved;
    LONG generation = 0, chunks = gpu_memory_generation();
    ULONGLONG *pairs;
    UINT n, cap;

    if (!resolved)
    {
        HMODULE xmem = GetModuleHandleA("xmem.dll");
        ranges_of = xmem ? (void *)GetProcAddress(xmem, "ReboxWatchedMemory") : NULL;
        resolved = TRUE;
    }
    if (!ranges_of)
        return;
    n = ranges_of(NULL, 0, &generation);
    if (generation == watched_generation && chunks == chunks_generation)
        return;
    for (;;) /* again if they grew in between */
    {
        cap = n + 16;
        if (!(pairs = malloc(cap * 2 * sizeof(*pairs))))
            return;
        if ((n = ranges_of(pairs, cap, &generation)) <= cap)
            break;
        free(pairs);
    }
    free(watched_ranges);
    if (!(watched_ranges = calloc(n ? n : 1, sizeof(*watched_ranges))))
        n = 0;
    for (UINT i = 0; i < n; i++)
        watched_ranges[i].start = pairs[2 * i], watched_ranges[i].end = pairs[2 * i] + pairs[2 * i + 1];
    free(pairs);
    watched_count = n;
    for (cap = chunk_count ? chunk_count * 2 : 64;; cap *= 2)
    {
        free(chunk_regions);
        if (!(chunk_regions = calloc(cap, sizeof(*chunk_regions))))
        {
            chunk_count = 0;
            break;
        }
        if ((chunk_count = gpu_memory_regions(chunk_regions, cap)) < cap)
            break;
    }
    watched_generation = generation;
    chunks_generation = chunks;
    {
        static LONG logged;
        UINT count = 0, in = 0;
        for (struct title_texture *t = textures; t; t = t->next, count++)
        {
            set_watched(t);
            in += t->watched;
            /* Made GPU memory in place since: the driver's mapping isn't tracked. */
            if (t->tracked && in_chunk((UINT64)(ULONG_PTR)t->cpu, (UINT64)(ULONG_PTR)t->cpu + t->size))
            {
                t->tracked = FALSE;
                if (!t->dirty && !t->unreadable && !t->gpu_written)
                {
                    t->dirty = TRUE;
                    InterlockedIncrement(&dirty_count);
                }
            }
        }
        if (InterlockedIncrement(&logged) <= 8)
            rebox_log("d3d12_xs: %u write-watched reservation(s), %u GPU memory chunk(s); %u of %u placed textures "
                      "write-watched", watched_count, chunk_count, in, count);
    }
}

/* Under the lock: textures whose memory overlaps [start, end), written, made dirty. */
static UINT mark_written(UINT64 start, UINT64 end)
{
    UINT changed = 0;
    for (UINT64 page = start >> 16; page <= (end - 1) >> 16; page++)
        for (struct page_node *n = by_page[page_bucket(page)]; n; n = n->next)
        {
            struct title_texture *t = n->t;
            UINT64 cpu = (UINT64)(ULONG_PTR)t->cpu;
            if (n->page == page && t->watched && !t->dirty && !t->unreadable && !t->gpu_written
                    && start < cpu + t->size && cpu < end)
            {
                t->dirty = TRUE;
                InterlockedIncrement(&dirty_count);
                changed++;
            }
        }
    return changed;
}

/* Under the lock: the pages written in [start, end) since the last asking, which resets them.
 * FALSE if the system won't say (not write-watched memory). */
static BOOL scan_written(UINT64 start, UINT64 end, UINT *changed)
{
    static void *written[4096];
    UINT64 at = start;

    while (at < end)
    {
        ULONG_PTR count = ARRAYSIZE(written);
        ULONG granularity = 0;
        UINT64 run_start = 0, run_end = 0;

        if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, (void *)(ULONG_PTR)at, (SIZE_T)(end - at), written, &count,
                          &granularity) || !granularity)
            return FALSE;
        for (ULONG_PTR i = 0; i < count; i++)
        {
            UINT64 page = (UINT64)(ULONG_PTR)written[i];
            if (page != run_end)
            {
                if (run_end)
                    *changed += mark_written(run_start, run_end);
                run_start = page;
            }
            run_end = page + granularity;
        }
        if (run_end)
            *changed += mark_written(run_start, run_end);
        if (count < ARRAYSIZE(written))
            break;
        at = run_end;
    }
    return TRUE;
}

/* Under the lock: textures in write-watched memory written since the last submission made dirty.
 * Each watched span is asked about around the GPU memory chunks in it. */
static UINT watch_written(void)
{
    UINT changed = 0;

    refresh_watched();
    for (UINT i = 0; i < watched_count; i++)
    {
        struct watched_range *r = &watched_ranges[i];
        UINT64 at = r->lo & ~0xfffull, end = (r->hi + 0xfff) & ~0xfffull;
        BOOL ok = TRUE;

        while (!r->failed && at < end && ok)
        {
            UINT64 next = end, after = end;
            for (UINT k = 0; k < chunk_count; k++)
            {
                UINT64 c0 = (UINT64)(ULONG_PTR)chunk_regions[k].cpu, c1 = c0 + chunk_regions[k].size;
                if (c0 < end && c1 > at && max(c0, at) < next)
                    next = max(c0, at), after = c1;
            }
            if (next > at)
                ok = scan_written(at, next, &changed);
            at = after;
        }
        if (!ok)
        {
            /* Released behind xmem's back, or not watched after all: its textures are sampled. */
            static LONG logged;
            if (InterlockedIncrement(&logged) <= 4)
                rebox_log("d3d12_xs: GetWriteWatch on %#llx-%#llx failed (error %lu); its textures are fingerprinted",
                          (unsigned long long)r->start, (unsigned long long)r->end, GetLastError());
            r->failed = TRUE;
            for (struct title_texture *t = textures; t; t = t->next)
                if (t->watched && (UINT64)(ULONG_PTR)t->cpu >= r->start && (UINT64)(ULONG_PTR)t->cpu < r->end)
                {
                    t->watched = FALSE;
                    if (!t->dirty && !t->unreadable && !t->gpu_written)
                    {
                        t->dirty = TRUE; /* written or not: its fingerprint is old */
                        InterlockedIncrement(&dirty_count);
                    }
                }
        }
    }
    return changed;
}

/* Under the lock: textures on what xmem's write tracking noted dirty since the last submission
 * (written pages of physical-page views, pages mapped anew) made dirty, tracked or not. */
static UINT track_written(void)
{
    static ULONGLONG pairs[2 * 1024];
    UINT changed = 0, n;
    BOOL overflow = FALSE, lost = FALSE;

    if (!tracked_dirty)
        return 0;
    do
    {
        n = tracked_dirty(pairs, ARRAYSIZE(pairs) / 2, &overflow);
        lost |= overflow;
        track_stats.ranges += n;
        for (UINT i = 0; i < n; i++)
            for (UINT64 start = pairs[2 * i], end = start + pairs[2 * i + 1], page = start >> 16; page <= (end - 1) >> 16; page++)
                for (struct page_node *node = by_page[page_bucket(page)]; node; node = node->next)
                {
                    struct title_texture *t = node->t;
                    UINT64 cpu = (UINT64)(ULONG_PTR)t->cpu;
                    if (node->page == page && !t->dirty && !t->unreadable && !t->gpu_written
                            && start < cpu + t->size && cpu < end)
                    {
                        t->dirty = TRUE;
                        InterlockedIncrement(&dirty_count);
                        changed++;
                    }
                }
    } while (n == ARRAYSIZE(pairs) / 2);
    if (lost) /* too many to keep: every tracked texture may have changed */
        for (struct title_texture *t = textures; t; t = t->next)
            if (t->tracked && !t->dirty && !t->unreadable && !t->gpu_written)
            {
                t->dirty = TRUE;
                InterlockedIncrement(&dirty_count);
                changed++;
            }
    track_stats.textures += changed;
    return changed;
}

/* Under the lock: a tracked texture's slow safety net is due (its fingerprint, at most every
 * SAFETY_MS). Writes tracking can't see: the title's system calls other than ReadFile, a view
 * briefly writable while it is mapped again. */
static BOOL safety_due(struct title_texture *t, ULONGLONG now)
{
    if (now - t->checked_at < SAFETY_MS)
        return FALSE;
    t->checked_at = now;
    track_stats.safety_checks++;
    return TRUE;
}

/* Under the lock: a hot texture's fingerprint compared, its bytes counted in *hashed if hashed
 * whole; TRUE if it changed (made dirty). */
static BOOL hot_check(struct title_texture *t, UINT64 *hashed)
{
    if (t->watched || t->tracked || t->target || t->dirty || t->gpu_written || t->unreadable || !still_readable(t))
        return FALSE;
    *hashed += full_hash(t) ? t->size : 0;
    if (fingerprint(t) == t->fingerprint)
        return FALSE;
    t->dirty = TRUE;
    InterlockedIncrement(&dirty_count);
    return TRUE;
}

/* Called with the lock held. */
static void watch(void)
{
    struct title_texture *t = textures;
    UINT skipped = 0, checked = 0, changed = 0, written = watch_written(), tracked = track_written();
    UINT64 hashed = 0;
    BOOL sample = watch_enabled();
    ULONGLONG now = GetTickCount64();

    if (written)
    {
        static LONG logged;
        if (InterlockedIncrement(&logged) <= 4)
            rebox_log("d3d12_xs: %u placed texture(s) on pages written since the last submission", written);
    }
    if (tracked)
    {
        static LONG logged;
        if (InterlockedIncrement(&logged) <= 4)
            rebox_log("d3d12_xs: %u placed texture(s) on physical pages written or mapped since the last submission",
                      tracked);
    }
    for (; t && skipped < watch_cursor; t = t->next)
        skipped++;
    if (!t)
        t = textures, watch_cursor = 0;
    for (; t && checked < WATCH_BUDGET && hashed < FULL_HASH_BUDGET; t = t->next, checked++)
        if (t->unreadable)
            recheck(t);
        else if (sample && !t->watched && !t->dirty && !t->gpu_written && (!t->tracked || safety_due(t, now))
                 && still_readable(t) && (hashed += full_hash(t) ? t->size : 0, TRUE)
                 && fingerprint(t) != t->fingerprint)
        {
            t->dirty = TRUE;
            InterlockedIncrement(&dirty_count);
            changed++;
            if (t->tracked)
            {
                /* A write tracking missed. */
                static LONG logged;
                track_stats.safety_changes++;
                if (InterlockedIncrement(&logged) <= 8)
                    rebox_log("d3d12_xs: write-tracked texture at %#llx (%ux%u) changed without a write noted; found "
                              "by its fingerprint", (unsigned long long)t->address, t->surface.width, t->surface.height);
            }
        }
    watch_cursor = t ? watch_cursor + checked : 0;
    /* Hot textures, ones the CPU has rewritten since they were made (a glyph atlas gains letters
     * as text is laid out, and the frame draws them at once), are checked every submission: the
     * round robin over hundreds of textures came back to them only after the text had been
     * drawn without its letters. Large ones are sampled, which costs little: a video's planes, which
     * the decoder rewrites every frame and the round robin reached only every few frames, so the
     * video jumped back to a buffer's older frame. Not render targets: uploading those more often
     * overwrote what the GPU drew into them (menu text got worse). */
    /* The rewritten ones first, from the newest, within HOT_BUDGET; then the presumed-hot ones of
     * each tier in turn, from where the last submission's turn stopped, within the tier's budget
     * and what the rewritten ones left. From the newest only, the budget went to the textures a
     * streaming world had made since: a credits sequence's glyph atlas, made early, was never
     * reached, nor (being presumed hot) by the round robin's budget soon enough, and its new
     * letters were missing. Large ones are sampled at no cost to a budget, all of them. */
    if (sample)
    {
        UINT64 left = HOT_BUDGET;
        hashed = 0;
        for (t = textures; t; t = t->next)
            if (hot_tier(t) == HOT_REWRITTEN && hashed < HOT_BUDGET && hot_check(t, &hashed))
                changed++;
        left = hashed < HOT_BUDGET ? HOT_BUDGET - hashed : 0;
        for (int tier = HOT_BYTE; tier <= HOT_OTHER; tier++)
        {
            UINT64 budget = (tier == HOT_BYTE ? BYTE_HOT_BUDGET : OTHER_HOT_BUDGET) + left;
            UINT index = 0, next = 0, count = 0;
            BOOL stopped = FALSE;
            hashed = 0;
            /* From the cursor to the end, then from the start up to it. */
            for (int pass = 0; pass < 2; pass++)
            {
                index = 0;
                for (t = textures; t; t = t->next)
                {
                    if (hot_tier(t) != tier)
                        continue;
                    index++;
                    if (pass == 0)
                        count++;
                    if ((pass == 0) != (index > hot_cursor[tier]))
                        continue;
                    if (!full_hash(t))
                    {
                        if (hot_check(t, &hashed))
                            changed++;
                        continue;
                    }
                    if (stopped)
                        continue;
                    if (hashed >= budget)
                    {
                        stopped = TRUE;
                        next = index - 1;
                        continue;
                    }
                    if (hot_check(t, &hashed))
                        changed++;
                }
            }
            if (!stopped || next >= count)
            {
                next = 0;
                hot_turn[tier] = submissions - hot_turn_start[tier];
                hot_turn_start[tier] = submissions;
            }
            hot_cursor[tier] = next;
            left = hashed < budget ? budget - hashed : 0;
        }
    }
    if (changed)
    {
        static LONG logged;
        if (InterlockedIncrement(&logged) <= 4)
            rebox_log("d3d12_xs: %u placed texture(s) changed in title memory without a report", changed);
    }
}

static int mode_setting(void)
{
    static int setting = -2;
    if (setting == -2)
    {
        char *v = rebox_getenv("REBOX_TEXTURE_SWIZZLE");
        setting = !v || !strcmp(v, "auto") ? -1 : !strcmp(v, "off") ? -3 : !strcmp(v, "64KB_D") ? DETILE_64KB_D
                : !strcmp(v, "4KB_S") ? DETILE_4KB_S : !strcmp(v, "linear") ? DETILE_LINEAR : -1;
        free(v);
    }
    return setting;
}


/* 2D arrays are read a mip (all its slices) at a time: the texture pools measured store them that way, and a
 * pool is the texture's memory image (helpers/textures: every array matches Leonida's decoder
 * mip-major, cubemaps and 2D textures match as they are). REBOX_TEXTURE_ARRAYS=slice-major
 * reads them in addrlib's order instead, a slice (its whole mip chain) at a time. */
static BOOL arrays_mip_major(void)
{
    static int setting = -1;
    if (setting == -1)
    {
        char *v = rebox_getenv("REBOX_TEXTURE_ARRAYS");
        setting = !v || strcmp(v, "slice-major");
        free(v);
    }
    return setting;
}

static void WINAPI destroyed(void *data)
{
    struct title_texture *t = data, **p;

    AcquireSRWLockExclusive(&lock);
    for (p = &textures; *p; p = &(*p)->next)
        if (*p == t)
        {
            *p = t->next;
            index_remove(t);
            if (t->dirty)
                InterlockedDecrement(&dirty_count);
            free(t);
            break;
        }
    ReleaseSRWLockExclusive(&lock);
}

/* The CPU address of a title GPU address: through GPU memory's mapping when it is in one of its
 * chunks (identity-mapped or not; *chunk set), else the same address (memory the title allocated
 * itself). */
static const BYTE *cpu_address(UINT64 address, UINT64 size, BOOL *chunk)
{
    struct gpu_memory_region r[64];
    UINT n = gpu_memory_regions(r, ARRAYSIZE(r));

    *chunk = FALSE;
    for (UINT i = 0; i < n; i++)
        if (address >= r[i].gpu && address + size <= r[i].gpu + r[i].size)
        {
            *chunk = TRUE;
            return (const BYTE *)r[i].cpu + (address - r[i].gpu);
        }
    return (const BYTE *)(ULONG_PTR)address;
}

/* Whether [address, address + size) is committed, readable memory. Title memory made GPU memory in
 * place is, without asking: Wine's VirtualQuery doesn't know the Vulkan driver's mappings, and the
 * walk over them on every submission held the render thread for most of an intro's frame time. */
static BOOL readable(UINT64 address, UINT64 size)
{
    UINT64 at = address, end = address + size, extent;

    while (at < end)
    {
        MEMORY_BASIC_INFORMATION mbi;
        if (gpu_memory_identity_extent(at, &extent))
        {
            at = extent;
            continue;
        }
        if (!VirtualQuery((void *)(ULONG_PTR)at, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT
                || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) || !mbi.Protect)
            return FALSE;
        at = (UINT64)(ULONG_PTR)mbi.BaseAddress + mbi.RegionSize;
    }
    return TRUE;
}

/* Whether a texture of size bytes at title GPU address address has memory that can't be read yet
 * (not in GPU memory, not committed): proxy.c gives such a view texture memory only when it has
 * something to hold (REBOX_VIEW_TEXTURE_DEFER). */
BOOL title_texture_memory_unreadable(UINT64 address, UINT64 size)
{
    BOOL chunk;
    const BYTE *cpu = cpu_address(address, size, &chunk);
    return size && !chunk && !readable((UINT64)(ULONG_PTR)cpu, size);
}

/* The console surface for a texture: its shape from the PC desc, its swizzle mode from the setting,
 * the console layout or the shape. FALSE for what isn't uploaded (buffers, MSAA, unknown formats). */
static BOOL console_surface(const D3D12_RESOURCE_DESC *console_desc, const D3D12_RESOURCE_DESC *desc,
                            struct detile_surface *surface)
{
    int setting = mode_setting();
    UINT block, bytes;

    if (setting == -3 || !desc || desc->Dimension < D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || desc->SampleDesc.Count > 1 || !(bytes = detile_element(desc->Format, &block)))
        return FALSE;
    memset(surface, 0, sizeof(*surface));
    surface->format = desc->Format;
    surface->width = (UINT)desc->Width;
    surface->height = desc->Height;
    surface->layers = desc->DepthOrArraySize;
    surface->volume = desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    surface->mips = desc->MipLevels ? desc->MipLevels : 1;
    surface->mip_major = arrays_mip_major();
    if (setting >= 0)
        surface->mode = setting;
    else if (console_desc->Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR || console_desc->Layout == 0x100)
        surface->mode = DETILE_LINEAR;
    else if (console_desc->Layout == 0x105)
        surface->mode = DETILE_4KB_S;
    else if (console_desc->Layout == 0x10a || console_desc->Layout == 2 || (console_desc->Flags & 0x08000000))
        surface->mode = DETILE_64KB_D;
    /* By shape, as Leonida's census of the pools found (docs/textures_arrays.md): cubemaps (6
     * faces) and volumes SW_4KB_S, other 2D arrays SW_64KB_D, whatever their size. */
    else if (surface->volume || surface->layers == 6)
        surface->mode = DETILE_4KB_S;
    else if (surface->layers > 1)
        surface->mode = DETILE_64KB_D;
    else
        surface->mode = DETILE_4KB_S;
    return TRUE;
}

/* A small single-sample colour render target with nothing else allowed (REBOX_SYNC_SMALL_TARGETS=0
 * turns this off): 64x64 to 1024x1024, as xcl's. */
static BOOL small_cpu_filled_target(const D3D12_RESOURCE_DESC *d)
{
    static int enabled = -1;
    if (enabled < 0)
    {
        char *v = rebox_getenv("REBOX_SYNC_SMALL_TARGETS");
        enabled = !v || strcmp(v, "0");
        free(v);
    }
    return enabled && d->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && (d->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
                        | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) == D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
        && d->SampleDesc.Count <= 1 && d->DepthOrArraySize == 1
        && d->Width >= 64 && d->Width <= 1024 && d->Height >= 64 && d->Height <= 1024;
}

void title_texture_gpu_written(ID3D12Resource *resource)
{
    struct title_texture *t;

    if (!resource || !textures)
        return;
    AcquireSRWLockExclusive(&lock);
    t = texture_of(resource);
    if (t && !t->gpu_written)
    {
        static LONG logged;
        t->gpu_written = TRUE;
        if (t->dirty)
        {
            t->dirty = FALSE;
            InterlockedDecrement(&dirty_count);
        }
        if (InterlockedIncrement(&logged) <= 4)
            rebox_log("d3d12_xs: texture at %#llx (%ux%u) is written by GPU copies; title memory is no longer "
                         "uploaded over it", (unsigned long long)t->address, t->surface.width, t->surface.height);
    }
    ReleaseSRWLockExclusive(&lock);
}

/* resource is the title's no longer (a retired view texture, proxy.c): not filled from title
 * memory, and not found at its address, from now on. */
void title_texture_unregister(ID3D12Resource *resource)
{
    struct title_texture *t, **p;
    ID3DDestructionNotifier *notifier;

    AcquireSRWLockExclusive(&lock);
    if ((t = texture_of(resource)))
    {
        for (p = &textures; *p && *p != t; p = &(*p)->next)
            ;
        if (*p)
            *p = t->next;
        index_remove(t);
        if (t->dirty)
            InterlockedDecrement(&dirty_count);
        if (SUCCEEDED(ID3D12Resource_QueryInterface(resource, &IID_ID3DDestructionNotifier, (void **)&notifier)))
        {
            ID3DDestructionNotifier_UnregisterDestructionCallback(notifier, t->callback);
            ID3DDestructionNotifier_Release(notifier);
            free(t);
        }
        /* else its callback still names it: left allocated */
    }
    ReleaseSRWLockExclusive(&lock);
}

/* Under the lock. */
static ID3D12Resource *texture_at_locked(UINT64 address)
{
    struct title_texture *t;
    ID3D12Resource *res = NULL;

    /* By the page of its address: every T# a draw binds that isn't in a heap asked, and walking
     * every texture (tens of thousands in a world) took most of a draw's bindings. Pages are
     * indexed newest first, as the list is kept. */
    if (!unpaged)
    {
        for (struct page_node *n = by_page[page_bucket(address >> 16)]; n && !res; n = n->next)
            if (n->page == address >> 16 && n->t->address == address)
                res = n->t->resource;
    }
    else
        for (t = textures; t && !res; t = t->next)
            if (t->address == address)
                res = t->resource;
    return res;
}

ID3D12Resource *title_texture_at(UINT64 address)
{
    ID3D12Resource *res;

    AcquireSRWLockShared(&lock);
    res = texture_at_locked(address);
    ReleaseSRWLockShared(&lock);
    return res;
}

/* title_texture_at without waiting: FALSE if the lock is held (an upload or watch pass holds it
 * exclusive for milliseconds). */
BOOL title_texture_at_nowait(UINT64 address, ID3D12Resource **res)
{
    if (!TryAcquireSRWLockShared(&lock))
        return FALSE;
    *res = texture_at_locked(address);
    ReleaseSRWLockShared(&lock);
    return TRUE;
}

BOOL title_texture_info(ID3D12Resource *resource, UINT64 *address, UINT *uploads)
{
    struct title_texture *t;
    BOOL found = FALSE;

    AcquireSRWLockShared(&lock);
    if ((t = texture_of(resource)))
        *address = t->address, *uploads = t->uploads, found = TRUE;
    ReleaseSRWLockShared(&lock);
    return found;
}

/* Whether the texture can be released and made again from title memory alone (a view texture
 * nothing names, proxy.c): registered, never written by the GPU, and not holding contents its
 * memory no longer has (uploaded, then the memory went). *filled: uploaded at least once. */
BOOL title_texture_refillable(ID3D12Resource *resource, BOOL *filled)
{
    struct title_texture *t;
    BOOL refillable = FALSE;

    *filled = FALSE;
    AcquireSRWLockShared(&lock);
    if ((t = texture_of(resource)))
    {
        *filled = t->uploads > 0;
        refillable = !t->gpu_written && !t->target && !(t->unreadable && t->uploads);
    }
    ReleaseSRWLockShared(&lock);
    return refillable;
}

/* The bytes a texture takes in console memory (0 if it isn't uploaded). */
UINT64 title_texture_size(const D3D12_RESOURCE_DESC *desc)
{
    struct detile_surface surface;
    return console_surface(desc, desc, &surface) ? detile_size(&surface) : 0;
}

void title_texture_register(ID3D12Resource *resource, UINT64 address, const D3D12_RESOURCE_DESC *console_desc,
                            const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES state)
{
    static struct { UINT layout, x34; } seen[32];
    static LONG seen_count;
    struct title_texture *t;
    ID3DDestructionNotifier *notifier;
    UINT x34 = *(const UINT *)((const BYTE *)console_desc + 0x34);
    BOOL logged = FALSE;

    if (!resource || !(t = calloc(1, sizeof(*t))))
        return;
    if (!console_surface(console_desc, desc, &t->surface))
    {
        free(t);
        return;
    }
    t->resource = resource;
    t->address = address;
    t->state = state;
    t->subresources = t->surface.mips * (t->surface.volume ? 1 : t->surface.layers);
    t->size = detile_size(&t->surface);

    for (LONG i = 0; i < seen_count && !logged; i++)
        logged = seen[i].layout == (UINT)console_desc->Layout && seen[i].x34 == x34;
    if (!logged && seen_count < (LONG)ARRAYSIZE(seen))
    {
        LONG i = InterlockedIncrement(&seen_count) - 1;
        if (i < (LONG)ARRAYSIZE(seen))
            seen[i].layout = console_desc->Layout, seen[i].x34 = x34;
        rebox_log("d3d12_xs: placed texture layout %#x (+0x34 %#x, +0x38 %#llx), %ux%u x%u, %u mips, format %u: "
                     "%s, %llu bytes at %#llx", console_desc->Layout, x34,
                     *(const unsigned long long *)((const BYTE *)console_desc + 0x38), t->surface.width,
                     t->surface.height, t->surface.layers, t->surface.mips, desc->Format,
                     t->surface.mode == DETILE_64KB_D ? "SW_64KB_D" : t->surface.mode == DETILE_4KB_S ? "SW_4KB_S" : "linear",
                     (unsigned long long)t->size, (unsigned long long)address);
    }
    {
        BOOL chunk;
        t->cpu = cpu_address(address, t->size, &chunk);
        t->chunk = chunk;
        if (t->size && !chunk && !readable((UINT64)(ULONG_PTR)t->cpu, t->size))
        {
            static LONG logged;
            if (InterlockedIncrement(&logged) <= 4)
                rebox_log("d3d12_xs: placed texture at %#llx: its memory isn't readable yet; rechecked until it is",
                             (unsigned long long)address);
            t->unreadable = TRUE;
        }
    }
    if (!t->size)
    {
        static LONG warned;
        if (!InterlockedExchange(&warned, 1))
            rebox_log("d3d12_xs: placed texture at %#llx: no layout for it; it stays as created",
                         (unsigned long long)address);
        free(t);
        return;
    }
    if (FAILED(ID3D12Resource_QueryInterface(resource, &IID_ID3DDestructionNotifier, (void **)&notifier)))
    {
        free(t);
        return;
    }
    ID3DDestructionNotifier_RegisterDestructionCallback(notifier, destroyed, t, &t->callback);
    ID3DDestructionNotifier_Release(notifier);
    /* A render target, depth or UAV texture is the GPU's (as xcl's GPU-owned textures): never
     * filled from title memory, which would overwrite what the GPU drew into it. Except a small
     * single-sample colour target: a title's glyph caches are 256x256 R8 render targets the CPU
     * fills as text is laid out (xcl, ivall/xcl #211), and without them most letters were missing.
     * Those are filled when their memory changes, which a GPU draw into them doesn't do. */
    t->gpu_written = (console_desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
            | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) != 0;
    if (t->gpu_written && small_cpu_filled_target(console_desc))
        t->gpu_written = FALSE, t->target = TRUE;
    /* A console-only format made a desktop one (proxy.c's pc_format) has another layout than the
     * console memory's: it isn't uploaded (detiling it overran the upload buffer). */
    if (desc->Format != console_desc->Format)
    {
        static LONG logged;
        if (InterlockedIncrement(&logged) <= 4)
            rebox_log("d3d12_xs: placed texture at %#llx in console format %u (desktop %u): not filled from title memory",
                      (unsigned long long)address, console_desc->Format, desc->Format);
        t->gpu_written = TRUE;
    }
    t->dirty = !t->unreadable && !t->gpu_written;
    AcquireSRWLockExclusive(&lock);
    t->next = textures;
    textures = t;
    index_add(t);
    refresh_watched();
    set_watched(t);
    /* Armed now, if its memory is in physical-page views; the first upload arms it again. */
    if (!t->unreadable)
        t->tracked = track(t);
    if (t->dirty)
        InterlockedIncrement(&dirty_count);
    ReleaseSRWLockExclusive(&lock);
}

/* A range the title's memory changed in, by CPU or GPU address. */
void WINAPI ReboxTitleMemoryWritten(UINT64 address, UINT64 size)
{
    if (!textures || !size)
        return;
    AcquireSRWLockExclusive(&lock);
    for (UINT64 page = address >> 16; page <= (address + size - 1) >> 16; page++)
        for (struct page_node *n = by_page[page_bucket(page)]; n; n = n->next)
        {
            struct title_texture *t = n->t;
            if (n->page == page && !t->dirty && !t->unreadable
                    && ((address < t->address + t->size && t->address < address + size)
                        || (address < (UINT64)(ULONG_PTR)t->cpu + t->size && (UINT64)(ULONG_PTR)t->cpu < address + size)))
            {
                t->dirty = TRUE;
                InterlockedIncrement(&dirty_count);
            }
        }
    ReleaseSRWLockExclusive(&lock);
}

/* Uploads one texture through an upload buffer: detile each subresource into its copyable
 * footprint, then copy. Returns the buffer, which must live until the copy has run. */
static ID3D12Resource *record_upload(ID3D12Device *device, ID3D12GraphicsCommandList *list, struct title_texture *t)
{
    D3D12_RESOURCE_DESC desc;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT *fp;
    UINT *rows;
    D3D12_HEAP_PROPERTIES upload = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 0, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
    D3D12_RESOURCE_BARRIER barrier = { D3D12_RESOURCE_BARRIER_TYPE_TRANSITION };
    ID3D12Resource *buffer = NULL;
    UINT64 total = 0;
    BYTE *mapped;
    BOOL ok = TRUE;

    desc = ID3D12Resource_GetDesc(t->resource);
    fp = calloc(t->subresources, sizeof(*fp));
    rows = calloc(t->subresources, sizeof(*rows));
    if (!fp || !rows)
    {
        free(fp);
        free(rows);
        return NULL;
    }
    ID3D12Device_GetCopyableFootprints(device, &desc, 0, t->subresources, 0, fp, rows, NULL, &total);
    /* Slack past the last footprint: detiling writes whole tiles, padded rows included, so the
     * last subresource's can run past its footprint's end (it wrote past the buffer and crashed). */
    bd.Width = total + (256u << 10);
    if (FAILED(ID3D12Device_CreateCommittedResource(device, &upload, D3D12_HEAP_FLAG_NONE, &bd,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource,
                                                    (void **)&buffer))
            || FAILED(ID3D12Resource_Map(buffer, 0, NULL, (void **)&mapped)))
    {
        if (buffer)
            ID3D12Resource_Release(buffer);
        free(fp);
        free(rows);
        return NULL;
    }
    /* The console surface has to fit the desktop resource's footprints: detiling writes each of
     * its mips' rows at the footprint's pitch, unchecked (a mismatch overran the upload buffer). */
    if (t->subresources > (UINT)desc.MipLevels * (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1 : desc.DepthOrArraySize))
        ok = FALSE;
    for (UINT i = 0; i < t->subresources && ok; i++)
    {
        UINT mip = i % t->surface.mips;
        UINT w = max(1u, t->surface.width >> mip), h = max(1u, t->surface.height >> mip);
        if (fp[i].Footprint.Width < w || fp[i].Footprint.Height < h)
            ok = FALSE;
    }
    if (!ok)
    {
        static LONG logged;
        if (InterlockedIncrement(&logged) <= 4)
            rebox_log("d3d12_xs: placed texture at %#llx (%ux%u, %u mips): its console layout doesn't fit the "
                      "resource (%llux%u, %u mips, format %u); not filled from title memory",
                      (unsigned long long)t->address, t->surface.width, t->surface.height, t->surface.mips,
                      (unsigned long long)desc.Width, desc.Height, desc.MipLevels, desc.Format);
        t->gpu_written = TRUE;
    }
    {
        /* Copied out first: the title can unmap the texture's pages while this reads them. */
        BYTE *source = ok ? malloc(t->size) : NULL;
        if (ok && (!source || !read_title(t->cpu, source, t->size)))
            ok = FALSE;
        if (ok && t->tracked)
            t->fingerprint = fingerprint_bytes(t, source);
        for (UINT i = 0; i < t->subresources && ok; i++)
        {
            UINT mip = i % t->surface.mips, layer = i / t->surface.mips;
            /* rows: rows of pixels, or of 4x4 blocks for BC formats; a 3D slice is that many rows. */
            ok = detile_to_linear(&t->surface, source, mip, layer, mapped + fp[i].Offset,
                                  fp[i].Footprint.RowPitch, (UINT64)fp[i].Footprint.RowPitch * rows[i]);
        }
        free(source);
    }
    ID3D12Resource_Unmap(buffer, 0, NULL);
    if (!ok)
    {
        static LONG warned;
        if (!InterlockedExchange(&warned, 1))
            rebox_log("d3d12_xs: detiling a placed texture failed (format %u, %ux%u)", t->surface.format,
                         t->surface.width, t->surface.height);
        ID3D12Resource_Release(buffer);
        free(fp);
        free(rows);
        return NULL;
    }
    barrier.Transition.pResource = t->resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = t->state;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    if (t->state != D3D12_RESOURCE_STATE_COPY_DEST)
        ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &barrier);
    for (UINT i = 0; i < t->subresources; i++)
    {
        D3D12_TEXTURE_COPY_LOCATION dst = { t->resource, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, { .SubresourceIndex = i } };
        D3D12_TEXTURE_COPY_LOCATION src = { buffer, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, { .PlacedFootprint = fp[i] } };
        ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
    }
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = t->state;
    if (t->state != D3D12_RESOURCE_STATE_COPY_DEST)
        ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &barrier);
    free(fp);
    free(rows);
    return buffer;
}

/* Uploads in flight: their buffers, list and allocator live until the fence says the GPU is done.
 * Nothing waits on the CPU; a queue other than the one an upload ran on waits for it on the GPU
 * before running the title's lists, so a texture is never read before it's filled. */
struct batch
{
    ID3D12Fence *fence;
    ID3D12CommandAllocator *alloc;
    ID3D12GraphicsCommandList *list;
    ID3D12CommandQueue *queue; /* not a reference: only compared */
    ID3D12Resource *buffers[64];
    UINT count;
    UINT64 bytes;
};

static struct batch *batches[32];
static SRWLOCK batch_lock = SRWLOCK_INIT;

static void batch_free(struct batch *b)
{
    while (b->count)
        ID3D12Resource_Release(b->buffers[--b->count]);
    if (b->fence) ID3D12Fence_Release(b->fence);
    if (b->list) ID3D12GraphicsCommandList_Release(b->list);
    if (b->alloc) ID3D12CommandAllocator_Release(b->alloc);
    free(b);
}

/* Frees finished batches and makes queue wait for unfinished ones another queue ran. Returns a
 * free slot, or -1 when all are busy (the dirty textures then wait for a later submission). */
static int batches_reclaim(ID3D12CommandQueue *queue)
{
    int free_slot = -1;

    AcquireSRWLockExclusive(&batch_lock);
    for (int i = 0; i < (int)ARRAYSIZE(batches); i++)
    {
        struct batch *b = batches[i];

        if (b && ID3D12Fence_GetCompletedValue(b->fence) >= 1)
        {
            batch_free(b);
            batches[i] = b = NULL;
        }
        if (!b && free_slot < 0)
            free_slot = i;
        else if (b && b->queue != queue)
            ID3D12CommandQueue_Wait(queue, b->fence, 1);
    }
    ReleaseSRWLockExclusive(&batch_lock);
    return free_slot;
}

/* REBOX_DUMP_TITLE_TEXTURE=<address>: with REBOX_DUMP_FRAMES, the title memory of the texture
 * there is written at each round of the state log as DIR\texture_<address>_<round>.bin (its bytes
 * as laid out; the log line gives the size and mode), to see what the title put in it. */
static void dump_title_texture(const struct title_texture *t, UINT round)
{
    static UINT64 address = ~0ull;
    const char *dir = getenv("REBOX_DUMP_FRAMES");
    char path[MAX_PATH];
    FILE *f;

    if (address == ~0ull)
    {
        char *v = rebox_getenv("REBOX_DUMP_TITLE_TEXTURE");
        address = v ? _strtoui64(v, NULL, 16) : 0;
        free(v);
    }
    if (!address || t->address != address || !dir || t->unreadable || !t->cpu)
        return;
    snprintf(path, sizeof(path), "%s\\texture_%llx_%u.bin", dir, (unsigned long long)address, round);
    if (!(f = fopen(path, "wb")))
    {
        rebox_log("d3d12_xs: cannot write %s", path);
        return;
    }
    fwrite(t->cpu, 1, t->size, f);
    fclose(f);
}

/* Screen-sized textures kept in step (a video's planes) and A8 ones (glyph atlases), every 600 submissions, eight times: how
 * they are laid out, how often uploaded, and whether their memory holds anything. Called with the
 * lock held. */
static void log_large(void)
{
    static UINT rounds;
    if (submissions % 600 || rounds >= 8)
        return;
    rounds++;
    for (struct title_texture *t = textures; t; t = t->next)
    {
        UINT64 nonzero = 0, sampled = 0;
        if ((t->surface.width < 1920 || t->surface.height < 1080) && t->surface.format != DXGI_FORMAT_A8_UNORM)
            continue;
        if (!t->unreadable && t->cpu && still_readable(t))
            for (UINT64 at = 0, step = t->size <= (4u << 20) ? 1 : 4093; at < t->size; at += step, sampled++)
                nonzero += t->cpu[at] != 0; /* up to 4 MiB (a glyph atlas) every byte */
        rebox_log("d3d12_xs: large title texture at %#llx: %ux%u format %u, mode %u, %u uploads%s%s%s%s, "
                  "%llu of %llu sampled bytes nonzero", (unsigned long long)t->address, t->surface.width,
                  t->surface.height, t->surface.format, t->surface.mode, t->uploads, t->dirty ? ", dirty" : "",
                  t->unreadable ? ", unreadable" : "", t->gpu_written ? ", GPU's" : "",
                  t->watched ? ", watched" : t->tracked ? ", tracked" : t->chunk ? ", in a GPU memory chunk"
                  : in_chunk((UINT64)(ULONG_PTR)t->cpu, (UINT64)(ULONG_PTR)t->cpu + t->size) ? ", GPU memory in place"
                  : ", fingerprinted",
                  (unsigned long long)nonzero, (unsigned long long)sampled);
        dump_title_texture(t, rounds);
    }
}

/* Where title_textures_upload's time goes, logged every 1024 submissions with uploads to make or
 * changes to look for: the change scan, the batch's objects, recording uploads (buffer, copy-out,
 * detiling), and how many bytes. */
static struct { LONG64 scan, objects, record, submit, bytes; LONG calls, uploads; } upload_stats;

static LONG64 stat_ticks(void)
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}

static void upload_stats_log(void)
{
    LARGE_INTEGER f;
    double ms;
    LONG n = InterlockedIncrement(&upload_stats.calls);

    if (n % 1024)
        return;
    QueryPerformanceFrequency(&f);
    ms = 1000.0 / (double)f.QuadPart / 1024;
    rebox_log("d3d12_xs: title textures, ms per submission over 1024: change scan %.2f, batch objects %.2f, "
              "recording %.2f, submitting %.2f; %ld uploads, %.1f MiB", upload_stats.scan * ms,
              upload_stats.objects * ms, upload_stats.record * ms, upload_stats.submit * ms, upload_stats.uploads,
              upload_stats.bytes / 1048576.0);
    memset(&upload_stats, 0, sizeof(upload_stats));
    rebox_log("d3d12_xs: presumed-hot textures' last full turns took %u submissions (1-byte texels) and %u (others)",
              hot_turn[HOT_BYTE], hot_turn[HOT_OTHER]);
    if (tracked_dirty)
    {
        UINT count = 0, tracked = 0;
        AcquireSRWLockShared(&lock);
        for (struct title_texture *t = textures; t; t = t->next, count++)
            tracked += t->tracked;
        ReleaseSRWLockShared(&lock);
        rebox_log("d3d12_xs: write tracking over those: %u of %u placed textures tracked; arming %.3f ms per "
                  "submission; %lld dirty ranges made %lld dirty; safety net %lld checks, %lld changes found", tracked,
                  count, track_stats.arm_ticks * ms, track_stats.ranges, track_stats.textures,
                  track_stats.safety_checks, track_stats.safety_changes);
        memset(&track_stats, 0, sizeof(track_stats));
    }
}

void title_textures_upload(ID3D12CommandQueue *queue)
{
    LONG64 t0, t1;
    struct title_texture *t;
    struct batch *b = NULL;
    ID3D12Device *device = NULL;
    D3D12_COMMAND_QUEUE_DESC qd;
    int slot;

    if (!textures)
        return;
    t0 = stat_ticks();
    slot = batches_reclaim(queue);
    AcquireSRWLockExclusive(&lock);
    submissions++;
    watch();
    log_large();
    ReleaseSRWLockExclusive(&lock);
    t1 = stat_ticks();
    upload_stats.scan += t1 - t0;
    upload_stats_log();
    if (!dirty_count || slot < 0)
        return;
    qd = ID3D12CommandQueue_GetDesc(queue);
    if (qd.Type != D3D12_COMMAND_LIST_TYPE_DIRECT && qd.Type != D3D12_COMMAND_LIST_TYPE_COMPUTE)
        return;
    if (FAILED(ID3D12CommandQueue_GetDevice(queue, &IID_ID3D12Device, (void **)&device)))
        return;
    if (!(b = calloc(1, sizeof(*b)))
            || FAILED(ID3D12Device_CreateCommandAllocator(device, qd.Type, &IID_ID3D12CommandAllocator, (void **)&b->alloc))
            || FAILED(ID3D12Device_CreateCommandList(device, 0, qd.Type, b->alloc, NULL, &IID_ID3D12GraphicsCommandList,
                                                     (void **)&b->list))
            || FAILED(ID3D12Device_CreateFence(device, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void **)&b->fence)))
        goto done;
    b->queue = queue;
    t0 = stat_ticks();
    upload_stats.objects += t0 - t1;

    /* Under the lock so a texture can't be destroyed while its upload is recorded; up to 64 a
     * submission, the rest the next time. */
    AcquireSRWLockExclusive(&lock);
    for (t = textures; t && b->count < ARRAYSIZE(b->buffers); t = t->next)
        if (t->dirty && still_readable(t))
        {
            t->dirty = FALSE;
            InterlockedDecrement(&dirty_count);
            /* Armed before it is read: a write from here on faults and makes it dirty again, one
             * before is in what is read. A texture no longer in views (remapped as a copy, made GPU
             * memory in place) is fingerprinted from now on; one back in views is tracked again. */
            if (tracking_enabled() && !t->watched && !t->chunk)
            {
                t->tracked = track(t);
                t->checked_at = GetTickCount64();
            }
            /* None for a watched texture: if it stops being one, the old one makes it upload again.
             * A tracked one's is taken from what the upload reads. */
            if (!t->watched && !t->tracked)
                t->fingerprint = fingerprint(t);
            /* One made without memory gets it first, mapped on this queue ahead of the copy. */
            if (!proxy_texture_memory(queue, t->resource))
                continue;
            if ((b->buffers[b->count] = record_upload(device, b->list, t)))
            {
                D3D12_RESOURCE_DESC bd = ID3D12Resource_GetDesc(b->buffers[b->count]);
                b->bytes += bd.Width;
                b->count++;
                t->uploads++;
                upload_stats.uploads++;
                upload_stats.bytes += t->size;
            }
        }
    ReleaseSRWLockExclusive(&lock);
    t1 = stat_ticks();
    upload_stats.record += t1 - t0;
    if (FAILED(ID3D12GraphicsCommandList_Close(b->list)) || !b->count)
        goto done;
    ID3D12CommandQueue_ExecuteCommandLists(queue, 1, (ID3D12CommandList **)&b->list);
    if (FAILED(ID3D12CommandQueue_Signal(queue, b->fence, 1)))
    {
        /* Can't tell when it's done: wait for the queue the slow way. */
        for (int i = 0; i < 1000 && ID3D12Fence_GetCompletedValue(b->fence) < 1; i++)
            Sleep(1);
        goto done;
    }
    {
        static LONG uploads;
        if (InterlockedIncrement(&uploads) <= 8)
            rebox_log("d3d12_xs: uploaded %u placed texture(s) from title memory", b->count);
    }
    upload_stats.submit += stat_ticks() - t1;
    /* Another queue may have taken the slot meanwhile: any free one will do. */
    AcquireSRWLockExclusive(&batch_lock);
    for (int i = 0; i < (int)ARRAYSIZE(batches) && b; i++)
        if (!batches[i])
            batches[i] = b, b = NULL;
    ReleaseSRWLockExclusive(&batch_lock);
    if (b)
        for (int i = 0; i < 1000 && ID3D12Fence_GetCompletedValue(b->fence) < 1; i++)
            Sleep(1);

done:
    if (b)
        batch_free(b);
    if (device) ID3D12Device_Release(device);
}

void title_textures_stats(struct title_texture_stats *s)
{
    memset(s, 0, sizeof(*s));
    AcquireSRWLockShared(&lock);
    for (struct title_texture *t = textures; t; t = t->next)
        s->textures++;
    ReleaseSRWLockShared(&lock);
    s->dirty = (UINT)dirty_count;
    AcquireSRWLockShared(&batch_lock);
    for (int i = 0; i < (int)ARRAYSIZE(batches); i++)
        if (batches[i])
        {
            s->uploads += batches[i]->count;
            s->upload_bytes += batches[i]->bytes;
        }
    ReleaseSRWLockShared(&batch_lock);
}

/* For tests and tools: the size of a surface in title memory, and tiling linear data into it. */
UINT64 WINAPI ReboxDetileSize(const struct detile_surface *s)
{
    return detile_size(s);
}

BOOL WINAPI ReboxTile(const struct detile_surface *s, void *surface, UINT mip, UINT layer, const void *linear,
                         UINT64 row_pitch, UINT64 slice_pitch)
{
    return detile_from_linear(s, surface, mip, layer, linear, row_pitch, slice_pitch);
}
