/* Compiles deferred console shaders with AnyPS5 at dispatch (docs/anyps5.md).
 *
 * vkd3d-proton (patch 0005) calls ReboxConsoleCompile at each dispatch of a deferred console
 * PSO with the user SGPRs it packed (descriptor tables at shadow-heap title addresses), the bound
 * heaps and where each descriptor kind lives among its bindless sets. Here:
 *
 * 1. AnyPS5 (rebox_anyps5.dll) compiles the code against the shadow heaps as guest memory,
 *    through its in-memory cache: the code is parsed once, and each new set of descriptors it
 *    reads gives a variant.
 * 2. Each guest binding's descriptors are found in the bound heap's shadow (ReboxShadowFind's
 *    lookup), giving heap slots; host buffers (flattened SRT, BDA page table, fault buffer, GDS)
 *    get GPU memory.
 * 3. sa5_finalize rebinds the module onto the bindless sets at those slots, with the host buffers
 *    as BDA pointers in the push block. Finalized modules are cached by a key over AnyPS5's
 *    pipeline identity and the slots; vkd3d-proton caches pipelines by the same key.
 *
 * Nothing is freed: modules and host buffers live for the process. A failing shader logs once
 * and its dispatches are skipped. REBOX_DUMP_SPIRV=DIR writes each finalized module there. */

#define COBJMACROS
#define WIDL_C_INLINE_WRAPPERS
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/rebox_log.h"
#include "../../anyps5/rebox_anyps5.h"
#include "fast.h"
#include "../../generated/standin_shaders.h"
#include "console_compile.h"
#include "console_fssm.h"
#include "console_desc.h"
#include "gpu_memory.h"
#include "proxy.h"
#include "shadow.h"
#include "title_textures.h"

#define TITLE_BASE 0x10000000ull
/* Shadow heaps and GPU memory chunks handed to AnyPS5 and mapped in its BDA page table. A world in
 * the target title passed 64 chunks; those left out made user-data pointers into them unreadable
 * (mesh stages refused by the SRT walk, compute reads of "inaccessible" memory). */
#define MAX_REGIONS 1024

/* A region list that came back full may have left regions out: logged once per list. */
static void note_regions_full(const char *what, UINT n, UINT max)
{
    static LONG logged[2];
    LONG *flag = &logged[what[0] == 'G'];
    if (n >= max && !InterlockedExchange(flag, 1))
        rebox_log("AnyPS5 regions: %s filled all %u entries; later ones are left out", what, max);
}
#define MAX_BINDINGS 64
/* Guest descriptors a stage binds, all bindings together (each element's heap slot goes in the draw
 * block). The target title's bindless compute and pixel stages index image tables of up to 23,725
 * T#s. */
#define MAX_ELEMENTS 32768
/* Graphics stages share the push block, 128 bytes each: AnyPS5's push data, then the addresses of
 * up to two host buffers whose contents change per draw (baked addresses made a module and a
 * pipeline per draw: a UE5 title's render thread spent its frames compiling). */
#define GRAPHICS_PUSH_DATA 112u
#define GRAPHICS_PUSHED_MAX SA5_DRAW_ADDRESSES /* per-draw host buffers: in the draw block */
#define COMPUTE_DRAW_BLOCK 128 /* where a dispatch's draw block address is in its push block */
/* A draw block (sa5_finalize_draw): its host buffers' addresses (2 words each), their lengths
 * (SA5_LENGTH_IN_DRAW_BLOCK, a word each), then its bindings' heap slots from DRAW_INDICES. */
#define DRAW_INDICES (3 * SA5_DRAW_ADDRESSES)
/* A mesh stage's push data ends earlier: AnyPS5's mesh path reads its draw parameters from push bytes
 * 104-127 (sa5_finalize_draw freezes them, so nothing is pushed there). */
#define MESH_PUSH_DATA 96u

enum role { ROLE_GDS = 3, ROLE_BDA_PAGETABLE = 4, ROLE_FAULT_BUFFER = 5, ROLE_FLATTENED_SRT = 6, ROLE_SHADER_DATA = 7,
            ROLE_IMAGE_TABLE = 8, ROLE_SAMPLER_TABLE = 9, ROLE_IMAGE_TABLE_MAP = 10, ROLE_IMAGE_STORAGE_TABLE = 11 };

/* Roles whose elements are guest descriptors bound like guest images and samplers: AnyPS5's
 * tables for T#s and S#s that a shader indexes at run time (from a table in memory). */
static BOOL guest_descriptor_role(UINT32 role)
{
    return role <= 2 || role == ROLE_IMAGE_TABLE || role == ROLE_SAMPLER_TABLE || role == ROLE_IMAGE_STORAGE_TABLE;
}

static struct
{
    INIT_ONCE once;
    int (*compile)(const struct sa5_compute_request *, struct sa5_result **, char *, size_t);
    int (*compile_vertex)(const struct sa5_vertex_request *, struct sa5_result **, char *, size_t);
    int (*compile_pixel)(const struct sa5_pixel_request *, struct sa5_result **, char *, size_t);
    int (*compile_mesh)(const struct sa5_mesh_request *, struct sa5_result **, char *, size_t);
    UINT32 (*vertex_input_count)(const struct sa5_result *);
    int (*vertex_input)(const struct sa5_result *, UINT32, UINT32 *, UINT32 *, UINT32 *);
    const UINT32 *(*spirv)(const struct sa5_result *, UINT32 *);
    UINT32 (*binding_count)(const struct sa5_result *);
    int (*binding)(const struct sa5_result *, UINT32, struct sa5_binding *);
    const void *(*push_constants)(const struct sa5_result *, UINT32 *);
    UINT32 (*subgroup_size)(const struct sa5_result *);
    void (*identity)(const struct sa5_result *, UINT64 *);
    int (*analyse)(const UINT32 *, UINT32, UINT32, struct sa5_layout *, struct sa5_slot *, UINT32, char *, size_t);
    void (*result_free)(struct sa5_result *);
    int (*finalize)(const struct sa5_result *, const struct sa5_remap *, UINT32, const struct sa5_host_buffer *,
            UINT32, UINT32 **, UINT32 *, char *, size_t);
    int (*finalize_draw)(const struct sa5_result *, const struct sa5_remap *, UINT32, const struct sa5_host_buffer *,
            UINT32, UINT32, UINT32 **, UINT32 *, char *, size_t);
    UINT32 (*fault_buffer_bytes)(void);
    UINT32 (*gds_bytes)(void);
    int (*rect_list_stages)(const struct sa5_result *, const struct sa5_result *, struct sa5_result **,
            struct sa5_result **, char *, size_t);
} sa5;

/* AnyPS5's settings, made before anything loads rebox_anyps5.dll (it reads some of them once,
 * into statics): from DllMain. */
static void set_anyps5_environment(void)
{
    /* Xbox Series vertex shaders take the vertex ID in v0 and the instance ID in v3 (the legacy
     * layout; AnyPS5 patch 0008), not PS5's v5 and v8. */
    if (!getenv("ANYPS5_LEGACY_VERTEX_INPUTS"))
        _putenv("ANYPS5_LEGACY_VERTEX_INPUTS=1");
    const char *local = getenv("LOCALAPPDATA");

    /* AnyPS5 keeps compiled variants on disk, by default in shader_cache next to the exe: the
     * game's directory. Unless ANYPS5_SHADER_CACHE_DIR or ANYPS5_NO_SHADER_CACHE says otherwise,
     * use %LOCALAPPDATA%\rebox\anyps5-cache. Both DLLs use msvcrt, so _putenv reaches
     * AnyPS5's getenv. */
    if (!getenv("ANYPS5_SHADER_CACHE_DIR") && !getenv("ANYPS5_NO_SHADER_CACHE") && local)
    {
        char dir[MAX_PATH], setting[MAX_PATH + 32];
        snprintf(dir, sizeof(dir), "%s\\rebox", local);
        CreateDirectoryA(dir, NULL);
        snprintf(dir, sizeof(dir), "%s\\rebox\\anyps5-cache", local);
        CreateDirectoryA(dir, NULL);
        snprintf(setting, sizeof(setting), "ANYPS5_SHADER_CACHE_DIR=%s", dir);
        _putenv(setting);
        rebox_log("d3d12_xs: AnyPS5 shader cache in %s", dir);
    }
    /* patches/anyps5/0004: when a descriptor or table pointer is a phi AnyPS5 can't evaluate (a
     * path the dispatch never takes clobbers it), speculate the evaluable value and check it at
     * run time. On by default; REBOX_ANYPS5_SPECULATE=0 turns it off. The host doesn't read
     * the check's fault buffer back yet, so a wrong guess goes unnoticed. */
    {
        const char *speculate = getenv("REBOX_ANYPS5_SPECULATE");
        if (!(speculate && speculate[0] == '0') && !getenv("APS5_SPECULATE_DESCRIPTORS"))
            _putenv("APS5_SPECULATE_DESCRIPTORS=1");
    }
    /* patches/anyps5/0012: every translated loop leaves after 2^20 iterations per entry, so a loop
     * whose exit depends on data we got wrong can't hang the GPU (a device lost after the kernel's
     * hard recovery). A loop stopped this way is a "loop limit" fault (poll_faults logs it).
     * APS5_LOOP_GUARD=0 turns it off; another value sets the limit. */
    if (!getenv("APS5_LOOP_GUARD"))
        _putenv("APS5_LOOP_GUARD=0x100000");
    /* patches/anyps5/0020: a graphics stage's T# or V# that isn't one when the draw is recorded is
     * bound as a null descriptor instead of the stage being refused. The target title fills a
     * draw's tables after recording it, so they still hold an earlier draw's words then (V#s where
     * T#s go, constants): 148 of its pixel stages were refused so for the run. Such a slot is
     * filled at submission as an empty one is (console_resolve_late). REBOX_ANYPS5_NULL_INVALID=0
     * turns it off; compute keeps refusing (its dispatches have no late slots). */
    {
        const char *null_invalid = getenv("REBOX_ANYPS5_NULL_INVALID");
        if (!(null_invalid && null_invalid[0] == '0') && !getenv("APS5_NULL_INVALID_DESCRIPTORS"))
            _putenv("APS5_NULL_INVALID_DESCRIPTORS=graphics");
    }
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, void *reserved)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        DisableThreadLibraryCalls(instance);
        set_anyps5_environment();
    }
    return TRUE;
}

static BOOL CALLBACK load_sa5(INIT_ONCE *once, void *param, void **context)
{
    HMODULE m;

    m = LoadLibraryA("rebox_anyps5.dll");

    if (!m)
    {
        rebox_log("d3d12_xs: rebox_anyps5.dll not found; console shaders can't be compiled");
        return TRUE;
    }
#define GET(field, name) sa5.field = (void *)GetProcAddress(m, name)
    GET(compile, "sa5_compile_compute");
    GET(compile_vertex, "sa5_compile_vertex");
    GET(compile_pixel, "sa5_compile_pixel");
    GET(compile_mesh, "sa5_compile_mesh");
    GET(vertex_input_count, "sa5_result_vertex_input_count");
    GET(vertex_input, "sa5_result_vertex_input");
    GET(spirv, "sa5_result_spirv");
    GET(binding_count, "sa5_result_binding_count");
    GET(binding, "sa5_result_binding");
    GET(push_constants, "sa5_result_push_constants");
    GET(subgroup_size, "sa5_result_subgroup_size");
    GET(identity, "sa5_result_identity");
    GET(analyse, "sa5_analyse");
    GET(result_free, "sa5_result_free");
    GET(finalize, "sa5_finalize");
    GET(finalize_draw, "sa5_finalize_draw");
    GET(fault_buffer_bytes, "sa5_fault_buffer_bytes");
    GET(gds_bytes, "sa5_gds_bytes");
    GET(rect_list_stages, "sa5_rect_list_stages");
#undef GET
    if (!sa5.finalize || !sa5.identity || !sa5.gds_bytes)
    {
        rebox_log("d3d12_xs: rebox_anyps5.dll is too old (no sa5_finalize)");
        sa5.compile = NULL;
    }
    return TRUE;
}

/* 64-bit FNV-1a, seeded. */
static UINT64 hash_bytes(UINT64 h, const void *data, SIZE_T size)
{
    const BYTE *p = data;
    for (SIZE_T i = 0; i < size; i++)
        h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

/* As hash_bytes, eight bytes a step: for keys that stay in the process (not the stage ids the log
 * and REBOX_WATCH_STAGE name), over whole shaders at every draw and dispatch. */
static UINT64 hash_words(UINT64 h, const void *data, SIZE_T size)
{
    const BYTE *p = data;
    SIZE_T i = 0;
    for (; i + 8 <= size; i += 8)
    {
        UINT64 w;
        memcpy(&w, p + i, sizeof(w));
        h = (h ^ w) * 0x100000001b3ull;
        h ^= h >> 29;
    }
    return hash_bytes(h, p + i, size - i);
}

/* ------------------------------------------------------------ late T#s */

/* T#s a title writes after recording the draw that reads them (the target title fills a per-draw
 * descriptor table between recording and submitting; the console GPU reads it when it runs). A
 * graphics stage's heap slots are in its draw block, which the GPU reads: such a texture's slot
 * is written at submission instead (console_resolve_late), from the descriptor at its address
 * then. Until resolved, it holds the dummy texture. */
struct late_texture { UINT32 *slot; UINT64 source; SIZE_T heap; UINT tries; UINT64 stage; };
static BOOL guest_buffer_mappable(UINT64 *base, UINT64 *bytes);

/* A guest buffer table entry ({address, length in dwords, 0, 0}) for a V#: the module reads the
 * buffer at that address directly, in bounds of that length. One in a GPU-only view of title
 * memory is its CPU address, which must be GPU memory; one that can't be made so is read as empty
 * (a V# into such memory faulted the GPU). So is one whose base is 0 or in the first 64 KiB, which
 * nothing maps: AnyPS5 takes a V# with no base as absent, but the table has only the length to say
 * so, and a V# with records and no base (its address not written yet, or not a V#) made every load
 * through it a scalar read of address 0, a GPU page fault (the first indirect mesh draws). */
void WINAPI ReboxGuestBufferEntry(const UINT32 w[4], UINT32 entry[4])
{
    UINT64 base = w[0] | (UINT64)(w[1] & 0xffff) << 32;
    UINT32 stride = (w[1] >> 16) & 0x3fff;
    UINT64 bytes = stride ? (UINT64)w[2] * stride : w[2];

    if (base < 0x10000)
    {
        static LONG logged;
        if (bytes && InterlockedIncrement(&logged) <= 8)
            rebox_log("d3d12_xs: V# %08x %08x %08x %08x has %llu bytes at %#llx: read as empty", w[0], w[1], w[2],
                      w[3], (unsigned long long)bytes, (unsigned long long)base);
        bytes = 0;
    }
    else if (!guest_buffer_mappable(&base, &bytes))
        bytes = 0;
    entry[0] = (UINT32)base;
    entry[1] = (UINT32)(base >> 32);
    entry[2] = (UINT32)min(bytes / 4, 0xffffffffull);
    entry[3] = 0;
}
static struct late_texture late[4096];
static UINT late_count;
/* V#s the same: a guest buffer table entry ({address, length in dwords}) whose V# was empty when the
 * stage was set up (a video shader's constants, written with its T#s), filled at submission from
 * the V# at its address then. Until then it reads as an empty buffer (zeros). */
struct late_buffer { UINT32 *entry; UINT64 source; UINT tries; UINT64 stage; };
static struct late_buffer late_buffers[1024];
static UINT late_buffer_count;
static SRWLOCK late_lock = SRWLOCK_INIT;

void console_resolve_late(void)
{
    BOOL WINAPI ReboxShadowRead(UINT64 address, void *out, SIZE_T size);
    UINT kept = 0, resolved = 0;
    static LONG logged;

    if (!late_count && !late_buffer_count)
        return;
    AcquireSRWLockExclusive(&late_lock);
    {
        UINT kept_buffers = 0;
        for (UINT i = 0; i < late_buffer_count; i++)
        {
            struct late_buffer *t = &late_buffers[i];
            UINT32 w[4] = { 0 };
            if (ReboxShadowRead(t->source, w, sizeof(w)) && (w[0] || w[1]))
            {
                UINT64 base, bytes;
                static LONG shown;
                ReboxGuestBufferEntry(w, t->entry);
                base = t->entry[0] | (UINT64)t->entry[1] << 32;
                bytes = (UINT64)t->entry[2] * 4;
                if ((console_stage_watched(t->stage) && InterlockedIncrement(&shown) <= 16)
                        || InterlockedIncrement(&shown) <= 4)
                    rebox_log("d3d12_xs: late V# %08x %08x %08x %08x of stage %016llx: buffer at %#llx, %llu bytes",
                              w[0], w[1], w[2], w[3], (unsigned long long)t->stage, (unsigned long long)base,
                              (unsigned long long)bytes);
            }
            else if (++t->tries < 8)
                late_buffers[kept_buffers++] = *t;
        }
        late_buffer_count = kept_buffers;
    }
    for (UINT i = 0; i < late_count; i++)
    {
        struct late_texture *t = &late[i];
        UINT32 w[8] = { 0 };
        UINT index;
        BOOL done = FALSE;
        if (ReboxShadowRead(t->source, w, sizeof(w)) && (w[0] || w[1] || w[2] || w[3]))
        {
            ID3D12Resource *res;
            if ((shadow_find_index(t->heap, w, 8, FALSE, TRUE, &index)
                        && shadow_slot_view(t->heap, index) != SHADOW_VIEW_NONE)
                    || ((res = console_texture_of(w)) && shadow_reserve_srv(t->heap, res, &index)))
            {
                if (shadow_slot_view(t->heap, index) == SHADOW_VIEW_UAV_TEXTURE)
                    shadow_reserve_srv_for_uav(t->heap, index, &index);
                {
                    static LONG shown, watched_shown;
                    ID3D12Resource *r = console_texture_of(w);
                    LONG wn = console_stage_watched(t->stage) ? InterlockedIncrement(&watched_shown) : 0;
                    /* A watched stage's: the first 32, then every 256th, and whenever its texture
                     * changes (a scene can switch what a final pass reads). */
                    static ID3D12Resource *watched_last;
                    if ((wn && (wn <= 32 || !(wn % 256)
                                || InterlockedExchangePointer((void **)&watched_last, r) != r))
                            || (!wn && InterlockedIncrement(&shown) <= 8))
                    {
                        D3D12_RESOURCE_DESC rd = { 0 };
                        UINT64 at = 0;
                        UINT uploads = 0;
                        BOOL title = r && title_texture_info(r, &at, &uploads);
                        if (r)
                            rd = ID3D12Resource_GetDesc(r);
                        rebox_log("d3d12_xs: late T# %08x %08x %08x %08x: heap slot %u, texture %p %llux%u format %u "
                                  "flags %#x, %s %#llx (uploaded %u times)", w[0], w[1], w[2], w[3], index, (void *)r,
                                  (unsigned long long)rd.Width, rd.Height, rd.Format, (UINT)rd.Flags,
                                  title ? "from title memory at" : "GPU-made or API-filled;", (unsigned long long)at,
                                  uploads);
                    }
                }
                *t->slot = index;
                resolved++;
                done = TRUE;
            }
        }
        if (!done)
        {
            static LONG unresolved;
            if (t->tries == 0 && (console_stage_watched(t->stage) || InterlockedIncrement(&unresolved) <= 4))
                rebox_log("d3d12_xs: late T# at %#llx: %08x %08x %08x %08x %08x %08x %08x %08x, no view for it yet",
                          (unsigned long long)t->source, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
        }
        if (!done && ++t->tries < 8)
            late[kept++] = *t;
    }
    late_count = kept;
    ReleaseSRWLockExclusive(&late_lock);
    if (resolved && InterlockedIncrement(&logged) <= 8)
        rebox_log("d3d12_xs: %u texture(s) the title wrote after recording their draws bound at submission", resolved);
}

/* ------------------------------------------------------------ host buffers */

static SRWLOCK buffers_lock = SRWLOCK_INIT;
static struct host_buffer { UINT64 hash, gpu; SIZE_T size; const void *cpu; struct host_buffer *next; } *buffers[256];

/* GPU memory for contents that change with every draw or dispatch (guest buffer tables, the
 * flattened SRT, shader data, image table maps: they hold the title's per-draw constant buffer
 * addresses and table pointers): a ring, bump-allocated and wrapped. Sharing by content never hit
 * for those, so each draw allocated GPU memory it never freed, about 10 ms each.
 *
 * A wrap reuses the ring only once the GPU has finished every submission made before
 * DRAW_RING_SUBMIT_MS after the lap began (proxy_gpu_finished_before; a block is written when its
 * draw is recorded and read when the list runs, normally well within that). Otherwise a fresh ring
 * takes over and the old one is freed once the GPU is past that much after its last block: a draw
 * block overwritten under a queued draw gave its stage another draw's guest buffer table, SRT and
 * heap slots, wild scalar reads. */
#define DRAW_RING (128u << 20)
#define DRAW_RING_SUBMIT_MS 2000
#define DRAW_RINGS_RETIRED 8

static SRWLOCK ring_lock = SRWLOCK_INIT;
static BYTE *ring_cpu;
static UINT64 ring_gpu, ring_head;
static LONG64 ring_lap_start; /* QueryPerformanceCounter time of the current lap's first block */
static struct { BYTE *cpu; LONG64 last; } rings_retired[DRAW_RINGS_RETIRED];
static UINT rings_retired_count;
static LONG64 (*ring_gpu_finished_before)(void) = proxy_gpu_finished_before;

/* For tests: the GPU progress the ring asks for, instead of the proxies' (NULL restores it). */
void WINAPI ReboxConsoleRingProgress(LONG64 (*finished_before)(void))
{
    AcquireSRWLockExclusive(&ring_lock);
    ring_gpu_finished_before = finished_before ? finished_before : proxy_gpu_finished_before;
    ReleaseSRWLockExclusive(&ring_lock);
}

static LONG64 qpc_now(void)
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return now.QuadPart;
}

static LONG64 qpc_ms(LONG64 ms)
{
    static LONG64 frequency;
    if (!frequency)
    {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        frequency = f.QuadPart;
    }
    return frequency * ms / 1000;
}

/* Under ring_lock: retired rings the GPU is past go. */
static void rings_reap(LONG64 finished_before)
{
    UINT kept = 0;
    for (UINT i = 0; i < rings_retired_count; i++)
    {
        if (rings_retired[i].last + qpc_ms(DRAW_RING_SUBMIT_MS) < finished_before)
            ReboxGpuFree(rings_retired[i].cpu);
        else
            rings_retired[kept++] = rings_retired[i];
    }
    rings_retired_count = kept;
}

static UINT64 draw_buffer(const void *data, SIZE_T size, BYTE **cpu)
{
    SIZE_T aligned = (size + 255) & ~(SIZE_T)255;
    UINT64 at;

    if (!size || aligned > DRAW_RING)
        return 0;
    AcquireSRWLockExclusive(&ring_lock);
    if (!ring_cpu && FAILED(ReboxGpuAlloc(DRAW_RING, 256, (void **)&ring_cpu, &ring_gpu)))
        ring_cpu = NULL;
    if (!ring_cpu)
    {
        ReleaseSRWLockExclusive(&ring_lock);
        return 0;
    }
    if (!ring_lap_start)
        ring_lap_start = qpc_now();
    if (ring_head + aligned > DRAW_RING)
    {
        static LONG laps, fresh;
        LONG64 now = qpc_now(), finished_before = ring_gpu_finished_before();
        BYTE *new_cpu;
        UINT64 new_gpu;
        if (rings_retired_count)
            rings_reap(finished_before);
        if (finished_before <= ring_lap_start + qpc_ms(DRAW_RING_SUBMIT_MS)
                && rings_retired_count < DRAW_RINGS_RETIRED
                && SUCCEEDED(ReboxGpuAlloc(DRAW_RING, 256, (void **)&new_cpu, &new_gpu)))
        {
            /* The GPU may still read this lap's blocks: a fresh ring. */
            if (InterlockedIncrement(&fresh) <= 8)
                rebox_log("d3d12_xs: the per-draw ring filled in %lld ms with the GPU not yet past its first "
                          "blocks: a fresh ring (%u waiting to be freed)", (now - ring_lap_start) / qpc_ms(1),
                          rings_retired_count + 1);
            rings_retired[rings_retired_count].cpu = ring_cpu;
            rings_retired[rings_retired_count++].last = now;
            ring_cpu = new_cpu;
            ring_gpu = new_gpu;
        }
        else if (finished_before <= ring_lap_start + qpc_ms(DRAW_RING_SUBMIT_MS))
        {
            static LONG logged;
            if (InterlockedIncrement(&logged) <= 8)
                rebox_log("d3d12_xs: the per-draw ring wrapped after %lld ms with the GPU not yet past its first "
                          "blocks, and no fresh ring could be had: reused", (now - ring_lap_start) / qpc_ms(1));
        }
        else if (InterlockedIncrement(&laps) <= 4)
            rebox_log("d3d12_xs: the per-draw ring wrapped after %lld ms", (now - ring_lap_start) / qpc_ms(1));
        ring_lap_start = now;
        ring_head = 0;
    }
    else if (rings_retired_count)
    {
        static UINT blocks;
        if (!(++blocks & 4095))
            rings_reap(ring_gpu_finished_before());
    }
    at = ring_head;
    ring_head += aligned;
    memcpy(ring_cpu + at, data, size);
    if (cpu)
        *cpu = ring_cpu + at;
    ReleaseSRWLockExclusive(&ring_lock);
    return ring_gpu + at;
}

UINT64 console_draw_buffer(const void *data, SIZE_T size)
{
    return draw_buffer(data, size, NULL);
}

/* For tests: a block of the per-draw ring holding size bytes of data; its CPU address in *cpu. */
UINT64 WINAPI ReboxConsoleRingBlock(const void *data, SIZE_T size, void **cpu)
{
    return draw_buffer(data, size, (BYTE **)cpu);
}

/* For tests: size bytes of the per-draw ring at a GPU address draw_buffer gave (a draw block). */
BOOL WINAPI ReboxConsoleDrawBlock(UINT64 va, void *out, SIZE_T size)
{
    BOOL ok;
    AcquireSRWLockShared(&ring_lock);
    ok = ring_cpu && va >= ring_gpu && size <= DRAW_RING && va - ring_gpu <= DRAW_RING - size;
    if (ok)
        memcpy(out, ring_cpu + (va - ring_gpu), size);
    ReleaseSRWLockShared(&ring_lock);
    return ok;
}

/* GPU memory holding these bytes; equal contents share it. */
static UINT64 host_buffer(const void *data, SIZE_T size)
{
    UINT64 h = hash_bytes(0xcbf29ce484222325ull, data, size), gpu = 0;
    struct host_buffer **bucket = &buffers[h & 255], *b;
    void *cpu;

    AcquireSRWLockExclusive(&buffers_lock);
    for (b = *bucket; b; b = b->next)
        if (b->hash == h && b->size == size && !memcmp(b->cpu, data, size))
            break;
    if (b)
        gpu = b->gpu;
    else if (SUCCEEDED(ReboxGpuAlloc(size, 256, &cpu, &gpu)) && (b = malloc(sizeof(*b))))
    {
        memcpy(cpu, data, size);
        *b = (struct host_buffer){ h, gpu, size, cpu, *bucket };
        *bucket = b;
    }
    ReleaseSRWLockExclusive(&buffers_lock);
    return gpu;
}

/* AnyPS5's BDA page table (BdaAbi.hpp): a header, then ranges sorted by guest address. GPU memory
 * is identity-mapped; shadow heaps map their title addresses to their GPU copies. */
struct page_range { UINT64 begin, end, device; UINT32 permissions, reserved; };

static int compare_ranges(const void *a, const void *b)
{
    const struct page_range *x = a, *y = b;
    return x->begin < y->begin ? -1 : x->begin > y->begin;
}

/* Returns the table's GPU address and its size in *bytes (AnyPS5 checks that its length in words
 * is exactly 4 + 8 * count, through OpArrayLength). The table has one address and a fixed length
 * (PAGE_TABLE_CAPACITY ranges, the unused ones empty at the top of the address space), and is
 * rebuilt in place only when what it maps changes: its address is baked into modules, and a table
 * per change (a UE5 title views new constant buffers all the time) made new modules and pipelines
 * for the same shaders, and cost milliseconds of sorting at every compile. */
#define PAGE_TABLE_CAPACITY 16384u

static UINT64 page_table(UINT32 *bytes)
{
    static SRWLOCK table_lock = SRWLOCK_INIT;
    static UINT64 table_va;
    static volatile UINT64 table_ready_va; /* table_va once a table has been made there */
    static LONG table_key[3];
    static ULONGLONG table_built;
    static BYTE *table_cpu;
    static struct shadow_region regions[MAX_REGIONS]; /* used only under table_lock, held exclusively */
    static struct gpu_memory_region chunks[MAX_REGIONS];
    UINT64 (*buffers)[2];
    UINT n, nchunks, nbuffers, count = 0, merged = 0;
    /* What it maps changes only with these (read before what they cover). */
    const LONG key[3] = { shadow_regions_generation(), gpu_memory_generation(), shadow_buffer_ranges_generation() };
    struct page_range *ranges;
    UINT64 va = 0;

    *bytes = 16 + PAGE_TABLE_CAPACITY * sizeof(struct page_range);
    /* Being made again by another thread (which takes milliseconds): its address is the same, as it is
     * made in place. Waiting for the lock here made every draw of every recording thread wait. */
    if (!TryAcquireSRWLockShared(&table_lock))
    {
        if ((va = table_ready_va))
            return va;
        AcquireSRWLockShared(&table_lock);
    }
    if (table_va && !memcmp(key, table_key, sizeof(key)))
    {
        va = table_va;
        ReleaseSRWLockShared(&table_lock);
        return va;
    }
    /* Remade at most every 200 ms, and never waited for: a world's constant buffers and adopted
     * memory change what it maps at nearly every dispatch, and remaking it each time (sorting
     * every buffer range) took most of a dispatch's 20 ms while the other recording threads
     * queued for the lock. A range added meanwhile is mapped a little later; a read of it before
     * then becomes an AnyPS5 fault record, not a GPU fault. */
    if (table_va && GetTickCount64() - table_built < 200)
    {
        va = table_va;
        ReleaseSRWLockShared(&table_lock);
        return va;
    }
    ReleaseSRWLockShared(&table_lock);
    if (!TryAcquireSRWLockExclusive(&table_lock))
    {
        if ((va = table_ready_va))
            return va;
        AcquireSRWLockExclusive(&table_lock);
    }
    if (table_va && !memcmp(key, table_key, sizeof(key)))
    {
        va = table_va;
        ReleaseSRWLockExclusive(&table_lock);
        return va;
    }
    table_built = GetTickCount64();
    n = shadow_regions(regions, MAX_REGIONS);
    nchunks = gpu_memory_regions(chunks, MAX_REGIONS);
    note_regions_full("Shadow heaps", n, MAX_REGIONS);
    note_regions_full("GPU memory chunks", nchunks, MAX_REGIONS);
    if (!table_cpu && FAILED(ReboxGpuAlloc(*bytes, 256, (void **)&table_cpu, &table_va)))
    {
        table_cpu = NULL;
        ReleaseSRWLockExclusive(&table_lock);
        return 0;
    }
    buffers = malloc(65536 * sizeof(*buffers));
    nbuffers = buffers ? shadow_buffer_ranges(buffers, 65536) : 0;
    if (!(ranges = malloc((n + nchunks + nbuffers) * sizeof(*ranges))))
    {
        free(buffers);
        ReleaseSRWLockExclusive(&table_lock);
        return 0;
    }
    /* Memory that exists: GPU memory chunks and buffers the title viewed, at their GPU VAs, and
     * the shadow heaps' title addresses mapped to their GPU copies. Anything else is unmapped, so
     * a stray access becomes an AnyPS5 fault record, not a GPU page fault. */
    for (UINT i = 0; i < nchunks; i++)
        ranges[count++] = (struct page_range){ chunks[i].gpu, chunks[i].gpu + chunks[i].size, chunks[i].gpu, 3, 0 };
    for (UINT i = 0; i < nbuffers; i++)
        ranges[count++] = (struct page_range){ buffers[i][0], buffers[i][1], buffers[i][0], 3, 0 };
    free(buffers);
    qsort(ranges, count, sizeof(*ranges), compare_ranges);
    /* Merge overlapping identity ranges (buffers inside chunks, repeated views). */
    for (UINT i = 0; i < count; i++)
    {
        if (merged && ranges[i].begin <= ranges[merged - 1].end)
        {
            if (ranges[i].end > ranges[merged - 1].end)
                ranges[merged - 1].end = ranges[i].end;
        }
        else
            ranges[merged++] = ranges[i];
    }
    count = merged;
    /* Shadow heaps win where they overlap (identity ranges below 4 GiB, as with RADV's low window,
     * could): identity ranges that touch one are dropped. */
    for (UINT i = 0; i < n; i++)
    {
        UINT64 begin = regions[i].title_va, end = begin + regions[i].size;
        if (!regions[i].gpu_va)
            continue;
        merged = 0;
        for (UINT j = 0; j < count; j++)
            if (ranges[j].end <= begin || ranges[j].begin >= end || ranges[j].device != ranges[j].begin)
                ranges[merged++] = ranges[j];
        count = merged;
        ranges[count++] = (struct page_range){ begin, end, regions[i].gpu_va, 3, 0 };
    }
    qsort(ranges, count, sizeof(*ranges), compare_ranges);
    if (count > PAGE_TABLE_CAPACITY)
    {
        static LONG logged;
        if (!InterlockedExchange(&logged, 1))
            rebox_log("d3d12_xs: AnyPS5 page table: %u ranges, %u kept", count, PAGE_TABLE_CAPACITY);
        count = PAGE_TABLE_CAPACITY;
    }
    {
        struct page_range *out = (struct page_range *)(table_cpu + 16);
        ((UINT32 *)table_cpu)[0] = 1;            /* version */
        ((UINT32 *)table_cpu)[1] = PAGE_TABLE_CAPACITY;
        ((UINT32 *)table_cpu)[2] = sizeof(struct page_range);
        ((UINT32 *)table_cpu)[3] = 0;
        memcpy(out, ranges, count * sizeof(*ranges));
        for (UINT i = count; i < PAGE_TABLE_CAPACITY; i++) /* empty, sorted after every real range */
            out[i] = (struct page_range){ ~0ull, ~0ull, 0, 0, 0 };
    }
    memcpy(table_key, key, sizeof(key));
    va = table_ready_va = table_va;
    ReleaseSRWLockExclusive(&table_lock);
    free(ranges);
    return va;
}

/* Buffers the GPU writes (AnyPS5's fault buffer and GDS) can't come from the content-shared pool:
 * one of each for the process, as AnyPS5's own driver keeps them. */
static struct { UINT64 gpu; volatile UINT32 *cpu; UINT32 size; } fault_buffer, gds_buffer;
static INIT_ONCE fault_buffer_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK init_fault_buffer(INIT_ONCE *once, void *param, void **context)
{
    void *cpu;
    fault_buffer.size = sa5.fault_buffer_bytes();
    if (SUCCEEDED(ReboxGpuAlloc(fault_buffer.size, 256, &cpu, &fault_buffer.gpu)))
    {
        memset(cpu, 0, fault_buffer.size);
        fault_buffer.cpu = cpu;
    }
    return TRUE;
}

static BOOL CALLBACK init_gds_buffer(INIT_ONCE *once, void *param, void **context)
{
    void *cpu;
    gds_buffer.size = sa5.gds_bytes();
    if (SUCCEEDED(ReboxGpuAlloc(gds_buffer.size, 256, &cpu, &gds_buffer.gpu)))
    {
        memset(cpu, 0, gds_buffer.size);
        gds_buffer.cpu = cpu;
    }
    return TRUE;
}

/* AnyPS5's fault record (BdaAbi.hpp): state (2 = ready), reason, address, bytes, stage,
 * instruction, ...; and the SRT guard word, which a failed speculation check (patches/anyps5/0004)
 * sets. Read at each compile, logged (the first few of each kind) and cleared. Faults come from
 * any earlier dispatch, so they aren't tied to the shader being compiled. */
#define FAULT_SRT_GUARD_WORD 6156
static LONG faults_seen;
static void poll_faults(void);
#define poll_faults_now poll_faults

/* Console shader faults and failed speculation checks seen so far (for tests). */
UINT WINAPI ReboxConsoleFaults(void)
{
    InitOnceExecuteOnce(&sa5.once, load_sa5, NULL, NULL);
    poll_faults_now();
    return (UINT)faults_seen;
}

static void poll_faults(void)
{
    static const char *const reasons[] = { "?", "unmapped address", "permission", "overflow", "invalid table",
            "invalid rectangle", "loop limit", "unaligned", "invalid descriptor", "image table", "unlinked call",
            "window lane spill", "split state" };
    static LONG fault_logs, guard_logs;
    volatile UINT32 *f = fault_buffer.cpu;

    if (!f)
        return;
    if (f[0] == 2)
    {
        UINT32 reason = f[1];
        if (reason == 6 /* LoopLimit: address is the shader hash, bytes the limit */)
        {
            if (InterlockedIncrement(&fault_logs) <= 20)
                rebox_log("d3d12_xs: console shader %#llx left a loop at its limit of %u iterations (exit at pc %#x; "
                             "APS5_LOOP_GUARD)", (unsigned long long)(f[2] | (UINT64)f[3] << 32), f[4], f[6]);
        }
        else if (InterlockedIncrement(&fault_logs) <= 20)
            rebox_log("d3d12_xs: console shader fault: %s at %#llx (%u bytes), instruction %#x",
                         reason < ARRAYSIZE(reasons) ? reasons[reason] : "?",
                         (unsigned long long)(f[2] | (UINT64)f[3] << 32), f[4], f[6]);
        for (UINT i = 0; i < 10; i++)
            f[i] = 0;
        InterlockedIncrement(&faults_seen);
    }
    if (fault_buffer.size > FAULT_SRT_GUARD_WORD * 4 && f[FAULT_SRT_GUARD_WORD])
    {
        if (InterlockedIncrement(&guard_logs) <= 20)
            rebox_log("d3d12_xs: a speculated console descriptor didn't match at run time (SRT guard %#x); "
                         "REBOX_ANYPS5_SPECULATE=0 turns speculation off", f[FAULT_SRT_GUARD_WORD]);
        f[FAULT_SRT_GUARD_WORD] = 0;
        InterlockedIncrement(&faults_seen);
    }
}

static UINT64 zero_buffer(UINT32 size)
{
    void *zeros = calloc(1, size);
    UINT64 gpu = zeros ? host_buffer(zeros, size) : 0;
    free(zeros);
    return gpu;
}

/* ------------------------------------------------------- finalized modules */

static SRWLOCK modules_lock = SRWLOCK_INIT;
/* result: a graphics stage's AnyPS5 result, kept when it was compiled for a rect-list draw
 * (ReboxConsoleRectStages builds that draw's tessellation stages from it). */
/* vk_key: the key vkd3d-proton gets (its pipelines are cached by it): the key of the first module
 * with the same SPIR-V, so two keys that finalize to the same words share one pipeline. Such an
 * alias shares that module's words. */
static struct module
{
    UINT64 key[2], vk_key[2], content;
    UINT32 *words, count;
    struct sa5_result *result;
    struct module *next, *content_next;
} *modules[256], *modules_by_content[256];
static UINT module_count, module_aliases;
static UINT64 module_bytes;

static struct module *find_module(const UINT64 key[2])
{
    struct module *m;

    AcquireSRWLockShared(&modules_lock);
    for (m = modules[key[0] & 255]; m && memcmp(m->key, key, sizeof(m->key)); m = m->next)
        ;
    ReleaseSRWLockShared(&modules_lock);
    return m;
}

/* Modules being built: a thread wanting one waits for the thread building it instead of building
 * it too. Draws that overran their wait asked again and again, and all 8 compile workers built the
 * same 1.7M-word module at once while a loading screen's compiles queued behind them. */
static SRWLOCK building_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE building_done = CONDITION_VARIABLE_INIT;
static UINT64 building[64][2];

/* The module for key if another thread built it meanwhile; NULL with the key claimed (or, with
 * no room to note it, unclaimed) for the caller to build and then pass to module_built. */
static struct module *module_wait(const UINT64 key[2])
{
    struct module *m = NULL;
    UINT i, free_slot;

    AcquireSRWLockExclusive(&building_lock);
    for (;;)
    {
        free_slot = ARRAYSIZE(building);
        for (i = 0; i < ARRAYSIZE(building); i++)
        {
            if (!memcmp(building[i], key, sizeof(building[i])))
                break;
            if (free_slot == ARRAYSIZE(building) && !building[i][0] && !building[i][1])
                free_slot = i;
        }
        if (i == ARRAYSIZE(building))
            break;
        SleepConditionVariableSRW(&building_done, &building_lock, INFINITE, 0);
    }
    if (!(m = find_module(key)) && free_slot < ARRAYSIZE(building))
        memcpy(building[free_slot], key, sizeof(building[free_slot]));
    ReleaseSRWLockExclusive(&building_lock);
    return m;
}

static void module_built(const UINT64 key[2])
{
    AcquireSRWLockExclusive(&building_lock);
    for (UINT i = 0; i < ARRAYSIZE(building); i++)
        if (!memcmp(building[i], key, sizeof(building[i])))
            memset(building[i], 0, sizeof(building[i]));
    ReleaseSRWLockExclusive(&building_lock);
    WakeAllConditionVariable(&building_done);
}

static struct module *add_module(const UINT64 key[2], const UINT32 *words, UINT32 count)
{
    struct module *m = malloc(sizeof(*m)), *other, *same = NULL;
    const UINT64 content = hash_words(0xcbf29ce484222325ull ^ count, words, count * sizeof(*words));

    if (!m)
        return NULL;
    memset(m, 0, sizeof(*m));
    memcpy(m->key, key, sizeof(m->key));
    memcpy(m->vk_key, key, sizeof(m->vk_key));
    m->content = content;
    m->count = count;
    AcquireSRWLockExclusive(&modules_lock);
    for (other = modules[key[0] & 255]; other && memcmp(other->key, key, sizeof(other->key)); other = other->next)
        ;
    if (!other)
        for (same = modules_by_content[content & 255]; same; same = same->content_next)
            if (same->content == content && same->count == count && !memcmp(same->words, words, count * sizeof(*words)))
                break;
    if (other)
    {
        free(m);
        m = other;
    }
    else if (same)
    {
        /* The same SPIR-V under another key (inputs that differ only in what the module doesn't
         * keep): one pipeline for both. */
        m->words = same->words;
        memcpy(m->vk_key, same->vk_key, sizeof(m->vk_key));
        m->next = modules[key[0] & 255];
        modules[key[0] & 255] = m;
        module_aliases++;
    }
    else if ((m->words = malloc(count * sizeof(*words))))
    {
        memcpy(m->words, words, count * sizeof(*words));
        m->next = modules[key[0] & 255];
        modules[key[0] & 255] = m;
        m->content_next = modules_by_content[content & 255];
        modules_by_content[content & 255] = m;
        module_count++;
        module_bytes += (UINT64)count * sizeof(*words);
    }
    else
    {
        free(m);
        m = NULL;
    }
    ReleaseSRWLockExclusive(&modules_lock);
    return m;
}

/* REBOX_VARIANT_LOG=1: for each console stage's second and later modules, what they differ from
 * its first one by: AnyPS5's result (the code it made for that call's descriptors and memory), the
 * host buffers given to sa5_finalize_draw (field by field), the bindings, and the SPIR-V. With
 * vkd3d-proton's "variants:" lines (patch 0018), to see why a stage has more than one pipeline. */
static const char *stage_name(UINT32 stage);

static BOOL variant_log(void)
{
    static int on = -1;
    if (on < 0)
    {
        const char *env = getenv("REBOX_VARIANT_LOG");
        on = env && *env == '1';
    }
    return on;
}

static void variant_note(UINT64 stage, UINT32 stage_bit, const UINT64 identity[2], const struct sa5_host_buffer *hosts,
                         UINT32 host_count, UINT64 bind_hash, const struct module *m)
{
    static struct variant_first
    {
        UINT64 stage, identity[2], bind_hash;
        struct sa5_host_buffer hosts[8];
        UINT32 host_count, modules;
        const struct module *module;
    } firsts[4096];
    static SRWLOCK lock = SRWLOCK_INIT;
    struct variant_first *v = NULL, copy;
    char line[512];
    int at = 0;
    UINT32 differing = 0;

    AcquireSRWLockExclusive(&lock);
    for (UINT i = 0; i < 64; i++)
    {
        struct variant_first *e = &firsts[(stage + i) % ARRAYSIZE(firsts)];
        if (!e->stage || e->stage == stage)
        {
            v = e;
            break;
        }
    }
    if (v && !v->stage)
    {
        v->stage = stage;
        memcpy(v->identity, identity, sizeof(v->identity));
        v->bind_hash = bind_hash;
        v->host_count = min(host_count, 8u);
        memcpy(v->hosts, hosts, v->host_count * sizeof(*hosts));
        v->module = m;
    }
    if (v)
        v->modules++;
    if (v)
        copy = *v;
    ReleaseSRWLockExclusive(&lock);
    if (!v || copy.modules < 2)
        return;

    if (memcmp(copy.identity, identity, sizeof(copy.identity)))
        at += snprintf(line + at, sizeof(line) - at, " AnyPS5 result,");
    if (copy.bind_hash != bind_hash)
        at += snprintf(line + at, sizeof(line) - at, " bindings,");
    if (copy.host_count != host_count)
        at += snprintf(line + at, sizeof(line) - at, " host buffer count (%u, was %u),", host_count, copy.host_count);
    for (UINT32 i = 0; i < min(host_count, copy.host_count) && at < 400; i++)
    {
        const struct sa5_host_buffer *a = &copy.hosts[i], *b = &hosts[i];
        if (a->set != b->set || a->binding != b->binding)
            at += snprintf(line + at, sizeof(line) - at, " host %u binding,", i);
        if (a->push_slot != b->push_slot || a->push_base != b->push_base)
            at += snprintf(line + at, sizeof(line) - at, " host %u slot,", i);
        if (a->array_length != b->array_length)
            at += snprintf(line + at, sizeof(line) - at, " host %u length (%u, was %u),", i, b->array_length,
                           a->array_length);
        if (a->address != b->address)
            at += snprintf(line + at, sizeof(line) - at, " host %u address,", i);
        if (a->guest_array != b->guest_array)
            at += snprintf(line + at, sizeof(line) - at, " host %u kind,", i);
    }
    if (copy.module->count != m->count)
        at += snprintf(line + at, sizeof(line) - at, " SPIR-V size (%u words, was %u)", m->count, copy.module->count);
    else
    {
        for (UINT32 i = 0; i < m->count; i++)
            differing += copy.module->words[i] != m->words[i];
        at += snprintf(line + at, sizeof(line) - at, differing ? " %u SPIR-V words" : " same SPIR-V", differing);
    }
    if (at && line[at - 1] == ',')
        line[--at] = 0;
    rebox_log("d3d12_xs: variants: console %s shader %016llx module %u differs from its first by:%s",
              stage_name(stage_bit), (unsigned long long)stage, copy.modules, line);
}

/* For the memory log: the finalized modules kept, and their SPIR-V's bytes. */
void console_compile_module_stats(UINT *count, UINT64 *bytes)
{
    AcquireSRWLockShared(&modules_lock);
    *count = module_count;
    *bytes = module_bytes;
    ReleaseSRWLockShared(&modules_lock);
}

/* -------------------------------------------------------------- compiling */

/* Console shaders often start by setting FLAT_SCRATCH with s_setreg_imm32_b32 (SOPK top 9 bits
 * 0x175, a literal follows) or s_setreg_b32 (0x173) to HW_REG_FLAT_SCR_LO/HI (hwreg ids 20/21).
 * Scratch is the host's business here, so the leading run of those becomes s_nops of the same
 * size. Only the run at the start is touched: further in, a literal could look like one.
 * Returns how many instructions were replaced. */
#define S_NOP 0xbf800000u
static UINT nop_flat_scratch_setup(UINT32 *code, UINT32 dwords)
{
    UINT i = 0, count = 0, scanned = 0;

    /* The setup sits among the prologue's scalar instructions (a UE5 title's pixel shaders move M0
     * first): scan those, up to the first vector or memory instruction. */
    while (i < dwords && scanned++ < 32)
    {
        UINT32 w = code[i], op = w >> 23, id = w & 0x3f;
        if ((op == 0x175 || op == 0x173) && (id == 20 || id == 21))
        {
            code[i++] = S_NOP;
            if (op == 0x175 && i < dwords)
                code[i++] = S_NOP;
            count++;
        }
        else if (op == 0x17d || op == 0x17e)              /* SOP1, SOPC */
            i += 1 + ((w & 0xff) == 0xff || (op == 0x17e && ((w >> 8) & 0xff) == 0xff));
        else if (w >> 28 == 0xb)                          /* SOPK */
            i += 1 + (((w >> 23) & 0x1f) == 0x15);
        else if (w >> 30 == 2 && op != 0x17f)             /* SOP2 */
            i += 1 + ((w & 0xff) == 0xff || ((w >> 8) & 0xff) == 0xff);
        else if (w >> 16 == 0xbfa0 || w >> 16 == 0xbf80)  /* s_inst_prefetch (mesh shaders open with it), s_nop */
            i++;
        else
            break;
    }
    return count;
}

/* The last compile's outcome, for test harnesses (ReboxConsoleLastResult). */
static SRWLOCK last_lock = SRWLOCK_INIT;
static __thread HRESULT last_hr; /* per thread: the background compiler has its own */
static __thread char last_message[512];

static void set_last(HRESULT hr, const char *what, const char *detail)
{
    AcquireSRWLockExclusive(&last_lock);
    last_hr = hr;
    snprintf(last_message, sizeof(last_message), "%s%s%s", what, detail ? ": " : "", detail ? detail : "");
    ReleaseSRWLockExclusive(&last_lock);
}

/* The calling thread's most recent ReboxConsoleCompile result and a one-line description. */
HRESULT WINAPI ReboxConsoleLastResult(char *message, SIZE_T size)
{
    HRESULT hr;

    AcquireSRWLockShared(&last_lock);
    hr = last_hr;
    if (message && size)
        snprintf(message, size, "%s", last_message);
    ReleaseSRWLockShared(&last_lock);
    return hr;
}

/* Whether a vkd3d-proton descriptor of this view kind can back an AnyPS5 binding of this kind:
 * bindless sets alias every type at one slot, so a mismatch reads the wrong descriptor type. */
static BOOL view_fits(UINT32 kind, enum shadow_view view)
{
    switch (kind)
    {
    case 0: return view == SHADOW_VIEW_CBV;
    case 1: return view == SHADOW_VIEW_SRV_RAW || view == SHADOW_VIEW_UAV_RAW;
    case 2: return view == SHADOW_VIEW_SRV_TYPED || view == SHADOW_VIEW_SRV_RAW;
    case 3: return view == SHADOW_VIEW_UAV_TYPED || view == SHADOW_VIEW_UAV_RAW;
    case 4: return view == SHADOW_VIEW_SRV_TEXTURE;
    case 5: return view == SHADOW_VIEW_UAV_TEXTURE;
    case 6: return view == SHADOW_VIEW_SAMPLER;
    default: return FALSE;
    }
}

static const char *const view_names[] = { "nothing", "a texture SRV", "a typed buffer SRV", "a raw buffer SRV",
        "a texture UAV", "a typed buffer UAV", "a raw buffer UAV", "a CBV", "a sampler" };

/* Logs a shader's first failure only (keyed by its code). */
static void log_failure(const struct rebox_deferred_shader *h, const char *what, const char *detail)
{
    static SRWLOCK log_lock = SRWLOCK_INIT;
    static UINT64 logged[256];
    static UINT logged_count;
    UINT64 id = hash_bytes(0xcbf29ce484222325ull, (const BYTE *)h + h->code_offset, h->code_dwords * 4u);
    BOOL seen = FALSE;

    set_last(E_FAIL, what, detail);
    AcquireSRWLockExclusive(&log_lock);
    for (UINT i = 0; i < logged_count && !seen; i++)
        seen = logged[i] == id;
    if (!seen && logged_count < ARRAYSIZE(logged))
        logged[logged_count++] = id;
    ReleaseSRWLockExclusive(&log_lock);
    if (!seen)
        rebox_log("d3d12_xs: console shader %016llx: %s%s%s; its dispatches are skipped",
                     (unsigned long long)id, what, detail ? ": " : "", detail ? detail : "");
}

/* Graphics stages AnyPS5 refused for their code (their code and stage): not retried at every draw,
 * which falls back at once. Refusals that come from what the descriptors held are retried instead
 * (graphics_refused). */
static UINT64 failed_stages[256];
static LONG failed_stage_count;

/* Compile timings for the log, by kind (graphics stages, compute) and by who asked: a title thread
 * at its draw or dispatch, which waits for it, or a background worker (a program's first compile, a
 * graphics variant), whose time the title doesn't wait for. Phases: memory regions, waiting for the
 * compiler lock, AnyPS5, bindings (parts: identity, heap lookups, page table, host buffers, guest
 * buffer table entries, T#s found by their texture's address, dummy views and view checks, late
 * fills' sources, the log's notes), finalizing a module. */
enum { PHASE_REGIONS, PHASE_LOCK, PHASE_ANYPS5, PHASE_BINDINGS, PHASE_MODULE, PHASES };
enum { PART_IDENTITY, PART_LOOKUPS, PART_PAGE_TABLE, PART_HOST_BUFFERS, PART_GUEST_ENTRIES, PART_BY_ADDRESS,
       PART_SLOTS, PART_LATE, PART_NOTES, PARTS };
static struct compile_stats
{
    volatile LONG calls, memo_skips;
    volatile LONG64 ticks, phase[PHASES], part[PARTS];
} compile_stats[2][2]; /* [compute][background] */
static __thread struct compile_stats *stats;
static __thread LONG64 relock_ticks; /* a background compile's wait to take the lock back */

static LONG64 ticks_now(void)
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return now.QuadPart;
}

static void phase_end(LARGE_INTEGER *mark, UINT phase)
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    if (mark->QuadPart && stats)
        InterlockedAdd64(&stats->phase[phase], now.QuadPart - mark->QuadPart);
    *mark = now;
}

static void part_add(UINT part, LONG64 ticks)
{
    if (stats)
        InterlockedAdd64(&stats->part[part], ticks);
}

/* Times stmt into a part of the bindings phase. */
#define TIMED(part, stmt) do { LONG64 timed_start_ = ticks_now(); stmt; part_add(part, ticks_now() - timed_start_); } while (0)

static double qpc_frequency(void)
{
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    return (double)f.QuadPart;
}

/* ------------------------------------------------------------ binding cache */

/* A guest descriptor binding's heap slots, kept for the next draw or dispatch whose binding has the
 * same words: at thousands of draws a frame, resolving every descriptor again (heap lookups, T#s
 * found by their texture's address, views made in the reserved tail) was most of a draw's
 * bindings. An entry holds while nothing it was resolved from has changed: the shader-visible
 * heaps (a write or copy to any slot of one, a heap made or gone, a reserved view forgotten), and,
 * for a binding with a T# looked for by its address, which textures are at which addresses. Empty
 * T#s noted for a late fill are noted again at every draw, their sources found then. Guest buffer
 * tables and host buffers are made at every draw as before. REBOX_BINDING_CACHE=0 turns it off. */
#define BIND_CACHE_SLOTS 4096u
#define BIND_CACHE_ELEMENTS 4096u /* bindings of more elements aren't kept */
#define BIND_CACHE_LATE 64u
static volatile LONG heap_generation, texture_generation;
static volatile LONG bind_hits, bind_misses, late_notes;

void console_bindings_changed(BOOL textures)
{
    InterlockedIncrement(textures ? &texture_generation : &heap_generation);
}

/* xmem's GPU views (proxy_texture_at translates T# addresses through them). */
static LONG gpu_views_generation(void)
{
    static LONG (WINAPI *volatile get)(void);
    static volatile LONG resolved;
    if (!resolved)
    {
        get = (void *)GetProcAddress(GetModuleHandleA("xmem.dll"), "ReboxGpuViewsGeneration");
        resolved = 1;
    }
    return get ? get() : 0;
}

struct bind_entry
{
    UINT64 hash;
    SIZE_T heap;
    UINT32 role, kind, per, count, graphics;
    LONG heaps, textures, views;  /* the generations it was resolved under */
    BOOL by_address;              /* a T# was looked for by its texture's address */
    UINT late_count;
    UINT32 late[BIND_CACHE_LATE]; /* its elements that are empty T#s noted for a late fill */
    UINT32 *indices;              /* count heap slots, then the binding's words (count * per) */
};
static struct { SRWLOCK lock; struct bind_entry *entry; } bind_cache[BIND_CACHE_SLOTS];

/* A binding as the cache knows it, and the generations read before it is resolved: a change while
 * it is makes what it resolves to stale at once. */
struct bind_query { UINT64 hash; SIZE_T heap; UINT32 per, graphics; LONG heaps, textures, views; };

static BOOL bind_cache_enabled(void)
{
    static int enabled = -1;
    if (enabled < 0)
    {
        const char *v = getenv("REBOX_BINDING_CACHE");
        enabled = !v || strcmp(v, "0");
    }
    return enabled;
}

static void bind_query_make(struct bind_query *q, SIZE_T heap, BOOL graphics, const struct sa5_binding *b, UINT32 per)
{
    const UINT32 head[5] = { b->role, b->kind, per, b->count, graphics };
    q->heaps = heap_generation;
    q->textures = texture_generation;
    q->views = gpu_views_generation();
    MemoryBarrier();
    q->heap = heap;
    q->per = per;
    q->graphics = graphics;
    q->hash = hash_words(hash_words(0x62696e64696e6773ull ^ heap, head, sizeof(head)), b->words,
                         (SIZE_T)b->count * per * 4);
}

static struct bind_entry **bind_slot(const struct bind_query *q, SRWLOCK **lock)
{
    UINT i = (UINT)(q->hash >> 40) % BIND_CACHE_SLOTS;
    *lock = &bind_cache[i].lock;
    return &bind_cache[i].entry;
}

/* The binding's heap slots and late elements as last resolved, if that still holds. */
static BOOL bind_cache_get(const struct bind_query *q, const struct sa5_binding *b, UINT32 *indices, UINT32 *late,
                           UINT *late_count)
{
    SRWLOCK *lock;
    struct bind_entry **slot = bind_slot(q, &lock), *e;
    BOOL hit = FALSE;

    AcquireSRWLockShared(lock);
    if ((e = *slot) && e->hash == q->hash && e->heap == q->heap && e->role == b->role && e->kind == b->kind
            && e->per == q->per && e->count == b->count && e->graphics == q->graphics && e->heaps == q->heaps
            && (!e->by_address || (e->textures == q->textures && e->views == q->views))
            && !memcmp(e->indices + e->count, b->words, (SIZE_T)b->count * q->per * 4))
    {
        memcpy(indices, e->indices, b->count * sizeof(*indices));
        memcpy(late, e->late, e->late_count * sizeof(*late));
        *late_count = e->late_count;
        hit = TRUE;
    }
    ReleaseSRWLockShared(lock);
    InterlockedIncrement(hit ? &bind_hits : &bind_misses);
    return hit;
}

static void bind_cache_put(const struct bind_query *q, const struct sa5_binding *b, const UINT32 *indices,
                           BOOL by_address, const UINT32 *late, UINT late_count)
{
    SIZE_T words = (SIZE_T)b->count * q->per;
    struct bind_entry *e, *old;
    SRWLOCK *lock;
    struct bind_entry **slot;

    if (b->count > BIND_CACHE_ELEMENTS || late_count > BIND_CACHE_LATE || !(e = malloc(sizeof(*e))))
        return;
    if (!(e->indices = malloc((b->count + words) * sizeof(UINT32))))
    {
        free(e);
        return;
    }
    e->hash = q->hash;
    e->heap = q->heap;
    e->role = b->role;
    e->kind = b->kind;
    e->per = q->per;
    e->count = b->count;
    e->graphics = q->graphics;
    e->heaps = q->heaps;
    e->textures = q->textures;
    e->views = q->views;
    e->by_address = by_address;
    e->late_count = late_count;
    memcpy(e->late, late, late_count * sizeof(*late));
    memcpy(e->indices, indices, b->count * sizeof(UINT32));
    memcpy(e->indices + b->count, b->words, words * sizeof(UINT32));
    slot = bind_slot(q, &lock);
    AcquireSRWLockExclusive(lock);
    old = *slot;
    *slot = e;
    ReleaseSRWLockExclusive(lock);
    if (old)
    {
        free(old->indices);
        free(old);
    }
}

/* For tests: the binding cache's hits and misses, and the empty T#s noted for a late fill. */
void WINAPI ReboxConsoleBindingStats(UINT64 out[4])
{
    out[0] = bind_hits;
    out[1] = bind_misses;
    out[2] = late_notes;
    out[3] = 0;
}

/* Shaders whose bindings couldn't be made for a reason their next draw or dispatch meets again
 * (more descriptor elements or host buffers than there is room for, a binding of a kind not
 * handled, a module that can't be fitted): AnyPS5 isn't asked again at each one, only at every
 * BINDING_RETRY-th, in case what it captures has changed. A title's bindless compute shaders
 * captured tables of 24,000 descriptors at every dispatch, all refused. */
#define BINDING_RETRY 64
static struct { volatile UINT64 id; volatile LONG calls; } binding_failures[512];
static volatile LONG binding_failure_count;

/* Whether to skip this call of shader id (its bindings failed before, and it isn't time to retry). */
static BOOL binding_failure_skips(UINT64 id)
{
    LONG n = min(binding_failure_count, (LONG)ARRAYSIZE(binding_failures));
    for (LONG i = 0; i < n; i++)
        if (binding_failures[i].id == id)
            return InterlockedIncrement(&binding_failures[i].calls) % BINDING_RETRY != 0;
    return FALSE;
}

static void binding_failure_note(UINT64 id)
{
    LONG n = min(binding_failure_count, (LONG)ARRAYSIZE(binding_failures)), slot;
    for (LONG i = 0; i < n; i++)
        if (binding_failures[i].id == id)
            return;
    if ((slot = InterlockedIncrement(&binding_failure_count) - 1) < (LONG)ARRAYSIZE(binding_failures))
        binding_failures[slot].id = id;
}

/* A retry that worked: the shader is asked for at every call again. */
static void binding_failure_clear(UINT64 id)
{
    LONG n = min(binding_failure_count, (LONG)ARRAYSIZE(binding_failures));
    for (LONG i = 0; i < n; i++)
        if (binding_failures[i].id == id)
            binding_failures[i].id = 0;
}

static BOOL stage_failed(UINT64 id)
{
    LONG n = min(failed_stage_count, (LONG)ARRAYSIZE(failed_stages));
    for (LONG i = 0; i < n; i++)
        if (failed_stages[i] == id)
            return TRUE;
    return FALSE;
}

/* AnyPS5 refusals that come from what a stage's descriptors and tables held at that call, not from
 * its code: an SRT walk through memory the request didn't hold, or a descriptor read there that
 * isn't one (often a table the title has since written over). */
static BOOL runtime_refusal(const char *error)
{
    static const char *const reasons[] = { "SrtWalker::", "runtime image descriptor is invalid",
            "buffer descriptor uses an unsupported type", "is incompatible with the static runtime image interface" };
    for (UINT i = 0; error && i < ARRAYSIZE(reasons); i++)
        if (strstr(error, reasons[i]))
            return TRUE;
    return FALSE;
}

/* A graphics stage AnyPS5 refused: a refusal from its code is remembered for the run (stage_failed);
 * one from what it read at this call is retried at every BINDING_RETRY-th call, as a binding failure
 * is (the target title's mesh stages walked user-data tables outside the regions they were given). */
static void graphics_refused(UINT64 stage_id, const char *error)
{
    LONG slot;

    if (runtime_refusal(error))
    {
        binding_failure_note(stage_id);
        return;
    }
    slot = InterlockedIncrement(&failed_stage_count) - 1;
    if (slot < (LONG)ARRAYSIZE(failed_stages))
        failed_stages[slot] = stage_id;
}

/* The user SGPR count from RSRC2: graphics stages' USER_SGPR_MSB (bit 27) makes 32 possible
 * (compute's bit 27 is an exception enable). */
/* A graphics stage's name for the log. */
static const char *stage_name(UINT32 stage)
{
    return stage == 0x10 ? "pixel" : stage == 0x80 ? "mesh" : "vertex";
}

static UINT user_sgprs(const struct rebox_deferred_shader *h, BOOL graphics)
{
    return ((h->rsrc2 >> 1) & 0x1f) | (graphics ? ((h->rsrc2 >> 27) & 1) << 5 : 0);
}

/* The address of the n-th T# a graphics stage loads from its descriptor tables, or 0. The code is
 * scanned for GFX10 scalar loads of 8 dwords (s_load_dwordx8: SMEM, opcode 3) from an SGPR pair whose
 * low register is a table pointer (user SGPRs 8-15, a mesh stage's 16-23: the console's graphics
 * layout, used as {sN, 0}): the address is that SGPR plus the load's offset. */
static UINT64 late_load_source(const struct rebox_deferred_shader *h, const struct rebox_compile_in *in, UINT n,
                               UINT32 opcode);

/* The first SGPR of a graphics stage's root parameters, whose first 8 the late T#s and V#s are
 * traced through: s8, or a mesh stage's s16 (mesh_user_sgprs). */
static UINT table_sgprs(const struct rebox_deferred_shader *h)
{
    return h->stage == 0x80 ? 16 : 8;
}

/* Whether the guest memory at address holds a T# (dwords 8: an image type, 8-15, in word 3's top
 * nibble) or a V# (dwords 4: type 0 in word 3's top two bits, and an address or a size), or can't be
 * read. Empty words and an earlier draw's words left in a per-draw table don't. */
static BOOL slot_holds_descriptor(UINT64 address, UINT dwords)
{
    BOOL WINAPI ReboxShadowRead(UINT64 address, void *out, SIZE_T size);
    UINT32 w[8] = { 0 };

    if (!ReboxShadowRead(address, w, dwords * 4u))
        return TRUE;
    if (dwords == 8)
        return (w[3] >> 28) >= 8;
    return !(w[3] >> 30) && (w[0] || w[1] || w[2]);
}

/* The n-th T# the code's image instructions use, in the order they first use it (AnyPS5 numbers a
 * binding's elements so: a video shader samples its Y plane, loaded third, first), traced to the
 * s_load_dwordx8 off a table SGPR that loaded its registers, counting only those whose slot holds
 * no T# now; else the n-th such load. */
static UINT64 late_source(const struct rebox_deferred_shader *h, const struct rebox_compile_in *in, UINT n)
{
    const UINT32 *code = (const UINT32 *)((const BYTE *)h + h->code_offset);
    struct { UINT sgpr; UINT64 address; } loads[32];
    UINT load_count = 0, used[32], used_count = 0;

    for (UINT32 i = 0, size; i + 1 < h->code_dwords; i += size)
    {
        UINT32 w0 = code[i], w1 = code[i + 1];
        size = max(console_gfx10_size(code + i, h->code_dwords - i), 1u);
        if (w0 == 0xbf810000u) /* s_endpgm */
            break;
        if (w0 >> 26 == 0x3d) /* SMEM */
        {
            UINT32 op = (w0 >> 18) & 0xff, sbase = (w0 & 0x3f) * 2;
            if (op == 3 && sbase >= table_sgprs(h) && sbase < table_sgprs(h) + 8 && in->user_sgprs[sbase] && (w1 >> 25) == 0x7d
                    && load_count < ARRAYSIZE(loads))
            {
                loads[load_count].sgpr = (w0 >> 6) & 0x7f;
                loads[load_count++].address = (UINT64)in->user_sgprs[sbase] + (w1 & 0x1fffff);
            }
        }
        else if (w0 >> 26 == 0x3c) /* MIMG: srsrc (in fours) in the second dword */
        {
            UINT sgpr = ((w1 >> 16) & 0x1f) * 4, k;
            for (k = 0; k < used_count && used[k] != sgpr; k++)
                ;
            if (k == used_count && used_count < ARRAYSIZE(used))
                used[used_count++] = sgpr;
        }
    }
    /* The n-th of the T#s the code uses whose slot holds no T# now (empty, or an earlier draw's
     * words): AnyPS5 gives those empty, and the ones the title has written already as they are, so
     * the n-th empty binding is the n-th such slot, not the n-th T#. */
    {
        UINT64 unready[32];
        UINT unready_count = 0;
        for (UINT u = 0; u < used_count; u++)
            for (UINT k = load_count; k-- > 0;)
                if (loads[k].sgpr == used[u])
                {
                    if (!slot_holds_descriptor(loads[k].address, 8))
                        unready[unready_count++] = loads[k].address;
                    break;
                }
        if (n < unready_count)
            return unready[n];
    }
    if (n < used_count)
        for (UINT k = load_count; k-- > 0;) /* the last load into those registers */
            if (loads[k].sgpr == used[n])
                return loads[k].address;
    return late_load_source(h, in, n, 3 /* s_load_dwordx8: a T# */);
}

/* A guest buffer's address as the GPU should read it: a title address (4 GiB to 2^46, outside the
 * GPU window, as proxy.c's ia_adopt) in a GPU-only view of title memory becomes the CPU address
 * behind it, and must be GPU memory there or be made so; FALSE if it can't be. Other addresses are
 * vkd3d-proton's own. *bytes is cut to the GPU memory that runs on from the address: the shader
 * reads the buffer at that address directly, bounded only by its length, so a V# whose records
 * run past that memory (into title memory between two GPU memory ranges) faulted the GPU. */
/* Guest buffers in title memory read as empty since the last census (its first logs are long spent
 * by a world). */
static volatile LONG guest_buffers_empty;

static BOOL guest_buffer_mappable(UINT64 *base, UINT64 *bytes)
{
    static ULONGLONG (WINAPI *volatile to_cpu)(ULONGLONG);
    static volatile LONG resolved;
    UINT64 address = *base, end, check;

    /* Only its start must be GPU memory: V#s with all-ones record counts are raw pointers to
     * memory of no stated size, and requiring all of that to be GPU memory refused them, which
     * broke the shaders that use them. */
    check = min(*bytes, 0x10000ull);
    /* Below 4 GiB is the driver's low window, not title memory: only Rebox's own GPU memory
     * chunks and buffers the title made views of or asked the address of (proxy.c) are buffers
     * there. Anything else is a V# whose words aren't an address (a float, 1.0 = 0x3f800000,
     * faulted the GPU there). */
    if (address && check && address < 0x100000000ull)
    {
        /* Read no further than the chunk's or the buffer's end: a V# with all-ones records kept a
         * 4 GiB length there, enough to reach unmapped low addresses (a TCP read faulted the GPU at
         * 0x90061000). */
        if ((gpu_memory_holds_gpu(address, check) && gpu_memory_gpu_extent(address, &end))
                || (shadow_buffer_range_at(address, NULL, &end) && end >= address + check))
        {
            if (*bytes > end - address)
            {
                static LONG logged;
                if (*bytes - (end - address) >= 0x1000 && InterlockedIncrement(&logged) <= 8)
                    rebox_log("d3d12_xs: a guest buffer at %#llx+%#llx (low) runs past its memory at %#llx: cut to "
                              "%#llx bytes", (unsigned long long)address, (unsigned long long)*bytes,
                              (unsigned long long)end, (unsigned long long)(end - address));
                *bytes = end - address;
            }
            return TRUE;
        }
        {
            static LONG logged;
            if (InterlockedIncrement(&logged) <= 8)
                rebox_log("d3d12_xs: a guest buffer at %#llx (low, not GPU memory Rebox made): read as empty",
                          (unsigned long long)address);
        }
        return FALSE;
    }
    /* In the identity window (vkd3d-proton's host-visible heaps): not title memory, and no
     * allocation there passes 256 MiB. A V# with an all-ones record count there (a raw pointer of
     * no stated size) kept its 4 GiB-times-stride length, and a garbage index read 160 GiB past
     * the window's last allocation: a GPU fault, the same address every run. */
    if (address + check > 0x200000000000ull && address < 0x210000000000ull)
    {
        UINT64 end = gpu_memory_window_end();
        if (end && address >= end)
        {
            static LONG logged;
            if (InterlockedIncrement(&logged) <= 8)
                rebox_log("d3d12_xs: a guest buffer at %#llx is past every allocation in the GPU memory window "
                          "(%#llx): read as empty", (unsigned long long)address, (unsigned long long)end);
            return FALSE;
        }
        /* A buffer the title knows there (it made a view of it or asked its GPU address: root
         * views, vertex buffers) or a GPU memory chunk: read no further than its end. A root view
         * is a V# the code builds around the root address with an all-ones record count, and the
         * cut to 256 MiB still let a wild index read past its buffer into unmapped addresses. */
        if (shadow_buffer_range_at(address, NULL, &end) || gpu_memory_gpu_extent(address, &end))
        {
            if (*bytes > end - address)
            {
                static LONG logged;
                if (*bytes - (end - address) >= 0x1000 && InterlockedIncrement(&logged) <= 8)
                    rebox_log("d3d12_xs: a guest buffer at %#llx+%#llx in the GPU memory window runs past its "
                              "buffer at %#llx: cut to %#llx bytes", (unsigned long long)address,
                              (unsigned long long)*bytes, (unsigned long long)end, (unsigned long long)(end - address));
                *bytes = end - address;
            }
            return TRUE;
        }
        /* In no buffer the title knows: as AnyPS5's page table does for addresses, read as empty
         * (REBOX_WINDOW_UNKNOWN_BUFFERS=keep: cut to 256 MiB, as before). */
        {
            static LONG logged, keep = -1;
            if (keep < 0)
            {
                char v[8];
                DWORD n = GetEnvironmentVariableA("REBOX_WINDOW_UNKNOWN_BUFFERS", v, sizeof(v));
                keep = n && n < sizeof(v) && !strcmp(v, "keep");
            }
            if (InterlockedIncrement(&logged) <= 16)
                rebox_log("d3d12_xs: a guest buffer at %#llx+%#llx in the GPU memory window is in no buffer the "
                          "title knows: %s", (unsigned long long)address, (unsigned long long)*bytes,
                          keep ? "cut to 256 MiB (REBOX_WINDOW_UNKNOWN_BUFFERS=keep)" : "read as empty");
            if (!keep)
                return FALSE;
        }
        if (*bytes > (256ull << 20))
        {
            static LONG logged;
            if (InterlockedIncrement(&logged) <= 8)
                rebox_log("d3d12_xs: a guest buffer at %#llx+%#llx in the GPU memory window: cut to 256 MiB",
                          (unsigned long long)address, (unsigned long long)*bytes);
            *bytes = 256ull << 20;
        }
        return TRUE;
    }
    if (!address || !check || address + check > 0x400000000000ull)
        return TRUE;
    if (!resolved)
    {
        to_cpu = (void *)GetProcAddress(GetModuleHandleA("xmem.dll"), "ReboxGpuViewToCpu");
        resolved = 1;
    }
    if (to_cpu)
        address = to_cpu(address);
    /* check is a guess too: as much of it as can be (a mesh draw's root view in title memory whose
     * 64 KiB ran past the committed run, or into GPU memory above, was read as empty). Then cut to
     * the GPU memory there is. */
    if ((gpu_memory_identity_extent(address, &end) && end >= address + check) || gpu_memory_adopt_some(address, check))
    {
        if (!gpu_memory_identity_run_end(address, &end) || end <= address)
            end = address + check;
        if (*bytes > end - address)
        {
            static LONG logged;
            if (*bytes - (end - address) >= 0x1000 && *bytes < 0x100000000ull && InterlockedIncrement(&logged) <= 8)
                rebox_log("d3d12_xs: a guest buffer at %#llx+%#llx runs past GPU memory at %#llx: cut to %#llx bytes",
                          (unsigned long long)address, (unsigned long long)*bytes, (unsigned long long)end,
                          (unsigned long long)(end - address));
            *bytes = end - address;
        }
        *base = address;
        return TRUE;
    }
    {
        static LONG logged;
        InterlockedIncrement(&guest_buffers_empty);
        if (InterlockedIncrement(&logged) <= 8)
            rebox_log("d3d12_xs: a guest buffer at %#llx+%#llx isn't GPU memory and can't be made so: read as empty",
                      (unsigned long long)*base, (unsigned long long)check);
    }
    return FALSE;
}

/* The length a guest buffer table gives a V# at address of bytes (0: read as empty), for tests. */
UINT64 WINAPI ReboxGuestBufferLength(UINT64 address, UINT64 bytes)
{
    return guest_buffer_mappable(&address, &bytes) ? bytes : 0;
}

/* The same for the n-th V# the code loads (s_load_dwordx4). */
static UINT64 late_buffer_source(const struct rebox_deferred_shader *h, const struct rebox_compile_in *in, UINT n)
{
    return late_load_source(h, in, n, 2);
}

static UINT64 late_load_source(const struct rebox_deferred_shader *h, const struct rebox_compile_in *in, UINT n,
                               UINT32 opcode)
{
    BOOL WINAPI ReboxShadowRead(UINT64 address, void *out, SIZE_T size);
    const UINT32 *code = (const UINT32 *)((const BYTE *)h + h->code_offset);

    /* V#s: first only the slots that hold no V# now (AnyPS5 gives those empty, the rest as they
     * are), then any. */
    for (UINT pass = opcode == 2 ? 0 : 1; pass < 2; pass++)
    {
        UINT seen = 0;
        for (UINT32 i = 0; i + 1 < h->code_dwords; i++)
        {
            UINT32 w0 = code[i], w1 = code[i + 1], sbase, op, words[8] = { 0 };
            UINT64 address;
            if (w0 >> 26 != 0x3d) /* SMEM */
                continue;
            op = (w0 >> 18) & 0xff;
            sbase = (w0 & 0x3f) * 2;
            i++; /* SMEM is two dwords */
            if (op != opcode || sbase < table_sgprs(h) || sbase >= table_sgprs(h) + 8 || !in->user_sgprs[sbase]
                    || (w1 >> 25) != 0x7d /* no SOFFSET */)
                continue;
            address = (UINT64)in->user_sgprs[sbase] + (w1 & 0x1fffff);
            /* In a shadow heap, whatever it holds now: AnyPS5 may have read it empty a moment before. */
            if (ReboxShadowRead(address, words, sizeof(words)) && (pass || !slot_holds_descriptor(address, 4))
                    && seen++ == n)
                return address;
        }
    }
    return 0;
}

/* REBOX_CONSOLE_COMPUTE_BISECT=VALUE/BITS: only the compute programs whose code address hashes
 * to VALUE in its low BITS bits run; the rest are skipped (their dispatch dropped). Each program run
 * is logged once. Halving the set run by run finds a program that faults the GPU (The Matrix
 * Awakens: a GPUVM fault at address 0 with console compute on). */
static BOOL compute_bisect_skips(const struct rebox_deferred_shader *h, const struct rebox_compile_in *in)
{
    UINT64 code_address = h->code_address;
    static int state = -1; /* -1 unread, 0 off, 1 on */
    static UINT64 value, mask;
    UINT64 hv = hash_bytes(0xcbf29ce484222325ull, &code_address, sizeof(code_address));

    if (state < 0)
    {
        char v[64];
        unsigned long long a = 0, b = 0;
        DWORD n = GetEnvironmentVariableA("REBOX_CONSOLE_COMPUTE_BISECT", v, sizeof(v));
        state = n && n < sizeof(v) && sscanf(v, "%llu/%llu", &a, &b) == 2 && b < 64;
        mask = state ? (1ull << b) - 1 : 0;
        value = a & mask;
        if (state)
            rebox_log("d3d12_xs: console compute bisection: programs with hash & %#llx == %#llx run", (unsigned long long)mask,
                         (unsigned long long)value);
    }
    if (!state)
        return FALSE;
    if ((hv & mask) != value)
        return TRUE;
    {
        static UINT64 seen[256];
        static LONG count;
        BOOL known = FALSE;
        for (LONG i = 0; i < count && i < 256; i++)
            known |= seen[i] == code_address;
        if (!known && count < 256)
        {
            /* What it's given and what it reads: its user SGPRs, the tables the layout analysis
             * finds, and its code in %TEMP%\rebox-compute-<address>.bin to disassemble. */
            char line[400], tables[128], path[MAX_PATH];
            int at = 0, tat = 0;
            struct sa5_layout layout;
            static struct sa5_slot slots[64];
            char ignored[64];
            HANDLE f;
            seen[count++] = code_address;
            for (UINT32 i = 0; in->user_sgprs && i < in->user_sgpr_count && i < 32 && at < 380; i++)
                at += snprintf(line + at, sizeof(line) - at, " %08x", in->user_sgprs[i]);
            line[at] = 0;
            tables[0] = 0;
            InitOnceExecuteOnce(&sa5.once, load_sa5, NULL, NULL);
            memset(&layout, 0, sizeof(layout));
            if (sa5.analyse && !sa5.analyse((const UINT32 *)((const BYTE *)h + h->code_offset), h->code_dwords, user_sgprs(h, FALSE),
                                            &layout, slots, ARRAYSIZE(slots), ignored, sizeof(ignored)))
                for (UINT32 t = 0; t < layout.table_count && t < 16 && tat < 110; t++)
                    tat += snprintf(tables + tat, sizeof(tables) - tat, " s%u", layout.table_sgprs[t]);
            rebox_log("d3d12_xs: console compute bisection runs program %#llx (hash %#llx, %u dwords); tables through%s; "
                         "user SGPRs:%s", (unsigned long long)code_address, (unsigned long long)hv, h->code_dwords,
                         tat ? tables : " (none found)", line);
            GetTempPathA(sizeof(path) - 64, path);
            snprintf(path + strlen(path), 64, "rebox-compute-%llx.bin", (unsigned long long)code_address);
            if ((f = CreateFileA(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, 0, NULL)) != INVALID_HANDLE_VALUE)
            {
                DWORD written;
                WriteFile(f, (const BYTE *)h + h->code_offset, h->code_dwords * 4u, &written, NULL);
                CloseHandle(f);
            }
        }
    }
    return FALSE;
}

/* Background compiles. A program's first compile (AnyPS5's translation, hundreds of milliseconds
 * for Unreal 5's large shaders) used to run in the title's render thread at the dispatch or draw;
 * A UE5 sample's game thread gave up waiting for it (120 s). Now the first call for a program
 * queues the compile for a worker thread and fails (the dispatch is skipped, the draw uses the
 * fallback); once the worker has compiled it once, AnyPS5's memo and the caches here make later
 * calls quick, and they compile in the calling thread as before. REBOX_CONSOLE_ASYNC=0 turns it
 * off. */
static BOOL async_compiles(void)
{
    char v[8];
    DWORD n = GetEnvironmentVariableA("REBOX_CONSOLE_ASYNC", v, sizeof(v)); /* each call: tests change it */
    return !(n && n < sizeof(v) && v[0] == '0');
}

enum program_state { PROGRAM_COLD, PROGRAM_PENDING, PROGRAM_WARM, PROGRAM_FAILED };
#define PROGRAMS 16384
static struct { UINT64 key; LONG state; UINT calls, retries, uses; } programs[PROGRAMS];
static SRWLOCK programs_lock = SRWLOCK_INIT;

/* The state of a program (by its blob's hash), claiming it as pending if it was cold; *uses counts
 * the calls for it, this one included (0 if the table had no room; uses NULL: only look). */
static enum program_state program_claim(UINT64 key, BOOL claim, UINT *uses)
{
    UINT slot = (UINT)(key % PROGRAMS);
    enum program_state state = PROGRAM_COLD;

    if (uses)
        *uses = 0;
    AcquireSRWLockExclusive(&programs_lock);
    for (UINT i = 0; i < 64; i++)
    {
        UINT at = (slot + i) % PROGRAMS;
        if (programs[at].state != PROGRAM_COLD && programs[at].key == key)
        {
            state = programs[at].state;
            if (uses && programs[at].uses != ~0u)
                programs[at].uses++;
            if (uses)
                *uses = programs[at].uses;
            break;
        }
        if (programs[at].state == PROGRAM_COLD)
        {
            if (claim)
            {
                programs[at].key = key;
                programs[at].state = PROGRAM_PENDING;
                programs[at].calls = programs[at].retries = 0;
                programs[at].uses = 1;
                if (uses)
                    *uses = 1;
            }
            break;
        }
    }
    ReleaseSRWLockExclusive(&programs_lock);
    return state;
}

static void program_set(UINT64 key, enum program_state state)
{
    UINT slot = (UINT)(key % PROGRAMS);
    AcquireSRWLockExclusive(&programs_lock);
    for (UINT i = 0; i < 64; i++)
    {
        UINT at = (slot + i) % PROGRAMS;
        if (programs[at].state != PROGRAM_COLD && programs[at].key == key)
        {
            programs[at].state = state;
            break;
        }
    }
    ReleaseSRWLockExclusive(&programs_lock);
}

/* A program whose background compile failed: that compile ran on a worker, at a world's start
 * minutes after the call that queued it, with that call's user SGPRs pointing at constants and
 * tables the title had since written over (T#s that are floats, a UAV slot holding an SRV), and the
 * program was then refused at every call for the rest of the run. Its calls are counted, and at its
 * 64th failing call, then its 128th, 256th and so on (at most PROGRAM_RETRIES times), it's compiled
 * again in the background with that call's inputs; the caller still skips. Returns whether this
 * call is the retry (the program is pending again). */
#define PROGRAM_RETRIES 10
static BOOL program_retry(UINT64 key, UINT *retry)
{
    UINT slot = (UINT)(key % PROGRAMS);
    BOOL due = FALSE;

    AcquireSRWLockExclusive(&programs_lock);
    for (UINT i = 0; i < 64; i++)
    {
        UINT at = (slot + i) % PROGRAMS;
        if (programs[at].state == PROGRAM_COLD)
            break;
        if (programs[at].key != key)
            continue;
        if (programs[at].state == PROGRAM_FAILED && programs[at].retries < PROGRAM_RETRIES
                && ++programs[at].calls >= 64u << programs[at].retries)
        {
            programs[at].calls = 0;
            *retry = ++programs[at].retries;
            programs[at].state = PROGRAM_PENDING;
            due = TRUE;
        }
        break;
    }
    ReleaseSRWLockExclusive(&programs_lock);
    return due;
}

struct compile_job
{
    struct compile_job *next;
    UINT retry; /* 0, or which retry of a failed program this is (program_retry) */
    UINT64 key;
    struct rebox_compile_in in;
    UINT32 user_sgprs[64];
    BYTE shader[];
};
static struct compile_job *jobs_head, *jobs_tail;
static SRWLOCK jobs_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE jobs_ready = CONDITION_VARIABLE_INIT;
static LONG jobs_queued, jobs_done;

static HRESULT compile_now(const struct rebox_compile_in *in, struct rebox_compile_out *out, BOOL background);

static DWORD WINAPI compile_worker(void *unused)
{
    for (;;)
    {
        struct compile_job *job;
        struct rebox_compile_out out;
        HRESULT hr;
        LONG done;

        AcquireSRWLockExclusive(&jobs_lock);
        while (!jobs_head)
            SleepConditionVariableSRW(&jobs_ready, &jobs_lock, INFINITE, 0);
        job = jobs_head;
        if (!(jobs_head = job->next))
            jobs_tail = NULL;
        ReleaseSRWLockExclusive(&jobs_lock);

        memset(&out, 0, sizeof(out));
        hr = compile_now(&job->in, &out, TRUE);
        program_set(job->key, SUCCEEDED(hr) ? PROGRAM_WARM : PROGRAM_FAILED);
        if (job->retry)
        {
            static volatile LONG retried, recovered;
            LONG n = InterlockedIncrement(&retried), r = SUCCEEDED(hr) ? InterlockedIncrement(&recovered) : recovered;
            if (SUCCEEDED(hr) || !(n & (n - 1)))
                rebox_log("d3d12_xs: console shaders: program %016llx %s at retry %u (%ld of %ld retries of failed "
                          "programs compiled)", (unsigned long long)job->key, SUCCEEDED(hr) ? "compiled" : "failed again",
                          job->retry, r, n);
        }
        done = InterlockedIncrement(&jobs_done);
        if (!(done & (done - 1)))
            rebox_log("d3d12_xs: console shaders: %ld background compiles done of %ld queued", done, jobs_queued);
        free(job);
    }
    return 0;
}

static void queue_compile(UINT64 key, const struct rebox_compile_in *in, BOOL front, UINT retry)
{
    static INIT_ONCE started = INIT_ONCE_STATIC_INIT;
    BOOL pending;
    struct compile_job *job = malloc(sizeof(*job) + in->shader_size);

    if (!job)
    {
        program_set(key, PROGRAM_COLD);
        return;
    }
    memset(job, 0, sizeof(*job));
    job->key = key;
    job->retry = retry;
    job->in = *in;
    memcpy(job->shader, in->shader, in->shader_size);
    job->in.shader = job->shader;
    job->in.user_sgpr_count = min(in->user_sgpr_count, (UINT32)ARRAYSIZE(job->user_sgprs));
    if (in->user_sgprs)
        memcpy(job->user_sgprs, in->user_sgprs, job->in.user_sgpr_count * 4u);
    job->in.user_sgprs = job->user_sgprs;
    /* REBOX_CONSOLE_COMPILE_THREADS workers (default half the CPUs, at most 8): a world brings
     * a thousand new shaders at once, and one worker left its draws skipped for minutes. */
    if (InitOnceBeginInitialize(&started, 0, &pending, NULL) && pending)
    {
        SYSTEM_INFO si;
        char v[8];
        DWORD n = GetEnvironmentVariableA("REBOX_CONSOLE_COMPILE_THREADS", v, sizeof(v));
        UINT workers;
        GetSystemInfo(&si);
        workers = n && n < sizeof(v) && atoi(v) > 0 ? (UINT)atoi(v) : min(max(si.dwNumberOfProcessors / 2, 1u), 8u);
        for (UINT w = 0; w < workers; w++)
        {
            HANDLE t = CreateThread(NULL, 0, compile_worker, NULL, 0, NULL);
            if (t)
                CloseHandle(t);
        }
        rebox_log("d3d12_xs: console shaders compile in the background on %u threads", workers);
        InitOnceComplete(&started, 0, NULL);
    }
    InterlockedIncrement(&jobs_queued);
    AcquireSRWLockExclusive(&jobs_lock);
    if (front)
    {
        if (!(job->next = jobs_head))
            jobs_tail = job;
        jobs_head = job;
    }
    else
    {
        if (jobs_tail)
            jobs_tail->next = job;
        else
            jobs_head = job;
        jobs_tail = job;
    }
    ReleaseSRWLockExclusive(&jobs_lock);
    WakeConditionVariable(&jobs_ready);
}

/* Which console stages draw into the display-sized targets (proxy.c brackets those draws): each
 * distinct (target, vertex stage, pixel stage, count, rect list) with how often, logged at the first
 * presents, so a draw seen in a frame can be named for REBOX_GRAPHICS_DEBUG=magenta:<id>. */
static struct display_draw { int target; UINT count; BOOL rect, indexed; UINT64 vs, ps; LONG times; } display_draws[160];
static UINT display_draw_count;
static SRWLOCK display_draw_lock = SRWLOCK_INIT;
static __thread struct display_draw current_draw;
static __thread BOOL in_display_draw;

void console_display_draw_begin(int target, UINT count, BOOL rect, BOOL indexed)
{
    memset(&current_draw, 0, sizeof(current_draw));
    current_draw.target = target;
    current_draw.count = count;
    current_draw.rect = rect;
    current_draw.indexed = indexed;
    in_display_draw = TRUE;
}

void console_display_draw_end(void)
{
    UINT i;

    if (!in_display_draw)
        return;
    in_display_draw = FALSE;
    AcquireSRWLockExclusive(&display_draw_lock);
    for (i = 0; i < display_draw_count; i++)
        if (display_draws[i].target == current_draw.target && display_draws[i].count == current_draw.count
                && display_draws[i].rect == current_draw.rect && display_draws[i].indexed == current_draw.indexed
                && display_draws[i].vs == current_draw.vs && display_draws[i].ps == current_draw.ps)
            break;
    if (i == display_draw_count && i < ARRAYSIZE(display_draws))
        display_draws[display_draw_count++] = current_draw;
    if (i < display_draw_count)
        display_draws[i].times++;
    ReleaseSRWLockExclusive(&display_draw_lock);
}

/* What each ReboxConsoleCompile call came to, by stage kind, since the last census: a draw or
 * dispatch whose stage isn't "ok" is skipped (vkd3d-proton draws nothing, or logs "Failed to update
 * compute state"). The census only names DrawInstanced and DrawIndexedInstanced; these count every
 * draw, mesh draw, ExecuteIndirect and dispatch. */
enum { OUTCOME_OK, OUTCOME_BUDGET, OUTCOME_QUEUED, OUTCOME_WAIT, OUTCOME_PROGRAM_FAILED, OUTCOME_FAILED, OUTCOMES };
static const char *const outcome_names[OUTCOMES] = { "ok", "past the frame's budget", "compiling in the background",
                                                     "past their wait", "program failed before", "failed" };
static volatile LONG outcomes[4][OUTCOMES]; /* [compute, vertex, pixel, mesh] */
static volatile LONG standins; /* pixel stand-ins given (REBOX_STANDIN_PIXEL) */
static __thread UINT call_outcome;

static UINT outcome_kind(const struct rebox_deferred_shader *h, SIZE_T size)
{
    if (!h || size < sizeof(*h) || h->version != 2)
        return 0;
    return h->stage == 0x10 ? 2 : h->stage == 0x80 ? 3 : 1;
}

static void outcomes_log(void)
{
    static const char *const kinds[4] = { "compute", "vertex", "pixel", "mesh" };
    char line[2048]; /* at most about 1100 */
    int at = 0;
    LONG total = 0;
    for (UINT k = 0; k < 4; k++)
    {
        int start = at;
        at += snprintf(line + at, sizeof(line) - at, "%s%s:", at ? "; " : "", kinds[k]);
        for (UINT o = 0; o < OUTCOMES; o++)
        {
            LONG n = InterlockedExchange(&outcomes[k][o], 0);
            total += n;
            if (n && at < (int)sizeof(line) - 64)
                at += snprintf(line + at, sizeof(line) - at, " %s %ld,", outcome_names[o], n);
        }
        if (line[at - 1] == ',')
            line[--at] = 0;
        else
            line[at = start] = 0; /* none of this kind */
    }
    {
        LONG n = InterlockedExchange(&standins, 0);
        if (n && at < (int)sizeof(line) - 64)
            at += snprintf(line + at, sizeof(line) - at, "; pixel stand-ins (REBOX_STANDIN_PIXEL) %ld", n);
    }
    if (total)
        rebox_log("d3d12_xs:   console compiles since the last census: %s", line);
    {
        LONG empty = InterlockedExchange(&guest_buffers_empty, 0);
        if (empty)
            rebox_log("d3d12_xs:   guest buffers in title memory read as empty since the last census "
                      "(not GPU memory, and couldn't be made so): %ld", empty);
    }
    gpu_memory_adopt_failures_log();
}

void console_display_draws_log(void)
{
    outcomes_log();
    AcquireSRWLockExclusive(&display_draw_lock);
    for (UINT i = 0; i < display_draw_count; i++)
    {
        /* With the programs' names where known (fast.c), so a census reads as passes. */
        const char *vs_name = fast_id_name(display_draws[i].vs), *ps_name = fast_id_name(display_draws[i].ps);
        rebox_log("d3d12_xs:   %s %d: %ld %s of %u %s, vertex stage %016llx, pixel stage %016llx%s%s%s%s%s",
                  display_draws[i].target < 0 ? "other target" : "display target", display_draws[i].target, display_draws[i].times, display_draws[i].indexed ? "indexed draws" : "draws",
                  display_draws[i].count, display_draws[i].rect ? "(rect list)" : "",
                  (unsigned long long)display_draws[i].vs, (unsigned long long)display_draws[i].ps,
                  vs_name || ps_name ? " (" : "", vs_name ? vs_name : "", vs_name && ps_name ? ", " : "",
                  ps_name ? ps_name : "", vs_name || ps_name ? ")" : "");
    }
    display_draw_count = 0;
    ReleaseSRWLockExclusive(&display_draw_lock);
}

/* A deferred console stage's id (as the logs name stages), or 0. */
UINT64 console_stage_id(const void *shader, SIZE_T size)
{
    const struct rebox_deferred_shader *h = shader;
    if (!h || size < sizeof(*h) || h->version != 2 || h->code_offset > size || h->code_dwords * 4ull > size - h->code_offset)
        return 0;
    return hash_bytes(hash_bytes(0xcbf29ce484222325ull, &h->stage, sizeof(h->stage)), (const BYTE *)h + h->code_offset,
                      h->code_dwords * 4u);
}

/* Whether a stage is REBOX_WATCH_STAGE's. */
BOOL console_stage_watched(UINT64 id)
{
    /* Read once: bindings ask several times for every descriptor of every draw. */
    static volatile LONG state; /* 0 unread, 1 none, 2 watching */
    static char watch[17];
    char text[24];
    if (!state)
    {
        const char *v = getenv("REBOX_WATCH_STAGE");
        if (v && strlen(v) >= 16)
            memcpy(watch, v, 16);
        InterlockedExchange(&state, v && strlen(v) >= 16 ? 2 : 1);
    }
    if (state != 2 || !id)
        return FALSE;
    snprintf(text, sizeof(text), "%016llx", (unsigned long long)id);
    return !strncmp(watch, text, 16);
}

/* The vertex stage of the calling thread's last bracketed draw (console_display_draw_begin). */
UINT64 console_last_draw_vs(void)
{
    return current_draw.vs;
}

/* A frame's compile budget. Compiles of known programs run in the title's recording threads (most
 * take well under a millisecond, AnyPS5's memo and the module cache hitting); a scene that brings
 * in many new variants made frames take seconds, and the title's GPU watchdog fired. Once the
 * compiles in the title's threads since the last present have taken REBOX_FRAME_COMPILE_BUDGET_MS
 * (default 150; 0: no budget), graphics stages are skipped for the rest of the frame, their draws
 * left out as while a program first compiles; the next frame goes on from there (VIctory, GPL-3.0,
 * facts: a 500 ms budget a frame for pipelines, after which draws are dropped). Compute dispatches
 * aren't skipped: what they write can be read back by the title.
 *
 * Only slow calls count against it: those that took REBOX_FRAME_COMPILE_SLOW_MS (default 10; 0
 * counts every call) or more, a new variant or a wait that ran out. Every call used to count, and in
 * a world those that hit the memo still take 1 to 3 ms each (most of it making bindings): a frame's
 * first hundred or so dispatches spent the budget, and every later scene draw, mesh draw and
 * ExecuteIndirect of every frame was skipped (G-buffer, shadows, lighting, tone mapping), while the
 * display-sized targets' draws, exempt, drew the UI over black. */
static volatile LONG compile_frame;
static volatile LONG64 frame_compile_ticks;
static volatile LONG budget_frame, budget_skips;

void console_frame_presented(void)
{
    InterlockedIncrement(&compile_frame);
}

static LONG64 slow_compile_ticks(void)
{
    static LONG64 slow = -1;
    if (slow < 0)
    {
        char v[16];
        DWORD n = GetEnvironmentVariableA("REBOX_FRAME_COMPILE_SLOW_MS", v, sizeof(v));
        LONG ms = n && n < sizeof(v) ? atol(v) : 10;
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        slow = ms > 0 ? f.QuadPart * ms / 1000 : 0;
    }
    return slow;
}

/* REBOX_FRAME_TRACE=<present>[-<present>][,...] (at most 8 ranges): during those presents (counted
 * as the census counts them), a line for every ReboxConsoleCompile call, in order, with the stage,
 * what came of it and the draw, ExecuteIndirect or dispatch that asked (proxy.c names those, with
 * its first render target), to set a frame's passes beside a reference trace. */
static LONG trace_ranges[8][2];
static UINT trace_range_count;
static __thread struct { const char *op; UINT a, b, c, width, height, format; } trace_op;

static BOOL CALLBACK trace_parse(INIT_ONCE *once, void *param, void **ctx)
{
    const char *v = getenv("REBOX_FRAME_TRACE");
    while (v && *v && trace_range_count < ARRAYSIZE(trace_ranges))
    {
        char *end;
        LONG from = strtol(v, &end, 10), to = from;
        if (end == v)
            break;
        if (*end == '-')
            to = strtol(end + 1, &end, 10);
        trace_ranges[trace_range_count][0] = from;
        trace_ranges[trace_range_count++][1] = to;
        v = *end == ',' ? end + 1 : end;
        if (*end != ',')
            break;
    }
    if (trace_range_count)
        rebox_log("d3d12_xs: REBOX_FRAME_TRACE: tracing %u range(s) of presents from %ld", trace_range_count,
                  trace_ranges[0][0]);
    return TRUE;
}

BOOL console_trace_active(void)
{
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    LONG frame = compile_frame;
    InitOnceExecuteOnce(&once, trace_parse, NULL, NULL);
    for (UINT i = 0; i < trace_range_count; i++)
        if (frame >= trace_ranges[i][0] && frame <= trace_ranges[i][1])
            return TRUE;
    return FALSE;
}

void console_trace_op(const char *op, UINT a, UINT b, UINT c, UINT width, UINT height, UINT format)
{
    trace_op.op = op;
    trace_op.a = a;
    trace_op.b = b;
    trace_op.c = c;
    trace_op.width = width;
    trace_op.height = height;
    trace_op.format = format;
}

static void trace_call(const struct rebox_compile_in *in, HRESULT hr)
{
    static const char *const kinds[4] = { "compute", "vertex", "pixel", "mesh" };
    static volatile LONG seq;
    const struct rebox_deferred_shader *h = in ? in->shader : NULL;
    UINT64 code = 0, stage = 0;
    char message[512] = "", op[128] = "";

    if (h && in->shader_size >= sizeof(*h) && h->code_offset <= in->shader_size
            && h->code_dwords * 4ull <= in->shader_size - h->code_offset)
    {
        code = hash_bytes(0xcbf29ce484222325ull, (const BYTE *)h + h->code_offset, h->code_dwords * 4u);
        stage = console_stage_id(in->shader, in->shader_size);
    }
    if (FAILED(hr))
        ReboxConsoleLastResult(message, sizeof(message));
    if (trace_op.op)
        snprintf(op, sizeof(op), " | %s %u %u %u rt %ux%u f%u", trace_op.op, trace_op.a, trace_op.b, trace_op.c,
                 trace_op.width, trace_op.height, trace_op.format);
    {
        const char *name = fast_id_name(stage);
        if (!name)
            name = fast_id_name(code);
        rebox_log("d3d12_xs: trace %ld #%ld [%04lx] %s %016llx stage %016llx%s%s: %s%s%s%s%s", compile_frame,
                  InterlockedIncrement(&seq), GetCurrentThreadId(), kinds[outcome_kind(h, in ? in->shader_size : 0)],
                  (unsigned long long)code, (unsigned long long)stage, name ? " " : "", name ? name : "",
                  outcome_names[min(call_outcome, OUTCOMES - 1)], *message ? " (" : "", message, *message ? ")" : "", op);
    }
}

static LONG64 frame_budget_ticks(void)
{
    static LONG64 budget = -1;
    if (budget < 0)
    {
        char v[16];
        DWORD n = GetEnvironmentVariableA("REBOX_FRAME_COMPILE_BUDGET_MS", v, sizeof(v));
        LONG ms = n && n < sizeof(v) ? atol(v) : 150;
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        budget = ms > 0 ? f.QuadPart * ms / 1000 : 0;
    }
    return budget;
}

/* Compiles of known programs from the title's recording threads run on a pool of their own, and the
 * caller waits a bounded time: almost all are AnyPS5 memo hits taking well under a millisecond, but
 * a new variant of a large program can take tens of seconds, and the title's render thread, waiting
 * for that recording job, stopped presenting until its GPU watchdog fired. Past the wait the draw or
 * dispatch is skipped and the compile goes on; the next call finds it in AnyPS5's memo.
 * REBOX_INLINE_COMPILE_WAIT_MS (default 50; compute 250; 0: compile in the caller as before). */
struct bounded_job
{
    struct bounded_job *next;
    volatile LONG refs;
    HANDLE done;
    HRESULT hr;
    UINT64 program; /* a program's first compile (first_call_compile): its state is set when done */
    struct rebox_compile_out out;
    struct rebox_compile_in in;
    UINT32 sgprs[64];
    BYTE shader[];
};
static struct bounded_job *bounded_head, *bounded_tail;
static SRWLOCK bounded_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE bounded_ready = CONDITION_VARIABLE_INIT;

static void bounded_release(struct bounded_job *job)
{
    if (!InterlockedDecrement(&job->refs))
    {
        CloseHandle(job->done);
        free(job);
    }
}

static DWORD WINAPI bounded_worker(void *unused)
{
    for (;;)
    {
        struct bounded_job *job;
        AcquireSRWLockExclusive(&bounded_lock);
        while (!bounded_head)
            SleepConditionVariableSRW(&bounded_ready, &bounded_lock, INFINITE, 0);
        job = bounded_head;
        if (!(bounded_head = job->next))
            bounded_tail = NULL;
        ReleaseSRWLockExclusive(&bounded_lock);
        job->hr = compile_now(&job->in, &job->out, FALSE);
        if (job->program)
            program_set(job->program, SUCCEEDED(job->hr) ? PROGRAM_WARM : PROGRAM_FAILED);
        SetEvent(job->done);
        bounded_release(job);
    }
    return 0;
}

static DWORD inline_wait_ms(BOOL graphics)
{
    static LONG ms = -1;
    if (ms < 0)
    {
        char v[16];
        DWORD n = GetEnvironmentVariableA("REBOX_INLINE_COMPILE_WAIT_MS", v, sizeof(v));
        ms = n && n < sizeof(v) ? atol(v) : 50;
    }
    return !ms ? 0 : graphics ? (DWORD)ms : (DWORD)max(ms, 250);
}

/* Compiles in on the pool, waiting at most wait ms (0: in the caller); program, if not 0, is a
 * program claimed as pending whose state the compile sets. */
static HRESULT bounded_compile_for(const struct rebox_compile_in *in, struct rebox_compile_out *out, BOOL graphics,
                                   DWORD wait, UINT64 program)
{
    static INIT_ONCE started = INIT_ONCE_STATIC_INIT;
    static LONG timeouts;
    struct bounded_job *job;
    BOOL pending;
    HRESULT hr;

    if (!wait || !(job = calloc(1, sizeof(*job) + in->shader_size)))
    {
        hr = compile_now(in, out, FALSE);
        if (program)
            program_set(program, SUCCEEDED(hr) ? PROGRAM_WARM : PROGRAM_FAILED);
        return hr;
    }
    if (!(job->done = CreateEventA(NULL, TRUE, FALSE, NULL)))
    {
        free(job);
        hr = compile_now(in, out, FALSE);
        if (program)
            program_set(program, SUCCEEDED(hr) ? PROGRAM_WARM : PROGRAM_FAILED);
        return hr;
    }
    job->program = program;
    job->refs = 2;
    job->in = *in;
    memcpy(job->shader, in->shader, in->shader_size);
    job->in.shader = job->shader;
    job->in.user_sgpr_count = min(in->user_sgpr_count, (UINT32)ARRAYSIZE(job->sgprs));
    if (in->user_sgprs)
        memcpy(job->sgprs, in->user_sgprs, job->in.user_sgpr_count * 4u);
    job->in.user_sgprs = job->sgprs;
    if (InitOnceBeginInitialize(&started, 0, &pending, NULL) && pending)
    {
        for (UINT w = 0; w < 8; w++)
        {
            HANDLE t = CreateThread(NULL, 0, bounded_worker, NULL, 0, NULL);
            if (t)
                CloseHandle(t);
        }
        InitOnceComplete(&started, 0, NULL);
    }
    AcquireSRWLockExclusive(&bounded_lock);
    if (bounded_tail)
        bounded_tail->next = job;
    else
        bounded_head = job;
    bounded_tail = job;
    ReleaseSRWLockExclusive(&bounded_lock);
    WakeConditionVariable(&bounded_ready);
    if (WaitForSingleObject(job->done, wait) == WAIT_OBJECT_0)
    {
        hr = job->hr;
        *out = job->out;
    }
    else
    {
        LONG n = InterlockedIncrement(&timeouts);
        if (!(n & (n - 1)))
            rebox_log("d3d12_xs: console shaders: %ld %s compiles took past their wait and went on without "
                      "their draw or dispatch", n, graphics ? "graphics" : "compute");
        set_last(E_PENDING, "compiling past its wait", NULL);
        hr = E_PENDING;
    }
    bounded_release(job);
    return hr;
}

static HRESULT bounded_compile(const struct rebox_compile_in *in, struct rebox_compile_out *out, BOOL graphics)
{
    return bounded_compile_for(in, out, graphics, inline_wait_ms(graphics), 0);
}

/* First calls wait. A draw or dispatch skipped because its program is still compiling (its first
 * call queues the compile and is skipped; later ones past REBOX_INLINE_COMPILE_WAIT_MS or the frame's
 * budget are too) is only redrawn if the title draws it again. The world's first frame bakes
 * lighting tables with one-time full-screen passes, and one skipped stays undone: diffuse light was
 * gone for good (VIctory forks, GPL-3.0, facts: such passes get their own compile lane and a 20 s
 * wait). So a program's first REBOX_FIRST_CALLS_WAIT calls (default 4; 0: none) wait for their
 * compile, at most REBOX_FIRST_CALL_WAIT_MS each (default 20000), and so do a full-screen pass's
 * (proxy.c: a non-indexed draw of 3 or 4 vertices and one instance) first
 * REBOX_SCREEN_PASS_WAIT_CALLS (default 16). Later calls are skipped while compiling, as before, so
 * the world's repeated draws don't stall frames. Waits other than a full-screen pass's stop for the
 * rest of a frame once that frame has waited REBOX_FIRST_CALLS_FRAME_MS (default 3000) in all: a
 * world brings a thousand new programs at once, and the title's watchdog fired when its render
 * thread stopped presenting for long. Every wait of a second or more is logged. vkd3d-proton
 * builds the pipeline of such a call at once instead of on its workers (ReboxConsoleMustDraw). */
static __thread BOOL screen_pass_hint, must_draw;
static volatile LONG first_wait_frame;
static volatile LONG64 first_wait_ticks;

void console_screen_pass(BOOL on)
{
    screen_pass_hint = on;
}

/* Whether a call since the last query waited for its compile (patch 0030's pipeline lookup asks,
 * after the stages' compiles, and builds that pipeline at once). */
BOOL WINAPI ReboxConsoleMustDraw(void)
{
    BOOL was = must_draw;
    must_draw = FALSE;
    return was;
}

static LONG env_count(const char *name, LONG fallback)
{
    char v[16];
    DWORD n = GetEnvironmentVariableA(name, v, sizeof(v));
    return n && n < sizeof(v) ? atol(v) : fallback;
}

/* How long the uses-th call of a program may wait for its compile (0: as any other call). */
static DWORD first_call_wait(UINT uses, BOOL *screen)
{
    static LONG first = -1, screens = -1, wait_ms = -1, frame_ms = -1;
    LONG frame = compile_frame;

    if (first < 0)
    {
        first = env_count("REBOX_FIRST_CALLS_WAIT", 4);
        screens = env_count("REBOX_SCREEN_PASS_WAIT_CALLS", 16);
        wait_ms = env_count("REBOX_FIRST_CALL_WAIT_MS", 20000);
        frame_ms = env_count("REBOX_FIRST_CALLS_FRAME_MS", 3000);
    }
    *screen = FALSE;
    if (!uses || wait_ms <= 0)
        return 0;
    if (screen_pass_hint && uses <= (UINT)max(screens, 0))
    {
        *screen = TRUE;
        return (DWORD)wait_ms;
    }
    if (uses > (UINT)max(first, 0))
        return 0;
    if (InterlockedExchange(&first_wait_frame, frame) != frame)
        InterlockedExchange64(&first_wait_ticks, 0);
    if (frame_ms > 0 && first_wait_ticks * 1000.0 >= frame_ms * qpc_frequency())
        return 0;
    return (DWORD)wait_ms;
}

static void first_call_waited(UINT64 key, UINT uses, BOOL screen, BOOL graphics, LONG64 started, HRESULT hr)
{
    static volatile LONG logged;
    LONG64 ticks = ticks_now() - started;
    double ms = ticks * 1000.0 / qpc_frequency();
    LONG n;

    if (!screen)
        InterlockedAdd64(&first_wait_ticks, ticks);
    if (ms < 1000.0)
        return;
    n = InterlockedIncrement(&logged);
    if (n <= 256 || !(n & (n - 1)))
        rebox_log("d3d12_xs: console shaders: call %u of %s program %016llx%s waited %.0f ms for its compile: %s "
                  "(%ld such waits)", uses, graphics ? "graphics" : "compute", (unsigned long long)key,
                  screen ? " (a full-screen pass)" : "", ms,
                  SUCCEEDED(hr) ? "compiled" : hr == E_PENDING ? "still compiling, skipped" : "failed", n);
}

/* A first call's compile, waiting at most wait ms: the program's own first compile if it was cold
 * (claimed), else for its background compile to end and then for this call's variant. */
static HRESULT first_call_compile(UINT64 key, enum program_state state, const struct rebox_compile_in *in,
                                  struct rebox_compile_out *out, BOOL graphics, DWORD wait)
{
    LONG64 started = ticks_now(), deadline = started + (LONG64)(wait * qpc_frequency() / 1000);
    HRESULT hr;

    if (state == PROGRAM_COLD)
        return bounded_compile_for(in, out, graphics, wait, key);
    while (state == PROGRAM_PENDING && ticks_now() < deadline)
    {
        Sleep(2);
        state = program_claim(key, FALSE, NULL);
    }
    if (state == PROGRAM_PENDING)
    {
        set_last(E_PENDING, "compiling in the background past its first-call wait", NULL);
        return E_PENDING;
    }
    if (state == PROGRAM_FAILED)
    {
        set_last(E_FAIL, "its background compile failed", NULL);
        return E_FAIL;
    }
    hr = bounded_compile_for(in, out, graphics, (DWORD)max((deadline - ticks_now()) * 1000.0 / qpc_frequency(), 50.0), 0);
    return hr;
}

static HRESULT compile_within_budget(const struct rebox_compile_in *in, struct rebox_compile_out *out)
{
    const struct rebox_deferred_shader *h = in->shader;
    BOOL graphics = in->shader_size >= sizeof(*h) && h->version == 2;
    LONG64 budget = frame_budget_ticks(), started, elapsed;
    LONG frame = compile_frame;
    HRESULT hr;

    if (InterlockedExchange(&budget_frame, frame) != frame)
        InterlockedExchange64(&frame_compile_ticks, 0);
    /* Draws into display-sized targets (the UI, text, the final composite) are never left out:
     * they come late in a frame, after the scene's draws have spent its budget, and an intro's
     * title card text was skipped frame after frame. They still only wait a bounded time. */
    if (graphics && budget && frame_compile_ticks >= budget && !(in_display_draw && current_draw.target >= 0))
    {
        LONG n = InterlockedIncrement(&budget_skips);
        if (!(n & (n - 1)) && n >= 64)
            rebox_log("d3d12_xs: console shaders: %ld graphics stages skipped past their frame's compile budget", n);
        set_last(E_PENDING, "past this frame's compile budget", NULL);
        call_outcome = OUTCOME_BUDGET;
        return E_PENDING;
    }
    started = ticks_now();
    /* Those draws also compile on their own thread, as long as it takes: on the shared workers
     * they queued behind a loading scene's compiles, overran the wait, and a loading screen's
     * layers went black for a frame whenever they needed a new variant. */
    if (in_display_draw && current_draw.target >= 0)
        hr = compile_now(in, out, FALSE);
    else
        hr = bounded_compile(in, out, graphics);
    call_outcome = SUCCEEDED(hr) ? OUTCOME_OK : hr == E_PENDING ? OUTCOME_WAIT : OUTCOME_FAILED;
    elapsed = ticks_now() - started;
    if (elapsed >= slow_compile_ticks())
        InterlockedAdd64(&frame_compile_ticks, elapsed);
    return hr;
}

static HRESULT console_compile(const struct rebox_compile_in *in, struct rebox_compile_out *out)
{
    enum program_state state;
    BOOL screen;
    DWORD wait;
    UINT64 key;
    UINT uses;

    if (in_display_draw && in && in->shader && in->shader_size >= sizeof(struct rebox_deferred_shader))
    {
        const struct rebox_deferred_shader *h = in->shader;
        if (h->version == 2 && (h->stage == 0x1 || h->stage == 0x10 || h->stage == 0x80) && h->code_offset <= in->shader_size
                && h->code_dwords * 4ull <= in->shader_size - h->code_offset)
        {
            UINT64 id = hash_bytes(hash_bytes(0xcbf29ce484222325ull, &h->stage, sizeof(h->stage)),
                                   (const BYTE *)h + h->code_offset, h->code_dwords * 4u);
            if (h->stage != 0x10)
                current_draw.vs = id;
            else
                current_draw.ps = id;
        }
    }

    if (!async_compiles() || !in || !in->shader || !in->shader_size || !out)
    {
        HRESULT hr = compile_now(in, out, FALSE);
        call_outcome = SUCCEEDED(hr) ? OUTCOME_OK : OUTCOME_FAILED;
        return hr;
    }
    key = hash_words(0xcbf29ce484222325ull, in->shader, in->shader_size);
    state = program_claim(key, TRUE, &uses);
    /* One of a program's first calls, or a full-screen pass's: waits for its compile. */
    if (state != PROGRAM_FAILED && (wait = first_call_wait(uses, &screen)))
    {
        const struct rebox_deferred_shader *h = in->shader;
        BOOL graphics = in->shader_size >= sizeof(*h) && h->version == 2;
        LONG64 started = ticks_now();
        HRESULT hr;

        if (state == PROGRAM_WARM && in_display_draw && current_draw.target >= 0)
            hr = compile_now(in, out, FALSE);
        else
            hr = first_call_compile(key, state, in, out, graphics, wait);
        first_call_waited(key, uses, screen, graphics, started, hr);
        if (SUCCEEDED(hr))
            must_draw = TRUE;
        call_outcome = SUCCEEDED(hr) ? OUTCOME_OK : hr == E_PENDING ? OUTCOME_WAIT : OUTCOME_FAILED;
        return hr;
    }
    switch (state)
    {
    case PROGRAM_COLD: /* claimed: compile it in the background */
        /* Ahead of the rest: a stage drawing into a display-sized target (the UI, text, the final
         * composite). A scene queues hundreds of programs at once, and the intro's title card's
         * text waited behind them until the run ended. */
        queue_compile(key, in, in_display_draw && current_draw.target >= 0, 0);
        set_last(E_PENDING, "compiling in the background", NULL);
        call_outcome = OUTCOME_QUEUED;
        return E_PENDING;
    case PROGRAM_PENDING:
        set_last(E_PENDING, "compiling in the background", NULL);
        call_outcome = OUTCOME_QUEUED;
        return E_PENDING;
    case PROGRAM_FAILED:
    {
        UINT retry = 0;
        /* Ahead of the rest, so it runs while this call's tables still hold what they hold now. */
        if (program_retry(key, &retry))
            queue_compile(key, in, TRUE, retry);
        set_last(E_FAIL, retry ? "its background compile failed; compiling it again in the background"
                               : "its background compile failed", NULL);
        call_outcome = OUTCOME_PROGRAM_FAILED;
        return E_FAIL;
    }
    default:
        return compile_within_budget(in, out);
    }
}

/* With REBOX_WATCH_STAGE naming a graphics stage, each of its compiles' results is logged (the first
 * 64): whether its draws are skipped, and why. Each call's outcome is counted for the census, and
 * traced under REBOX_FRAME_TRACE. */
/* REBOX_STANDIN_PIXEL=mesh (the default under REBOX_FAST) or all: a draw whose first stage compiled
 * but whose pixel stage is still compiling (or failed, or is past the frame's budget or its wait)
 * gets a stand-in pixel stage (generated/standin_shaders.h: every colour output a grey that darkens
 * with distance) instead of being left out, so geometry reaches the G-buffer and the forward target
 * while the title's own pixel stages compile. Once the real one is ready the draw's pipeline key
 * changes and it takes over. "mesh": only mesh pipelines (the G-buffer, decals, shadows), not
 * vertex-stage ones, among which are the full-screen passes a grey would cover the picture with.
 * REBOX_STANDIN_PIXEL=0 turns it off. */
static int standin_mode(void)
{
    static int mode = -1;
    if (mode < 0)
    {
        const char *v = getenv("REBOX_STANDIN_PIXEL"), *fast = getenv("REBOX_FAST");
        if (v && *v)
            mode = !strcmp(v, "all") ? 2 : !strcmp(v, "mesh") || !strcmp(v, "1") ? 1 : 0;
        else
            mode = fast && *fast && strcmp(fast, "0") ? 1 : 0;
        if (mode)
            rebox_log("d3d12_xs: REBOX_STANDIN_PIXEL: %s draws whose pixel stage isn't ready draw with a grey stand-in",
                      mode == 2 ? "all" : "mesh");
    }
    return mode;
}

static __thread struct { UINT32 stage; BOOL ok; } first_stage; /* this thread's last vertex or mesh compile */

/* What became of one draw (console_draw_watch_begin/_end, around the inner call; debug_view.c counts
 * them per target). vkd3d-proton compiles a console draw's first stage, then its pixel stage, then
 * looks up the pipeline made of the two, and tells which it found (patch 0032,
 * ReboxConsoleDrawPipeline): a variant whose pipeline its workers are still building is listed with
 * none, and the draw is left out without a word (patch 0016), however "ok" the census calls its
 * stages. */
static __thread struct
{
    BOOL on, standin;
    int first, pixel;  /* the stages' outcomes (OUTCOME_*), -1: not compiled */
    int pipeline;      /* 0: not told, 1: found, 2: none (being built, or failed) */
} draw_watch;

void WINAPI ReboxConsoleDrawPipeline(BOOL found)
{
    draw_watch.pipeline = found ? 1 : 2;
}

void console_draw_watch_begin(void)
{
    draw_watch.on = TRUE;
    draw_watch.standin = FALSE;
    draw_watch.first = draw_watch.pixel = -1;
    draw_watch.pipeline = 0;
}

UINT console_draw_watch_end(void)
{
    UINT fate;
    draw_watch.on = FALSE;
    if (draw_watch.first < 0)
        fate = DRAW_FATE_PLAIN;
    else if (draw_watch.first == OUTCOME_FAILED || draw_watch.first == OUTCOME_PROGRAM_FAILED)
        fate = DRAW_FATE_FIRST_FAILED;
    else if (draw_watch.first != OUTCOME_OK)
        fate = DRAW_FATE_FIRST_COMPILING;
    else if (draw_watch.pixel >= 0 && draw_watch.pixel != OUTCOME_OK && !draw_watch.standin)
        fate = DRAW_FATE_PIXEL_NOT_READY;
    else if (draw_watch.pipeline == 2)
        fate = DRAW_FATE_PIPELINE_BUILDING;
    else if (draw_watch.pipeline == 0)
        fate = DRAW_FATE_COMPILED;
    else
        fate = draw_watch.standin ? DRAW_FATE_STANDIN : DRAW_FATE_RECORDED;
    return fate;
}

const char *console_draw_fate_name(UINT fate)
{
    static const char *const names[DRAW_FATES] = {
        "recorded", "recorded with a stand-in pixel stage", "compiled (pipeline unknown: vkd3d-proton without patch 0032)",
        "left out while vkd3d-proton builds the pipeline", "left out, first stage compiling",
        "left out, first stage failed", "left out, pixel stage not ready", "not a console draw" };
    return fate < DRAW_FATES ? names[fate] : "?";
}


HRESULT WINAPI ReboxConsoleCompile(const struct rebox_compile_in *in, struct rebox_compile_out *out)
{
    HRESULT hr;
    const struct rebox_deferred_shader *h = in ? in->shader : NULL;

    call_outcome = OUTCOME_FAILED;
    hr = console_compile(in, out);
    /* vkd3d-proton compiles a draw's first stage, then its pixel stage, on the same thread. */
    if (h && in->shader_size >= sizeof(*h) && h->version == 2)
    {
        if (h->stage != 0x10)
        {
            first_stage.stage = h->stage;
            first_stage.ok = SUCCEEDED(hr);
        }
        else if (FAILED(hr) && out && first_stage.ok && standin_mode()
                 && (standin_mode() == 2 || first_stage.stage == 0x80))
        {
            memset(out, 0, sizeof(*out));
            out->spirv = standin_pixel_spirv;
            out->spirv_words = ARRAYSIZE(standin_pixel_spirv);
            out->key[0] = 0x5354414e44494e31ull; /* "STANDIN1": a variant of its own */
            out->key[1] = 0x50495845ull;
            InterlockedIncrement(&standins);
            draw_watch.standin = TRUE;
            hr = S_OK;
        }
        if (draw_watch.on)
        {
            if (h->stage != 0x10)
                draw_watch.first = (int)min(call_outcome, OUTCOMES - 1);
            else
                draw_watch.pixel = (int)min(call_outcome, OUTCOMES - 1);
        }
    }
    InterlockedIncrement(&outcomes[outcome_kind(h, in ? in->shader_size : 0)][min(call_outcome, OUTCOMES - 1)]);
    if (console_trace_active())
        trace_call(in, hr);

    if (getenv("REBOX_WATCH_STAGE") && h && in->shader_size >= sizeof(*h) && h->version == 2
            && h->code_offset <= in->shader_size && h->code_dwords * 4ull <= in->shader_size - h->code_offset)
    {
        UINT64 id = hash_bytes(hash_bytes(0xcbf29ce484222325ull, &h->stage, sizeof(h->stage)),
                               (const BYTE *)h + h->code_offset, h->code_dwords * 4u);
        static LONG logged;
        if (console_stage_watched(id) && InterlockedIncrement(&logged) <= 64)
        {
            char message[512];
            ReboxConsoleLastResult(message, sizeof(message));
            rebox_log("d3d12_xs: watched %s stage %016llx: compile %#lx (%s)", stage_name(h->stage),
                      (unsigned long long)id, (unsigned long)hr, message);
        }
    }
    return hr;
}

/* Guest memory for AnyPS5: the shadow heaps at their title addresses, then the GPU memory chunks
 * (xmem's allocations, at their GPU VAs) that don't overlap them, so the SRT walk can follow
 * user-data pointers into constant data. Title-created resources aren't included. Made again only
 * when the heaps or chunks change: walking the heaps' pages for GPU memory at every draw and
 * dispatch took a share of each. Returns the count, at most MAX_REGIONS. */
static UINT guest_regions(struct sa5_region *regions)
{
    static SRWLOCK cache_lock = SRWLOCK_INIT;
    static struct sa5_region cached[MAX_REGIONS];
    static UINT cached_count;
    static LONG cached_key[2];
    static BOOL cached_valid;
    const LONG key[2] = { shadow_regions_generation(), gpu_memory_generation() };
    struct shadow_region *shadows;
    struct gpu_memory_region *chunks;
    UINT n, count, shadow_count;

    AcquireSRWLockShared(&cache_lock);
    if (cached_valid && !memcmp(key, cached_key, sizeof(key)))
    {
        n = cached_count;
        memcpy(regions, cached, n * sizeof(*regions));
        ReleaseSRWLockShared(&cache_lock);
        return n;
    }
    ReleaseSRWLockShared(&cache_lock);

    shadows = malloc(MAX_REGIONS * sizeof(*shadows));
    chunks = malloc(MAX_REGIONS * sizeof(*chunks));
    if (!shadows || !chunks)
    {
        free(shadows);
        free(chunks);
        return 0;
    }
    n = shadow_regions(shadows, MAX_REGIONS);
    for (UINT i = 0; i < n; i++)
        regions[i] = (struct sa5_region){ shadows[i].title_va, shadows[i].data, shadows[i].size };
    count = gpu_memory_regions(chunks, MAX_REGIONS - n);
    note_regions_full("Shadow heaps", n, MAX_REGIONS);
    note_regions_full("GPU memory chunks", n + count, MAX_REGIONS);
    shadow_count = n;
    for (UINT i = 0; i < count; i++)
    {
        BOOL overlaps = FALSE;
        for (UINT j = 0; j < shadow_count && !overlaps; j++)
            overlaps = chunks[i].gpu < regions[j].address + regions[j].size
                    && regions[j].address < chunks[i].gpu + chunks[i].size;
        if (!overlaps && chunks[i].size)
            regions[n++] = (struct sa5_region){ chunks[i].gpu, chunks[i].cpu, chunks[i].size };
    }
    free(shadows);
    free(chunks);

    /* Kept under the generations read before it was made: a change since makes the next call
     * make it again. */
    AcquireSRWLockExclusive(&cache_lock);
    memcpy(cached, regions, n * sizeof(*regions));
    cached_count = n;
    memcpy(cached_key, key, sizeof(key));
    cached_valid = TRUE;
    ReleaseSRWLockExclusive(&cache_lock);
    return n;
}

/* Compiles run on any number of threads at once: AnyPS5's caches have their own locks, and so do
 * the module table, the per-draw ring, host buffers, the page table and the late lists here; what
 * compile_now keeps between calls is per thread. A lock around it used to make a title's recording
 * threads skip draws (or, waited for, queue behind background compiles' bookkeeping). */
#define unlocked_compile(background, call) ((void)(background), (call))

/* A mesh stage's user SGPRs: the hardware stage it runs as (NGG, a merged GS) has 8 system SGPRs
 * before its user data, so user-data dword i is s(8 + i), and the root parameters a vertex or pixel
 * stage gets from s8 are from s16 (the FSSM's root layout names the same user-data dwords for both;
 * VIctory's HwStage::user_sgpr; every one of the target title's refused mesh stages read its first
 * table through s16, where the root order put the sampler heap). Shifts in's SGPRs into mesh_sgprs
 * and returns a copy of in that points at them. */
#define MESH_SYSTEM_SGPRS 8
static const struct rebox_compile_in *mesh_user_sgprs(const struct rebox_compile_in *in, struct rebox_compile_in *copy,
                                                       UINT32 mesh_sgprs[64])
{
    UINT count = min(in->user_sgpr_count, 64u - MESH_SYSTEM_SGPRS);

    memset(mesh_sgprs, 0, 64 * sizeof(*mesh_sgprs));
    if (in->user_sgprs)
        memcpy(mesh_sgprs + MESH_SYSTEM_SGPRS, in->user_sgprs, count * sizeof(*mesh_sgprs));
    *copy = *in;
    copy->user_sgprs = mesh_sgprs;
    copy->user_sgpr_count = count + MESH_SYSTEM_SGPRS;
    return copy;
}

static HRESULT compile_now(const struct rebox_compile_in *in, struct rebox_compile_out *out, BOOL background)
{
    const struct rebox_deferred_shader *h = in ? in->shader : NULL;
    struct rebox_compile_in mesh_in;
    static __thread UINT32 mesh_sgprs[64];
    static __thread struct sa5_region regions[MAX_REGIONS];
    struct sa5_remap remaps[MAX_BINDINGS];
    struct sa5_host_buffer hosts[8];
    UINT64 host_va[8], key[2], pushed_va[GRAPHICS_PUSHED_MAX], identity[2] = { 0 }, bind_hash = 0;
    UINT32 pushed_count = 0, draw_lengths[SA5_DRAW_ADDRESSES] = { 0 };
    UINT32 remap_count = 0, host_count = 0, n, index_count = 0, *words = NULL, word_count = 0;
    static __thread UINT32 indices[MAX_ELEMENTS];
    static __thread struct { UINT position; UINT64 source; SIZE_T heap; } pending[64]; /* late T#s of this stage */
    UINT pending_count = 0, zero_textures = 0, zero_buffers = 0;
    BYTE *block_cpu = NULL;
    struct sa5_compute_request req;
    struct sa5_result *result = NULL;
    const void *push;
    struct module *module = NULL;
    char error[1024] = "";
    HRESULT hr = E_FAIL;
    UINT32 *code, nops;
    UINT32 push_size, stage_push;

    const struct rebox_deferred_graphics *g = h ? (const struct rebox_deferred_graphics *)(h + 1) : NULL;
    BOOL graphics;
    UINT64 stage_id, failure_id;
    LARGE_INTEGER started = { 0 }, mark = { 0 };

    if (!h || !out || in->shader_size < sizeof(*h) || h->magic != REBOX_DEFERRED_MAGIC
            || (h->version != 1 && h->version != 2)
            || h->size > in->shader_size || h->code_offset % 4 || h->code_offset > h->size
            || h->code_dwords > (h->size - h->code_offset) / 4)
        return E_INVALIDARG;
    /* Version 2: a vertex (0x1), pixel (0x10) or mesh (0x80) stage, rebox_deferred_graphics after the
     * header. */
    graphics = h->version == 2;
    if (graphics && ((h->stage != 0x1 && h->stage != 0x10 && h->stage != 0x80)
                     || h->code_offset < sizeof(*h) + sizeof(*g)))
        return E_INVALIDARG;
    stage_push = graphics && h->stage == 0x80 ? MESH_PUSH_DATA : GRAPHICS_PUSH_DATA;
    if (graphics && h->stage == 0x80 && !getenv("REBOX_MESH_USER_DATA_S8"))
        in = mesh_user_sgprs(in, &mesh_in, mesh_sgprs);
    stage_id = graphics ? hash_bytes(hash_bytes(0xcbf29ce484222325ull, &h->stage, sizeof(h->stage)),
                                     (const BYTE *)h + h->code_offset, h->code_dwords * 4u) : 0;
    /* Counts for the log, at powers of two of each kind's compiles: what they took, for title threads
     * (the draw or dispatch waits) and background workers apart, and graphics stages AnyPS5 refused. */
    {
        struct compile_stats *kind = compile_stats[!graphics];
        LONG n;
        stats = &kind[background != 0];
        relock_ticks = 0;
        n = InterlockedIncrement(&stats->calls);
        if (!(n & (n - 1)))
        {
            char rows[2][640];
            for (UINT b = 0; b < 2; b++)
            {
                const struct compile_stats *st = &kind[b];
                double ms = 1000.0 / qpc_frequency() / max(st->calls - (st == stats), 1);
                snprintf(rows[b], sizeof(rows[b]), "%ld, %.2f ms each (memory regions %.2f, lock %.2f, AnyPS5 %.2f, "
                         "bindings %.2f [identity %.2f, lookups %.2f, page table %.2f, host buffers %.2f, V#s %.2f, "
                         "T#s by address %.2f, dummies and views %.2f, late fills %.2f, notes %.2f], module "
                         "%.2f), %ld skipped as failing", st->calls - (st == stats), st->ticks * ms,
                         st->phase[PHASE_REGIONS] * ms, st->phase[PHASE_LOCK] * ms, st->phase[PHASE_ANYPS5] * ms,
                         st->phase[PHASE_BINDINGS] * ms, st->part[PART_IDENTITY] * ms, st->part[PART_LOOKUPS] * ms,
                         st->part[PART_PAGE_TABLE] * ms, st->part[PART_HOST_BUFFERS] * ms,
                         st->part[PART_GUEST_ENTRIES] * ms, st->part[PART_BY_ADDRESS] * ms, st->part[PART_SLOTS] * ms,
                         st->part[PART_LATE] * ms, st->part[PART_NOTES] * ms, st->phase[PHASE_MODULE] * ms,
                         st->memo_skips);
            }
            char refused[64] = "";
            if (graphics)
                snprintf(refused, sizeof(refused), "; %ld distinct stages refused so far",
                         min(failed_stage_count, (LONG)ARRAYSIZE(failed_stages)));
            rebox_log("d3d12_xs: console %s: compiles at title threads %s; at background workers %s%s; descriptor "
                      "bindings kept from earlier draws %ld, resolved %ld", graphics ? "graphics" : "compute", rows[0],
                      rows[1], refused, bind_hits, bind_misses);
        }
        QueryPerformanceCounter(&started);
        mark = started;
    }
    if (graphics && stage_failed(stage_id))
        return E_FAIL;
    failure_id = graphics ? stage_id : hash_words(0x636f6d70757465ull, (const BYTE *)h + h->code_offset, h->code_dwords * 4u);
    if (binding_failure_skips(failure_id))
    {
        InterlockedIncrement(&stats->memo_skips);
        set_last(E_FAIL, "its bindings failed before (retried at every 64th call)", NULL);
        return E_FAIL;
    }
    /* REBOX_WATCH_STAGE=<id>[:<offset>]: at that stage's draws, the 16 bytes at the offset (hex,
     * default 0x20) of the buffer its s8 table's first V# names: the first 16 draws, then every 100th. */
    if (graphics && getenv("REBOX_WATCH_STAGE"))
    {
        BOOL WINAPI ReboxShadowRead(UINT64 address, void *out, SIZE_T size);
        const char *watch = getenv("REBOX_WATCH_STAGE"), *colon = strchr(watch, ':');
        char id[24];
        snprintf(id, sizeof(id), "%016llx", (unsigned long long)stage_id);
        if (!strncmp(watch, id, 16))
        {
            static LONG draws;
            LONG n = InterlockedIncrement(&draws);
            UINT32 v[4];
            UINT64 offset = colon ? strtoull(colon + 1, NULL, 16) : 0x20;
            if ((n <= 16 || !(n % 100)) && in->user_sgprs[8] && ReboxShadowRead(in->user_sgprs[8], v, sizeof(v)))
            {
                UINT64 base = (UINT64)(v[1] & 0xffff) << 32 | v[0];
                const float *f = (const float *)(ULONG_PTR)(base + offset);
                if (base && !IsBadReadPtr(f, 16))
                    rebox_log("d3d12_xs: stage %s draw %ld: buffer %#llx + %#llx holds %g %g %g %g", id, n,
                              (unsigned long long)base, (unsigned long long)offset, f[0], f[1], f[2], f[3]);
                {
                    extern volatile UINT64 rebox_value_watch;
                    if (base)
                        rebox_value_watch = base + offset;
                }
            }
            /* At its first 4 draws, every buffer V# in its s8 table (16-byte entries whose type
             * nibble is a buffer's) and its first 48 values: its constants. */
            if (n <= 4 && in->user_sgprs[8])
            {
                UINT32 table[64];
                if (ReboxShadowRead(in->user_sgprs[8], table, sizeof(table)))
                    for (UINT e = 0; e < 16; e++)
                    {
                        const UINT32 *w = table + e * 4;
                        UINT64 base = (UINT64)(w[1] & 0xffff) << 32 | w[0];
                        const float *f = (const float *)(ULONG_PTR)base;
                        if (!w[0] || (w[3] >> 28) >= 8 || !base || IsBadReadPtr(f, 32))
                            continue;
                        rebox_log("d3d12_xs: stage %s draw %ld: table +%#x V# %08x %08x %08x %08x", id, n, e * 16,
                                  w[0], w[1], w[2], w[3]);
                        for (UINT q = 0; q < 48 && !IsBadReadPtr(f + q, 32); q += 8)
                            rebox_log("d3d12_xs:   +%#x: %g %g %g %g %g %g %g %g (%08x %08x %08x %08x %08x %08x %08x %08x)",
                                      q * 4, f[q], f[q + 1], f[q + 2], f[q + 3], f[q + 4], f[q + 5], f[q + 6], f[q + 7],
                                      ((const UINT32 *)f)[q], ((const UINT32 *)f)[q + 1], ((const UINT32 *)f)[q + 2],
                                      ((const UINT32 *)f)[q + 3], ((const UINT32 *)f)[q + 4], ((const UINT32 *)f)[q + 5],
                                      ((const UINT32 *)f)[q + 6], ((const UINT32 *)f)[q + 7]);
                    }
            }
        }
    }
    /* REBOX_DUMP_STAGES=<id>,... (with REBOX_DUMP_SPIRV): those stages' code and the first 256 bytes
     * of the table their s8 points at, once each, for disassembly. */
    if (graphics && getenv("REBOX_DUMP_STAGES") && getenv("REBOX_DUMP_SPIRV"))
    {
        BOOL WINAPI ReboxShadowRead(UINT64 address, void *out, SIZE_T size);
        static UINT64 dumped[16];
        static LONG dumped_count;
        char id[24], path[MAX_PATH];
        BOOL done = FALSE;
        snprintf(id, sizeof(id), "%016llx", (unsigned long long)stage_id);
        for (LONG i = 0; i < dumped_count && i < 16; i++)
            done |= dumped[i] == stage_id;
        if (!done && strstr(getenv("REBOX_DUMP_STAGES"), id) && dumped_count < 16)
        {
            UINT32 table[64];
            FILE *f;
            { LONG k = InterlockedIncrement(&dumped_count) - 1; if (k < 16) dumped[k] = stage_id; }
            snprintf(path, sizeof(path), "%s\\%s.code.bin", getenv("REBOX_DUMP_SPIRV"), id);
            if ((f = fopen(path, "wb")))
            {
                fwrite((const BYTE *)h + h->code_offset, 4, h->code_dwords, f);
                fclose(f);
            }
            snprintf(path, sizeof(path), "%s\\%s.table.bin", getenv("REBOX_DUMP_SPIRV"), id);
            if (in->user_sgprs[8] && ReboxShadowRead(in->user_sgprs[8], table, sizeof(table)) && (f = fopen(path, "wb")))
            {
                fwrite(table, 1, sizeof(table), f);
                fclose(f);
            }
            rebox_log("d3d12_xs: dumped stage %s (%u dwords; s8-s15 %08x %08x %08x %08x %08x %08x %08x %08x)", id,
                      h->code_dwords, in->user_sgprs[8], in->user_sgprs[9], in->user_sgprs[10], in->user_sgprs[11],
                      in->user_sgprs[12], in->user_sgprs[13], in->user_sgprs[14], in->user_sgprs[15]);
        }
    }
    if (graphics)
    {
        /* The first stages of each kind: the table pointers the code reads (sa5_analyse) next to the
         * user SGPRs it gets, to check the root parameter packing. */
        static LONG shown[2];
        if (InterlockedIncrement(&shown[h->stage == 0x10]) <= 6)
        {
            struct sa5_layout layout;
            static __thread struct sa5_slot slots[64];
            char line[512], ignored[64];
            int at = 0;
            InitOnceExecuteOnce(&sa5.once, load_sa5, NULL, NULL);
            memset(&layout, 0, sizeof(layout));
            if (sa5.analyse && !sa5.analyse((const UINT32 *)((const BYTE *)h + h->code_offset), h->code_dwords,
                                            32, &layout, slots, ARRAYSIZE(slots), ignored, sizeof(ignored)))
                for (UINT32 t = 0; t < layout.table_count && t < 16 && at < 400; t++)
                    at += snprintf(line + at, sizeof(line) - at, " s%u", layout.table_sgprs[t]);
            line[at] = 0;
            rebox_log("d3d12_xs: console %s shader %016llx reads tables through%s; user SGPRs s0-s15: %08x %08x %08x "
                         "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x",
                         stage_name(h->stage), (unsigned long long)stage_id, at ? line : " (none found)",
                         in->user_sgprs[0], in->user_sgprs[1], in->user_sgprs[2], in->user_sgprs[3], in->user_sgprs[4],
                         in->user_sgprs[5], in->user_sgprs[6], in->user_sgprs[7], in->user_sgprs[8], in->user_sgprs[9],
                         in->user_sgprs[10], in->user_sgprs[11], in->user_sgprs[12], in->user_sgprs[13],
                         in->user_sgprs[14], in->user_sgprs[15]);
            /* What the table s8 points at holds (in a shadow heap), for the zeros some stages read. */
            {
                BOOL WINAPI ReboxShadowRead(UINT64 address, void *out, SIZE_T size);
                UINT32 table[16];
                if (in->user_sgprs[8] && ReboxShadowRead(in->user_sgprs[8], table, sizeof(table)))
                    rebox_log("d3d12_xs:   at s8 %#x: %08x %08x %08x %08x %08x %08x %08x %08x  %08x %08x %08x %08x %08x "
                              "%08x %08x %08x", in->user_sgprs[8], table[0], table[1], table[2], table[3], table[4],
                              table[5], table[6], table[7], table[8], table[9], table[10], table[11], table[12],
                              table[13], table[14], table[15]);
                else if (in->user_sgprs[8])
                    rebox_log("d3d12_xs:   at s8 %#x: not in a shadow heap", in->user_sgprs[8]);
            }
        }
    }
    set_last(E_FAIL, "no result (out of memory?)", NULL);
    if (!graphics && compute_bisect_skips(h, in))
    {
        set_last(E_FAIL, "skipped by REBOX_CONSOLE_COMPUTE_BISECT", NULL);
        return E_FAIL;
    }
    InitOnceExecuteOnce(&sa5.once, load_sa5, NULL, NULL);
    poll_faults();
    if (!sa5.compile || (graphics && (!sa5.compile_vertex || !sa5.compile_pixel || !sa5.vertex_input || !sa5.finalize_draw))
            || (graphics && h->stage == 0x80 && !sa5.compile_mesh))
    {
        set_last(E_NOTIMPL, "rebox_anyps5.dll unavailable (or too old for graphics or mesh stages)", NULL);
        return E_NOTIMPL;
    }
    {
        static LONG logged;
        if (!InterlockedExchange(&logged, 1))
            rebox_log("d3d12_xs: console shaders: bindless sets (set/binding) ubo %d/%u ssbo %d/%u utexel %d/%u "
                         "stexel %d/%u sampled %d/%u storage %d/%u sampler %d/%u",
                         (int)in->bindless[0][0], in->bindless[0][1], (int)in->bindless[1][0], in->bindless[1][1],
                         (int)in->bindless[2][0], in->bindless[2][1], (int)in->bindless[3][0], in->bindless[3][1],
                         (int)in->bindless[4][0], in->bindless[4][1], (int)in->bindless[5][0], in->bindless[5][1],
                         (int)in->bindless[6][0], in->bindless[6][1]);
    }

    n = guest_regions(regions);
    if (!(code = malloc(h->code_dwords * 4u)))
        return E_OUTOFMEMORY;
    memcpy(code, (const BYTE *)h + h->code_offset, h->code_dwords * 4u);
    if ((nops = nop_flat_scratch_setup(code, h->code_dwords)))
    {
        static LONG logged;
        if (!InterlockedExchange(&logged, 1))
            rebox_log("d3d12_xs: console shaders: NOPed %u FLAT_SCRATCH setup instruction(s) (first seen)", nops);
    }
    memset(&req, 0, sizeof(req));
    req.code = code;
    req.code_dwords = h->code_dwords;
    req.code_address = h->code_address;
    req.wave = h->wave_size;
    if (!req.wave)
    {
        /* No wave size given (game containers carry one, console_fssm.c; SCRDs from other
         * sources may not): infer it from the code, as the layout analysis does (wave64 code
         * uses 64-bit exec and VCC). */
        struct sa5_layout layout;
        static struct sa5_slot slots[1];
        char ignored[64];
        req.wave = sa5.analyse && !sa5.analyse(code, h->code_dwords, user_sgprs(h, graphics), &layout, slots, 0,
                                                 ignored, sizeof(ignored)) ? layout.wave : 32;
    }
    req.user_data = in->user_sgprs;
    req.user_data_count = min(in->user_sgpr_count, user_sgprs(h, graphics));
    memcpy(req.threads, h->workgroup_size, sizeof(req.threads));
    req.rsrc1 = h->rsrc1;
    req.rsrc2 = h->rsrc2;
    req.regions = regions;
    req.region_count = n;
    req.use_cache = 1;

    phase_end(&mark, PHASE_REGIONS);
    phase_end(&mark, PHASE_LOCK); /* no lock now: kept so the phases line up */
    if (graphics && h->stage == 0x80)
    {
        /* A mesh stage (console_fssm.c): AnyPS5's mesh path with the group's size and limits, user
         * SGPRs as a vertex stage's (s8 on), push data 0-95 and the draw block's address at 96. */
        struct sa5_mesh_request mreq;
        memset(&mreq, 0, sizeof(mreq));
        mreq.code = req.code;
        mreq.code_dwords = req.code_dwords;
        mreq.code_address = req.code_address;
        mreq.wave = req.wave;
        mreq.user_data = req.user_data;
        mreq.user_data_count = max(in->user_sgpr_count, 8u);
        mreq.rsrc1 = h->rsrc1;
        mreq.threads = g->mesh.threads;
        mreq.max_vertices = g->mesh.max_vertices;
        mreq.max_primitives = g->mesh.max_primitives;
        mreq.group_stride = g->mesh.group_stride;
        mreq.primitive_vertices = g->mesh.primitive_vertices;
        mreq.lds_bytes = g->mesh.lds_bytes;
        mreq.clip_cntl = g->mesh.clip_cntl;
        mreq.push_offset = in->push_offset;
        mreq.push_size = MESH_PUSH_DATA;
        mreq.regions = regions;
        mreq.region_count = n;
        mreq.use_cache = 1;
        if (unlocked_compile(background, sa5.compile_mesh(&mreq, &result, error, sizeof(error))))
        {
            log_failure(h, "AnyPS5 failed (mesh)", error);
            graphics_refused(stage_id, error);
            goto done;
        }
    }
    else if (graphics && h->stage == 0x1)
    {
        /* The console's fetch call (s_swappc_b64 s[4:5]) becomes Vulkan vertex attributes: input r
         * fills v[4+4r..7+4r]. The V#s only steer AnyPS5's grouping; the attributes' formats and
         * buffers come from the PSO's input layout (patch 0006). s[4:5] holds a fetch address
         * outside the program, as the console driver's. */
        /* REBOX_GRAPHICS_DEBUG=fullscreen: a full-screen triangle from the vertex ID instead
         * (x = 4 * (vid & 1) - 1, y = 4 * ((vid >> 1) & 1) - 1, z = w = 1: nearest with reversed depth), with
         * magenta pixel shaders: tells vertex shaders that put everything off-screen apart from
         * draws that never reach the presented image. */
        static const UINT32 fullscreen[] = {
            0x36020081,             /* v_and_b32 v1, 1, v0 */
            0x2c040081,             /* v_lshrrev_b32 v2, 1, v0 */
            0x36040481,             /* v_and_b32 v2, 1, v2 */
            0x7e020d01,             /* v_cvt_f32_u32 v1, v1 */
            0x7e040d02,             /* v_cvt_f32_u32 v2, v2 */
            0x100202f6,             /* v_mul_f32 v1, 4.0, v1 */
            0x100404f6,             /* v_mul_f32 v2, 4.0, v2 */
            0x060202f3,             /* v_add_f32 v1, -1.0, v1 */
            0x060404f3,             /* v_add_f32 v2, -1.0, v2 */
            0x7e0a02f2,             /* v_mov_b32 v5, 1.0 */
            0x7e0c02f2,             /* v_mov_b32 v6, 1.0 */
            0xf80008cf, 0x06050201, /* exp pos0 v1, v2, v5, v6 done */
            0xbf810000,             /* s_endpgm */
        };
        /* REBOX_DEBUG_VS_CONST=<stage id>,...: those vertex stages draw the full-screen triangle and
         * export param0 = (1, 0, 1, 1), param1 = (0.5, 0.5, 0.5, 0.5): with REBOX_GRAPHICS_DEBUG=attr<N>
         * on the pixel stage, whether parameters reach it at all. */
        static const UINT32 constants[] = {
            0x36020081, 0x2c040081, 0x36040481, 0x7e020d01, 0x7e040d02, /* x, y from the vertex ID */
            0x100202f6, 0x100404f6, 0x060202f3, 0x060404f3,
            0x7e0a02f2, 0x7e0c02f2, 0xf80008cf, 0x06050201,             /* exp pos0 v1, v2, 1, 1 done */
            0x7e0e0280, 0x7e1002f0,                                     /* v7 = 0, v8 = 0.5 */
            0xf800020f, 0x06060706,                                     /* exp param0 v6, v7, v6, v6 */
            0xf800021f, 0x08080808,                                     /* exp param1 v8, v8, v8, v8 */
            0xbf810000,
        };
        const char *debug = getenv("REBOX_GRAPHICS_DEBUG"), *vs_const = getenv("REBOX_DEBUG_VS_CONST");
        BOOL replace = debug && !strcmp(debug, "fullscreen");
        BOOL constant = FALSE;
        if (vs_const)
        {
            char id[24];
            snprintf(id, sizeof(id), "%016llx", (unsigned long long)stage_id);
            constant = strstr(vs_const, id) != NULL;
        }
        struct sa5_vertex_request vreq;
        UINT32 user[32];
        memset(&vreq, 0, sizeof(vreq));
        memset(user, 0, sizeof(user));
        memcpy(user, req.user_data, min(req.user_data_count, 32u) * 4);
        user[4] = 0x30000000u;
        user[5] = 0;
        vreq.code = constant ? constants : replace ? fullscreen : req.code;
        vreq.code_dwords = constant ? ARRAYSIZE(constants) : replace ? ARRAYSIZE(fullscreen) : req.code_dwords;
        /* REBOX_DEBUG_VS_WHITE=<stage id>,...: the stage's own code, with its param0 export made
         * (1, 1, 1, 1): v_mov_b32 1.0 into the export's four sources just before it. Whether a
         * vertex colour it passes through is what blackens a draw. */
        /* REBOX_DEBUG_VS_UV=<stage id>,...: the same with param1 made (0.5, 0.5, 0.5, 0.5) (texture
         * coordinates at the centre). */
        static __thread UINT32 whitened[4096];
        const char *white = getenv("REBOX_DEBUG_VS_WHITE"), *uv = getenv("REBOX_DEBUG_VS_UV");
        char stage_text[24];
        snprintf(stage_text, sizeof(stage_text), "%016llx", (unsigned long long)stage_id);
        BOOL make_uv = uv && strstr(uv, stage_text);
        if (!constant && !replace && ((white && strstr(white, stage_text)) || make_uv)
                && req.code_dwords + 4 <= ARRAYSIZE(whitened))
        {
            const char *id = stage_text;
            UINT32 target = make_uv ? 0xf8000210u : 0xf8000200u; /* exp param1 or param0 */
            UINT32 value = make_uv ? 0xf0u : 0xf2u;             /* inline 0.5 or 1.0 */
            for (UINT k = 0; k + 1 < req.code_dwords; k++)
                    if ((req.code[k] & 0xfc0003f0u) == target)
                    {
                        UINT32 sources = req.code[k + 1];
                        memcpy(whitened, req.code, k * 4);
                        for (UINT c = 0; c < 4; c++) /* v_mov_b32 v<src>, <value> */
                            whitened[k + c] = 0x7e000200u | value | ((sources >> (8 * c)) & 0xff) << 17;
                        memcpy(whitened + k + 4, req.code + k, (req.code_dwords - k) * 4);
                        vreq.code = whitened;
                        vreq.code_dwords = req.code_dwords + 4;
                        rebox_log("d3d12_xs: REBOX_DEBUG_VS_%s: stage %s exports param%u as a constant (sources %08x)",
                                  make_uv ? "UV" : "WHITE", id, make_uv, sources);
                        break;
                    }
        }
        vreq.code_address = req.code_address;
        vreq.wave = req.wave;
        vreq.user_data = user;
        vreq.user_data_count = max(req.user_data_count, 6u);
        vreq.rsrc1 = h->rsrc1;
        vreq.input_count = replace || constant ? 0 : min(g->input_count, 32u);
        for (UINT32 r = 0; r < vreq.input_count; r++)
        {
            vreq.inputs[r].vsharp[0] = 0x20000000u + r * 0x100000u;
            vreq.inputs[r].vsharp[1] = 16u << 16;
            vreq.inputs[r].vsharp[2] = 0x10000;
            vreq.inputs[r].vsharp[3] = (77u << 12) | 0xfacu; /* 32_32_32_32_FLOAT, identity swizzle */
            vreq.inputs[r].register_start = 4 + 4 * r;
            vreq.inputs[r].register_count = 4;
        }
        vreq.push_offset = in->push_offset;
        vreq.push_size = GRAPHICS_PUSH_DATA;
        vreq.regions = regions;
        vreq.region_count = n;
        vreq.use_cache = 1;
        if (unlocked_compile(background, sa5.compile_vertex(&vreq, &result, error, sizeof(error))))
        {
            log_failure(h, "AnyPS5 failed (vertex)", error);
            graphics_refused(stage_id, error);
            goto done;
        }
    }
    else if (graphics)
    {
        /* REBOX_GRAPHICS_DEBUG=magenta: every console pixel shader exports magenta to MRT0 instead,
         * to tell whether draws reach the render targets at all (vertices, pipelines) from whether
         * the real pixel shaders' output is right. */
        static const UINT32 magenta[] = {
            0x7e0002f2,             /* v_mov_b32 v0, 1.0 */
            0x7e020280,             /* v_mov_b32 v1, 0 */
            0xf800180f, 0x00000100, /* exp mrt0 v0, v1, v0, v0 done vm */
            0xbf810000,             /* s_endpgm */
        };
        /* none:<stage id>,...: those pixel stages write nothing (an export to the null target):
         * what a full-screen pass, such as a fade, covers. */
        static const UINT32 nothing[] = {
            0xf8001890, 0x00000000, /* exp null off, off, off, off done vm */
            0xbf810000,             /* s_endpgm */
        };
        /* attr<N>:<stage id>,...: those pixel stages export their interpolant N's xyz instead (a
         * vertex colour, texture coordinates), alpha 1. */
        static __thread UINT32 attr0[] = {
            0xbefc0309,                         /* s_mov_b32 m0, s9 (the primitive mask; set below) */
            0xc8080000, 0xc8090001,             /* v_interp_p1/p2_f32 v2, attr0.x */
            0xc80c0100, 0xc80d0101,             /* v3, attr0.y */
            0xc8100200, 0xc8110201,             /* v4, attr0.z */
            0x7e0a02f2,                         /* v_mov_b32 v5, 1.0 */
            0x5e000702, 0x5e020b04,             /* v_cvt_pkrtz_f16_f32 v0, v2, v3; v1, v4, v5 */
            0xf8001c0f, 0x00000100,             /* exp mrt0 v0, v0, v1, v1 done compr vm */
            0xbf810000,                         /* s_endpgm */
        };
        /* texture<hex>:<stage id>,...: those pixel stages export, as grey, the first channel of the
         * texture whose T# is at that offset of their s8 table, sampled (level 0) at interpolant
         * 1's xy: whether a texture they read holds anything. */
        static __thread UINT32 texture[] = {
            0xbefc0309, 0xbe890380,             /* s_mov_b32 m0, s9; s_mov_b32 s9, 0 */
            0xf40c0504, 0xfa000040,             /* s_load_dwordx8 s[20:27], s[8:9], <offset> */
            0xc8080400, 0xc8090401,             /* v_interp_p1/p2_f32 v2, attr1.x */
            0xc80c0500, 0xc80d0501,             /* v3, attr1.y */
            0xb0240092, 0xbea503ff, 0x00c00000, /* s[36:39]: the video shader's sampler (linear, clamp) */
            0xbea603ff, 0x14500000, 0xbea70380,
            0xbf8cc07f,                         /* s_waitcnt lgkmcnt(0) */
            0xf09c0108, 0x01250402,             /* image_sample_lz v4, v[2:3], s[20:27], s[36:39] dmask:1 2D */
            0xbf8c3f70,                         /* s_waitcnt vmcnt(0) */
            0x7e0a02f2,                         /* v_mov_b32 v5, 1.0 */
            0x5e000904, 0x5e020b04,             /* v_cvt_pkrtz_f16_f32 v0, v4, v4; v1, v4, v5 */
            0xf8001c0f, 0x00000100,             /* exp mrt0 v0, v0, v1, v1 done compr vm */
            0xbf810000,                         /* s_endpgm */
        };
        const char *debug = getenv("REBOX_GRAPHICS_DEBUG");
        BOOL solid = debug && (!strcmp(debug, "magenta") || !strcmp(debug, "fullscreen"));
        const UINT32 *replacement = magenta;
        UINT replacement_dwords = ARRAYSIZE(magenta);
        if (debug && ((!strncmp(debug, "attr", 4) && debug[4] >= '0' && debug[4] <= '9') || !strncmp(debug, "texture", 7)))
        {
            char id[24];
            const char *ids = strchr(debug, ':');
            snprintf(id, sizeof(id), "%016llx", (unsigned long long)stage_id);
            if (ids && strstr(ids, id))
            {
                /* m0 (the primitive mask, which v_interp needs) from the SGPR the original code
                 * takes it from: s_mov_b32 m0, sX, through an earlier s_mov_b32 sX, sY. */
                UINT source = 9;
                for (UINT k = 0; k < req.code_dwords && k < 64; k++)
                    if ((req.code[k] & 0xffffff00u) == 0xbefc0300u)
                    {
                        UINT x = req.code[k] & 0xff;
                        source = x;
                        for (UINT j = 0; j < k; j++)
                            if ((req.code[j] & 0xff80ff00u) == 0xbe800300u && ((req.code[j] >> 16) & 0x7f) == x
                                    && (req.code[j] & 0xff) < 106)
                                source = req.code[j] & 0xff;
                        break;
                    }
                attr0[0] = 0xbefc0300u | source;
                texture[0] = 0xbefc0300u | source;
                solid = TRUE;
                if (debug[0] == 'a')
                {
                    UINT attr = strtoul(debug + 4, NULL, 10) & 31;
                    for (UINT k = 1; k <= 6; k++) /* the v_interp dwords: the attribute is bits 10-15 */
                        attr0[k] = (attr0[k] & ~(0x3fu << 10)) | attr << 10;
                    replacement = attr0, replacement_dwords = ARRAYSIZE(attr0);
                }
                else
                {
                    /* texturec<hex>: at the texture's centre, no interpolation */
                    BOOL centre = debug[7] == 'c';
                    texture[3] = 0xfa000000 | (strtoul(debug + 7 + centre, NULL, 16) & 0x1fffff);
                    texture[4] = centre ? 0x7e0402f0 : 0xc8080400; /* v_mov_b32 v2, 0.5 */
                    texture[5] = centre ? 0xbf800000 : 0xc8090401; /* s_nop 0 */
                    texture[6] = centre ? 0x7e0602f0 : 0xc80c0500; /* v_mov_b32 v3, 0.5 */
                    texture[7] = centre ? 0xbf800000 : 0xc80d0501;
                    replacement = texture, replacement_dwords = ARRAYSIZE(texture);
                }
            }
        }
        /* magenta:<stage id>,...: only those pixel stages (ids as the log names them). */
        if (debug && !strncmp(debug, "magenta:", 8))
        {
            char id[24];
            snprintf(id, sizeof(id), "%016llx", (unsigned long long)stage_id);
            solid = strstr(debug + 8, id) != NULL;
        }
        if (debug && !strncmp(debug, "none:", 5))
        {
            char id[24];
            snprintf(id, sizeof(id), "%016llx", (unsigned long long)stage_id);
            if ((solid = strstr(debug + 5, id) != NULL))
                replacement = nothing, replacement_dwords = ARRAYSIZE(nothing);
        }
        /* magenta-except:<stage id>,...: every pixel stage but those (a final pass kept real shows
         * whether anything drawn before it reaches what it reads). */
        if (debug && !strncmp(debug, "magenta-except:", 15))
        {
            char id[24];
            snprintf(id, sizeof(id), "%016llx", (unsigned long long)stage_id);
            solid = strstr(debug + 15, id) == NULL;
        }
        /* REBOX_DEBUG_PS_LZ=<stage id>,...: those pixel stages' own code with every image_sample made
         * image_sample_lz (level 0, no derivatives): whether implicit-LOD sampling is what fails. */
        static __thread UINT32 lz_code[4096];
        /* REBOX_DEBUG_PS_NOWQM=<stage id>,...: the same with the whole-quad-mode prologue (s_mov_b64
         * s[N:N+1], exec; s_wqm_b64 exec, exec; s_and_b64 exec, exec, s[N:N+1]) made s_nop: whether
         * the exec mask it leaves is what keeps the stage from writing. */
        const char *lz = getenv("REBOX_DEBUG_PS_LZ"), *nowqm = getenv("REBOX_DEBUG_PS_NOWQM");
        const UINT32 *own = req.code;
        char lz_id[24];
        snprintf(lz_id, sizeof(lz_id), "%016llx", (unsigned long long)stage_id);
        if (!solid && nowqm && strstr(nowqm, lz_id) && req.code_dwords <= ARRAYSIZE(lz_code))
        {
            UINT changed = 0;
            memcpy(lz_code, req.code, req.code_dwords * 4);
            for (UINT k = 0; k < req.code_dwords; k++)
                if ((lz_code[k] & 0xff80ffffu) == 0xbe80047eu      /* s_mov_b64 s[N:N+1], exec */
                        || lz_code[k] == 0xbefe0a7eu               /* s_wqm_b64 exec, exec */
                        || (lz_code[k] & 0xffff00ffu) == 0x87fe007eu /* s_and_b64 exec, exec, sN */)
                    lz_code[k] = 0xbf800000u, changed++;
            own = lz_code;
            rebox_log("d3d12_xs: REBOX_DEBUG_PS_NOWQM: stage %s, %u instructions made s_nop", lz_id, changed);
        }
        if (!solid && lz && req.code_dwords <= ARRAYSIZE(lz_code))
        {
            const char *id = lz_id;
            if (strstr(lz, id))
            {
                UINT changed = 0;
                memcpy(lz_code, req.code, req.code_dwords * 4);
                /* A dword that reads as image_sample (MIMG, opcode 0x20, gfx10 encoding 0xf080....);
                 * no length decoding, so a literal that looks like one is changed too: debugging only. */
                for (UINT k = 0; k + 1 < req.code_dwords; k++)
                    if ((lz_code[k] & 0xfdfc0000u) == 0xf0800000u)
                        lz_code[k] = (lz_code[k] & ~(0x7fu << 18)) | 0x27u << 18, changed++;
                own = lz_code;
                rebox_log("d3d12_xs: REBOX_DEBUG_PS_LZ: stage %s, %u image_sample made image_sample_lz", id, changed);
            }
        }
        /* REBOX_DEBUG_PS_CUT=<stage id>:<dword offset>:<r>,<g>,<b>: that pixel stage's own code up
         * to the offset (an instruction boundary), then a wait for its loads and an export of VGPRs
         * r, g, b (alpha 1): what the code has computed by then, cut down run by run to find an
         * instruction translated wrongly. */
        static __thread UINT32 cut_code[4096];
        const char *cut = getenv("REBOX_DEBUG_PS_CUT");
        UINT cut_dwords = 0;
        if (!solid && cut && !strncmp(cut, lz_id, 16) && cut[16] == ':')
        {
            char *end;
            UINT at = strtoul(cut + 17, &end, 10), rgb[3] = { 0, 0, 0 };
            if (*end == ':')
                sscanf(end + 1, "%u,%u,%u", &rgb[0], &rgb[1], &rgb[2]);
            if (at && at <= req.code_dwords && at + 8 <= ARRAYSIZE(cut_code))
            {
                memcpy(cut_code, own, at * 4);
                cut_code[at] = 0xbf8c0070u;                      /* s_waitcnt vmcnt(0) lgkmcnt(0) */
                cut_code[at + 1] = 0x7ffe02f2u;                  /* v_mov_b32 v255, 1.0 */
                if ((g->col_format & 0xf) == 4)
                {
                    /* The stage exports MRT0 as packed halves (compr): the same, or the target's
                     * format reads the words wrongly. */
                    cut_code[at + 2] = 0x5e000000u | 253u << 17 | (rgb[1] & 0xff) << 9 | (256 + (rgb[0] & 0xff));
                    cut_code[at + 3] = 0x5e000000u | 254u << 17 | 255u << 9 | (256 + (rgb[2] & 0xff));
                    cut_code[at + 4] = 0xf8001c0fu;              /* exp mrt0 v253, v254 done compr vm */
                    cut_code[at + 5] = 253u | 254u << 8;
                    cut_code[at + 6] = 0xbf810000u;
                    cut_dwords = at + 7;
                }
                else
                {
                    cut_code[at + 2] = 0xf800180fu;              /* exp mrt0 r, g, b, v255 done vm */
                    cut_code[at + 3] = (rgb[0] & 0xff) | (rgb[1] & 0xff) << 8 | (rgb[2] & 0xff) << 16 | 0xffu << 24;
                    cut_code[at + 4] = 0xbf810000u;              /* s_endpgm */
                    cut_dwords = at + 5;
                }
                own = cut_code;
                rebox_log("d3d12_xs: REBOX_DEBUG_PS_CUT: stage %s cut at dword %u, exporting v%u v%u v%u", lz_id, at,
                          rgb[0], rgb[1], rgb[2]);
            }
        }
        struct sa5_pixel_request preq;
        memset(&preq, 0, sizeof(preq));
        preq.code = solid ? replacement : own;
        preq.code_dwords = solid ? replacement_dwords : cut_dwords ? cut_dwords : req.code_dwords;
        preq.code_address = req.code_address;
        preq.wave = req.wave;
        preq.user_data = req.user_data;
        preq.user_data_count = req.user_data_count;
        preq.rsrc1 = h->rsrc1;
        preq.rsrc2 = h->rsrc2;
        preq.input_ena = g->input_ena;
        preq.interpolator_count = solid ? 0 : min(g->interpolator_count, 32u);
        memcpy(preq.interpolator_settings, g->interpolators, sizeof(preq.interpolator_settings));
        preq.col_format = solid ? 9 : g->col_format;
        preq.depth_export = g->depth_export;
        preq.push_offset = in->push_offset;
        preq.push_size = GRAPHICS_PUSH_DATA;
        preq.regions = regions;
        preq.region_count = n;
        preq.use_cache = 1;
        if (unlocked_compile(background, sa5.compile_pixel(&preq, &result, error, sizeof(error))))
        {
            log_failure(h, "AnyPS5 failed (pixel)", error);
            graphics_refused(stage_id, error);
            goto done;
        }
    }
    else if (unlocked_compile(background, sa5.compile(&req, &result, error, sizeof(error))))
    {
        log_failure(h, "AnyPS5 failed", error);
        goto done;
    }

    phase_end(&mark, PHASE_ANYPS5);
    if (relock_ticks)
    {
        /* Taking the lock back after a background compile is waiting, not AnyPS5. */
        InterlockedAdd64(&stats->phase[PHASE_ANYPS5], -relock_ticks);
        InterlockedAdd64(&stats->phase[PHASE_LOCK], relock_ticks);
        relock_ticks = 0;
    }
    {
        LONG64 t = ticks_now();
        sa5.identity(result, key);
        part_add(PART_IDENTITY, ticks_now() - t);
        memcpy(identity, key, sizeof(identity));
    }
    key[1] ^= hash_bytes(0xcbf29ce484222325ull, in->bindless, sizeof(in->bindless));
    if (graphics)
    {
        /* The same code as another stage or at another push offset is another module. */
        key[1] = hash_bytes(key[1], &h->stage, sizeof(h->stage));
        key[1] = hash_bytes(key[1], &in->push_offset, sizeof(in->push_offset));
    }
    for (UINT32 i = 0; i < sa5.binding_count(result); i++)
    {
        struct sa5_binding b;
        LONG64 notes_start = ticks_now();
        sa5.binding(result, i, &b);
        if (graphics)
        {
            /* The first graphics stages' bindings: which roles and kinds their resources come as. */
            static LONG shown[2];
            if (console_stage_watched(stage_id) || InterlockedIncrement(&shown[h->stage == 0x10]) <= 32)
                rebox_log("d3d12_xs: console %s shader %016llx binding %u: role %u kind %u set %u binding %u, "
                             "%u elements, %u words%s %08x %08x %08x %08x", stage_name(h->stage),
                             (unsigned long long)stage_id, i, b.role, b.kind, b.set, b.binding, b.count, b.word_count,
                             b.word_count ? ":" : "", b.word_count > 0 ? b.words[0] : 0, b.word_count > 1 ? b.words[1] : 0,
                             b.word_count > 2 ? b.words[2] : 0, b.word_count > 3 ? b.words[3] : 0);
        }
        part_add(PART_NOTES, ticks_now() - notes_start);
        if (b.role == 0 && b.kind == 1)
        {
            /* Guest storage buffers (every V#, constant buffers included) are reached through their
             * addresses, which buffers' GPU VAs make valid device addresses: a table of
             * {address, length in dwords} the module reads (sa5_host_buffer.guest_array). That
             * needs no descriptor, so V#s from memory work too, and CBVs (uniform-buffer
             * descriptors in vkd3d-proton) can be read as storage buffers. */
            UINT32 *table, per = b.count ? b.word_count / b.count : 0;
            UINT64 va;
            BYTE *table_cpu = NULL;
            UINT64 empty_sources[16];
            UINT empty_entries[16], empty_count = 0;
            if (host_count == ARRAYSIZE(hosts) || per != 4 || !(table = calloc(b.count ? b.count : 1, 16)))
            {
                log_failure(h, "no room for the guest buffer table", NULL);
                goto done;
            }
            for (UINT32 e = 0; e < b.count; e++)
            {
                const UINT32 *w = b.words + e * 4;
                UINT64 base = w[0] | (UINT64)(w[1] & 0xffff) << 32;
                UINT32 stride = (w[1] >> 16) & 0x3fff;
                UINT64 bytes = stride ? (UINT64)w[2] * stride : w[2];
                if (console_stage_watched(stage_id))
                {
                    static LONG buffers_shown;
                    if (InterlockedIncrement(&buffers_shown) <= 24)
                        rebox_log("d3d12_xs: watched stage %016llx buffer binding %u.%u: V# %08x %08x %08x %08x (base %#llx, "
                                  "%llu bytes)", (unsigned long long)stage_id, b.binding, e, w[0], w[1], w[2], w[3],
                                  (unsigned long long)base, (unsigned long long)bytes);
                }
                TIMED(PART_GUEST_ENTRIES, ReboxGuestBufferEntry(w, table + e * 4));
                /* An empty V# of a graphics stage: maybe one the title writes before it submits. */
                if (graphics && !(w[0] | w[1] | w[2] | w[3]) && empty_count < ARRAYSIZE(empty_sources))
                {
                    UINT64 source;
                    TIMED(PART_LATE, source = late_buffer_source(h, in, zero_buffers++));
                    if (source)
                    {
                        empty_sources[empty_count] = source;
                        empty_entries[empty_count++] = e;
                        if (console_stage_watched(stage_id))
                            rebox_log("d3d12_xs: console shader %016llx: empty V#, noted at %#llx",
                                      (unsigned long long)stage_id, (unsigned long long)source);
                    }
                }
                if (graphics)
                {
                    /* The first graphics stages' buffers and their first values (where guest
                     * memory has them): do vertex shaders get real transforms, or zeros? */
                    static LONG shown[2];
                    const float *f = NULL;
                    MEMORY_BASIC_INFORMATION mbi;
                    if (shown[h->stage == 0x10] >= 48)
                        continue;
                    for (UINT r = 0; r < n && !f; r++)
                        if (base >= regions[r].address && base + 32 <= regions[r].address + regions[r].size)
                            f = (const float *)((const BYTE *)regions[r].data + (base - regions[r].address));
                    /* xmem's GPU memory is identity-mapped: the title's own buffers read at their GPU VA. */
                    if (!f && base && VirtualQuery((const void *)(ULONG_PTR)base, &mbi, sizeof(mbi))
                            && mbi.State == MEM_COMMIT && !(mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD))
                            && base + 32 <= (UINT64)(ULONG_PTR)mbi.BaseAddress + mbi.RegionSize)
                        f = (const float *)(ULONG_PTR)base;
                    if (InterlockedIncrement(&shown[h->stage == 0x10]) <= 48)
                        rebox_log("d3d12_xs: console %s shader %016llx buffer %u.%u: base %#llx, stride %u, "
                                     "%u records%s %g %g %g %g %g %g %g %g", stage_name(h->stage),
                                     (unsigned long long)stage_id, i, e, (unsigned long long)base, stride, w[2],
                                     f ? ";" : " (not readable here)", f ? f[0] : 0, f ? f[1] : 0, f ? f[2] : 0,
                                     f ? f[3] : 0, f ? f[4] : 0, f ? f[5] : 0, f ? f[6] : 0, f ? f[7] : 0);
                }
            }
            {
                LONG64 t = ticks_now();
                va = draw_buffer(table, (b.count ? b.count : 1) * 16, &table_cpu);
                part_add(PART_HOST_BUFFERS, ticks_now() - t);
            }
            free(table);
            if (!va)
            {
                log_failure(h, "no GPU memory for the guest buffer table", NULL);
                goto done;
            }
            if (empty_count && table_cpu)
            {
                AcquireSRWLockExclusive(&late_lock);
                for (UINT k = 0; k < empty_count && late_buffer_count < ARRAYSIZE(late_buffers); k++)
                    late_buffers[late_buffer_count++] = (struct late_buffer){
                            (UINT32 *)(table_cpu + empty_entries[k] * 16), empty_sources[k], 0, stage_id };
                ReleaseSRWLockExclusive(&late_lock);
            }
            host_va[host_count] = va;
            /* Per-draw contents: pushed (graphics: after the stage's push data). */
            if (graphics && pushed_count == GRAPHICS_PUSHED_MAX)
            {
                log_failure(h, "more per-draw host buffers than a graphics stage has push room for", NULL);
                binding_failure_note(failure_id);
                goto done;
            }
            hosts[host_count] = (struct sa5_host_buffer){ b.set, b.binding, graphics ? pushed_count : host_count, 0, 1,
                    graphics ? in->push_offset + stage_push : 0, 0 };
            pushed_va[pushed_count++] = va;
            key[0] = hash_bytes(key[0], &hosts[host_count], sizeof(hosts[host_count]));
            host_count++;
        }
        else if (guest_descriptor_role(b.role) && b.role >= ROLE_IMAGE_TABLE && !b.count)
        {
            /* AnyPS5 fills image tables with the T#s it finds by walking tables in guest memory;
             * none found (as with synthetic inputs) leaves nothing to bind. */
            log_failure(h, "an image table AnyPS5 found no descriptors for", NULL);
            goto done;
        }
        else if (guest_descriptor_role(b.role))
        {
            BOOL sampler = b.kind == 6;
            UINT32 per = b.count ? b.word_count / b.count : 0;
            if (remap_count == MAX_BINDINGS || index_count + b.count > MAX_ELEMENTS || b.kind > 6
                    || (per != 4 && per != 8))
            {
                char detail[128];
                snprintf(detail, sizeof(detail), "role %u kind %u binding %u: %u elements of %u words (room for %u "
                         "elements in %u bindings)", b.role, b.kind, b.binding, b.count, per, MAX_ELEMENTS, MAX_BINDINGS);
                log_failure(h, "a guest binding is larger than d3d12_xs binds", detail);
                binding_failure_note(failure_id);
                goto done;
            }
            if (in->bindless[b.kind][0] == ~0u || !in->heap_cpu[sampler])
            {
                char detail[128];
                snprintf(detail, sizeof(detail), "role %u kind %u binding %u: %u elements of %u words", b.role,
                         b.kind, b.binding, b.count, per);
                log_failure(h, "a guest binding has no bindless set or heap here", detail);
                goto done;
            }
            /* Kept from an earlier draw (not a watched stage's: its notes are wanted at every draw). */
            struct bind_query query;
            UINT32 late_elements[BIND_CACHE_LATE];
            UINT late_element_count = 0;
            BOOL cacheable = bind_cache_enabled() && b.count <= BIND_CACHE_ELEMENTS && !console_stage_watched(stage_id);
            BOOL cached = FALSE, by_address = FALSE;
            if (cacheable)
            {
                LONG64 t = ticks_now();
                bind_query_make(&query, (SIZE_T)in->heap_cpu[sampler], graphics, &b, per);
                cached = bind_cache_get(&query, &b, &indices[index_count], late_elements, &late_element_count);
                part_add(PART_LOOKUPS, ticks_now() - t);
            }
            for (UINT k = 0; cached && k < late_element_count && pending_count < ARRAYSIZE(pending); k++)
            {
                /* As below, in the same order. */
                const UINT32 *w = b.words + late_elements[k] * per;
                UINT64 source;
                static LONG noted;
                TIMED(PART_LATE, source = late_source(h, in, zero_textures++));
                if (InterlockedIncrement(&noted) <= 4)
                    rebox_log("d3d12_xs: console shader %016llx: empty T# (%08x %08x %08x %08x), noted at %#llx (kept "
                              "bindings)", (unsigned long long)stage_id, w[4], w[5], w[6], w[7], (unsigned long long)source);
                if (source)
                {
                    pending[pending_count].position = index_count + late_elements[k];
                    pending[pending_count].source = source;
                    pending[pending_count].heap = (SIZE_T)in->heap_cpu[sampler];
                    pending_count++;
                }
            }
            for (UINT32 e = 0; !cached && e < b.count; e++)
            {
                const UINT32 *w = b.words + e * per;
                SIZE_T heap = (SIZE_T)in->heap_cpu[sampler];
                LONG64 part_start = ticks_now();
                if (graphics && !sampler)
                {
                    /* The first graphics stages' textures: which resource, and whether its texels
                     * come from title memory (uploaded how often) or through the API. */
                    static LONG shown;
                    UINT64 address = 0;
                    UINT uploads = 0;
                    /* Only when logging: console_texture_of walks every texture the title made. */
                    static LONG watched_shown;
                    /* The first 4 textures of each of the first 128 stages (one stage could use them all). */
                    static struct { UINT64 id; LONG count; } per_stage[128];
                    static LONG stages_seen;
                    BOOL stage_first = FALSE;
                    /* Read before they're counted: once the first 128 stages and their 4 textures
                     * are logged, every draw's every texture counted on the same shared lines. */
                    for (LONG k = 0; k < min(stages_seen, 128); k++)
                        if (per_stage[k].id == stage_id)
                        {
                            stage_first = per_stage[k].count < 4 && InterlockedIncrement(&per_stage[k].count) <= 4;
                            goto counted;
                        }
                    if (stages_seen < 128)
                    {
                        LONG k = InterlockedIncrement(&stages_seen) - 1;
                        if (k < 128)
                        {
                            per_stage[k].id = stage_id;
                            per_stage[k].count = 1;
                            stage_first = TRUE;
                        }
                    }
                counted:
                    /* REBOX_WATCH_TARGET: a stage binding the texture at that title address, and as what. */
                    {
                        static UINT64 target = ~0ull;
                        static LONG target_shown;
                        if (target == ~0ull)
                        {
                            const char *v = getenv("REBOX_WATCH_TARGET");
                            target = v ? _strtoui64(v, NULL, 16) : 0;
                        }
                        /* Only a T# of the target's address, or of a view of the texture there: asking
                         * every texture's title address (title_texture_info) waited on the title
                         * textures' lock, which their upload pass holds for milliseconds, at every
                         * texture of every draw (0.8 ms a draw, the bindings' "notes"). */
                        static ID3D12Resource *target_res;
                        static volatile LONG target_generation = -1;
                        LONG generation = texture_generation;
                        if (target && target_generation != generation)
                        {
                            ID3D12Resource *at_target;
                            if (title_texture_at_nowait(target, &at_target))
                            {
                                target_res = at_target;
                                target_generation = generation;
                            }
                        }
                        if (target && target_shown < 64)
                        {
                            UINT64 address = ((UINT64)(w[1] & 0xff) << 32 | w[0]) << 8, at = 0;
                            ID3D12Resource *res = address == target ? console_texture_of(w)
                                    : target_res && console_texture_by_id(w) == target_res ? target_res : NULL;
                            UINT up = 0;
                            if (res && title_texture_info(res, &at, &up) && at == target
                                    && InterlockedIncrement(&target_shown) <= 64)
                                rebox_log("d3d12_xs: console %s shader %016llx binds the watched target %#llx as "
                                          "binding %u.%u (kind %u, role %u)", stage_name(h->stage),
                                          (unsigned long long)stage_id, (unsigned long long)target, i, e,
                                          (UINT)b.kind, (UINT)b.role);
                        }
                    }
                    /* A watched stage's: the first 24, and again whenever the texture it samples
                     * changes (a scene can switch what a final pass reads). */
                    static ID3D12Resource *watched_last;
                    ID3D12Resource *watched_res = console_stage_watched(stage_id) ? console_texture_of(w) : NULL;
                    if ((console_stage_watched(stage_id)
                         && (InterlockedIncrement(&watched_shown) <= 24
                             || InterlockedExchangePointer((void **)&watched_last, watched_res) != watched_res))
                            || (stage_first && InterlockedIncrement(&shown) <= 512))
                    {
                        ID3D12Resource *res = console_texture_of(w);
                        D3D12_RESOURCE_DESC rd = { 0 };
                        BOOL title = res && title_texture_info(res, &address, &uploads);
                        if (res)
                            rd = ID3D12Resource_GetDesc(res);
                        rebox_log("d3d12_xs: console %s shader %016llx texture %u.%u: T# %08x %08x, resource %p "
                                     "%llux%u format %u flags %#x, %s %#llx, uploaded %u times",
                                     stage_name(h->stage), (unsigned long long)stage_id, i, e, w[0],
                                     w[1], (void *)res, (unsigned long long)rd.Width, rd.Height, rd.Format, rd.Flags,
                                     title ? "title memory at" : "not from title memory", (unsigned long long)address,
                                     uploads);
                    }
                }
                UINT *index = &indices[index_count + e];
                /* Samplers not in the heap (built from literals, or AnyPS5's point-filtered twins)
                 * are made in its reserved tail; matching that ignores rewritten fields is the
                 * last resort. */
                LONG64 lookup_start = ticks_now();
                UINT views = 0;
                part_add(PART_NOTES, lookup_start - part_start);
                for (UINT v = SHADOW_VIEW_NONE; v <= SHADOW_VIEW_SAMPLER; v++)
                    views |= view_fits(b.kind, (enum shadow_view)v) ? 1u << v : 0;
                BOOL found = shadow_find_index_view(heap, w, per, sampler, !sampler, views, index)
                        || (sampler && shadow_reserve_sampler(heap, w, index))
                        || (sampler && shadow_find_index(heap, w, per, sampler, TRUE, index));
                BOOL null_view = FALSE;
                part_start = ticks_now();
                part_add(PART_LOOKUPS, part_start - lookup_start);
                /* A texture descriptor the title wrote outside our heaps, or an empty slot, reads
                 * as zero, like a console null descriptor (a UE5 title: dummy T#s with base 0x9200):
                 * a zeroed dummy texture (shadow_reserve_null); binding the empty slot itself
                 * crashed lavapipe. */
                /* A T# the title wrote itself: the texture at its address, viewed in the reserved tail. */
                if (b.kind == 4 && (!found || shadow_slot_view(heap, *index) == SHADOW_VIEW_NONE))
                {
                    ID3D12Resource *res;
                    by_address |= w[0] || (w[1] & 0xff);
                    res = (w[0] || (w[1] & 0xff)) ? console_texture_of(w) : NULL;
                    if (res && shadow_reserve_srv(heap, res, index))
                    {
                        static LONG logged;
                        found = TRUE;
                        if (InterlockedIncrement(&logged) <= 16)
                            rebox_log("d3d12_xs: console shader %016llx: binding %u: a T# the title wrote itself "
                                      "(%08x %08x) is texture %p", (unsigned long long)stage_id, b.binding, w[0], w[1],
                                      (void *)res);
                    }
                }
                {
                    LONG64 now = ticks_now();
                    part_add(PART_BY_ADDRESS, now - part_start);
                    part_start = now;
                }
                if ((b.kind == 4 || b.kind == 5) && (!found || shadow_slot_view(heap, *index) == SHADOW_VIEW_NONE))
                    null_view = shadow_reserve_null(heap, b.kind == 5, index);
                {
                    LONG64 now = ticks_now();
                    part_add(PART_SLOTS, now - part_start);
                    part_start = now;
                }
                if (console_stage_watched(stage_id))
                {
                    static LONG resolved_shown;
                    if (InterlockedIncrement(&resolved_shown) <= 48)
                        rebox_log("d3d12_xs: watched stage %016llx binding %u.%u (kind %u, %u words %08x %08x %08x %08x): "
                                  "%s heap index %u, view %d%s", (unsigned long long)stage_id, b.binding, e, b.kind, per,
                                  w[0], w[1], w[2], w[3], found ? "found at" : "not found;", *index,
                                  found ? (int)shadow_slot_view(heap, *index) : -1, null_view ? ", a null view" : "");
                }
                /* An empty T# of a graphics stage: maybe one the title writes before it submits. Its
                 * address is the n-th texture slot of the code's tables that is empty now. */
                if (null_view && graphics && b.kind == 4 && !(w[0] | w[1] | w[2] | w[3]))
                {
                    if (late_element_count < ARRAYSIZE(late_elements))
                        late_elements[late_element_count] = e;
                    late_element_count++;
                }
                if (null_view && graphics && b.kind == 4 && !(w[0] | w[1] | w[2] | w[3])
                        && pending_count < ARRAYSIZE(pending))
                {
                    UINT64 source;
                    static LONG noted;
                    {
                        LONG64 late_start = ticks_now(), late_ticks;
                        source = late_source(h, in, zero_textures++);
                        late_ticks = ticks_now() - late_start;
                        part_add(PART_LATE, late_ticks);
                        part_start += late_ticks; /* not the notes' */
                    }
                    if (console_stage_watched(stage_id) || InterlockedIncrement(&noted) <= 4)
                        rebox_log("d3d12_xs: console shader %016llx: empty T# (%08x %08x %08x %08x), noted at %#llx",
                                  (unsigned long long)stage_id, w[4], w[5], w[6], w[7], (unsigned long long)source);
                    if (source)
                    {
                        pending[pending_count].position = index_count + e;
                        pending[pending_count].source = source;
                        pending[pending_count].heap = heap;
                        pending_count++;
                    }
                }
                if (null_view)
                {
                    static LONG logged;
                    if (InterlockedIncrement(&logged) <= 8)
                    {
                        BOOL WINAPI ReboxShadowRead(UINT64 address, void *out, SIZE_T size);
                        UINT32 table[8] = { 0 };
                        BOOL read = in->user_sgprs[8] && ReboxShadowRead(in->user_sgprs[8], table, sizeof(table));
                        rebox_log("d3d12_xs: console shader %016llx: binding %u (%s) gets a dummy texture: "
                                     "the descriptor %s (T# %08x %08x %08x %08x; s8 %#x %s %08x %08x %08x %08x %08x "
                                     "%08x %08x %08x)", (unsigned long long)hash_bytes(0xcbf29ce484222325ull,
                                     (const BYTE *)h + h->code_offset, h->code_dwords * 4u), b.binding,
                                     b.kind == 5 ? "storage image" : "sampled image",
                                     found ? "slot is empty" : "isn't in the bound heap", w[0], w[1], w[2], w[3],
                                     in->user_sgprs[8], read ? "holds" : "unreadable", table[0], table[1], table[2],
                                     table[3], table[4], table[5], table[6], table[7]);
                        /* The table again at the next submission (proxy.c): whether the title fills it
                         * between recording the draw and submitting it. */
                        {
                            extern volatile UINT64 rebox_table_watch;
                            if (in->user_sgprs[8])
                                rebox_table_watch = in->user_sgprs[8];
                        }
                        /* With REBOX_DUMP_SPIRV, the stage's code and the buffer the table's first V# names,
                         * for disassembly. */
                        if (getenv("REBOX_DUMP_SPIRV") && logged == 1)
                        {
                            char path[MAX_PATH];
                            FILE *f;
                            UINT64 base = read ? ((UINT64)(table[1] & 0xffff) << 32 | table[0]) : 0;
                            UINT64 bytes = (UINT64)table[2] * ((table[1] >> 16) & 0x3fff ? (table[1] >> 16) & 0x3fff : 1);
                            snprintf(path, sizeof(path), "%s\\%016llx.code.bin", getenv("REBOX_DUMP_SPIRV"),
                                     (unsigned long long)stage_id);
                            if ((f = fopen(path, "wb")))
                            {
                                fwrite((const BYTE *)h + h->code_offset, 4, h->code_dwords, f);
                                fclose(f);
                            }
                            snprintf(path, sizeof(path), "%s\\%016llx.cb.bin", getenv("REBOX_DUMP_SPIRV"),
                                     (unsigned long long)stage_id);
                            if (base && bytes && bytes <= 65536 && !IsBadReadPtr((void *)(ULONG_PTR)base, bytes)
                                    && (f = fopen(path, "wb")))
                            {
                                fwrite((void *)(ULONG_PTR)base, 1, bytes, f);
                                fclose(f);
                            }
                        }
                    }
                    found = TRUE;
                }
                if (!found)
                {
                    char detail[160];
                    const UINT32 *w = b.words + e * per;
                    snprintf(detail, sizeof(detail), "kind %u binding %u element %u: %08x %08x %08x %08x%s", b.kind,
                             b.binding, e, w[0], w[1], w[2], w[3], per == 8 ? " ..." : "");
                    if (per == 8)
                        snprintf(detail + strlen(detail) - 4, sizeof(detail) - strlen(detail) + 4,
                                 " %08x %08x %08x %08x", w[4], w[5], w[6], w[7]);
                    log_failure(h, "a descriptor it binds isn't in the bound heap", detail);
                    goto done;
                }
                {
                    LONG64 now = ticks_now();
                    part_add(PART_NOTES, now - part_start);
                    part_start = now;
                }
                /* A storage image AnyPS5 also binds as a sampled image (size queries): an SRV of
                 * the same texture in the reserved tail. Empty slots are refused: binding one
                 * crashed lavapipe. */
                if (!null_view && b.kind == 4 && shadow_slot_view(heap, *index) == SHADOW_VIEW_UAV_TEXTURE)
                    shadow_reserve_srv_for_uav(heap, *index, index);
                if (!null_view && !view_fits(b.kind, shadow_slot_view(heap, *index)))
                {
                    static const char *const kinds[] = { "uniform buffer", "storage buffer", "uniform texel buffer",
                            "storage texel buffer", "sampled image", "storage image", "sampler" };
                    char detail[128];
                    snprintf(detail, sizeof(detail), "binding %u wants a %s, heap slot %u holds %s", b.binding,
                             kinds[b.kind], *index, view_names[shadow_slot_view(heap, *index)]);
                    log_failure(h, "a binding doesn't match the view in its slot", detail);
                    goto done;
                }
                part_add(PART_SLOTS, ticks_now() - part_start);
            }
            if (cacheable && !cached)
                bind_cache_put(&query, &b, &indices[index_count], by_address, late_elements, late_element_count);
            remaps[remap_count++] = (struct sa5_remap){ b.set, b.binding, b.kind, in->bindless[b.kind][0],
                                                        in->bindless[b.kind][1], b.count, &indices[index_count],
                                                        DRAW_INDICES - 2 * SA5_DRAW_ADDRESSES + index_count };
            key[0] = hash_bytes(key[0], &b.binding, sizeof(b.binding));
            key[0] = hash_bytes(key[0], &b.count, sizeof(b.count));
            bind_hash = hash_bytes(hash_bytes(bind_hash, &b.binding, sizeof(b.binding)), &b.count, sizeof(b.count));
            /* The heap slots are in the draw block, not the module (a UE5 title copies its
             * descriptors to new slots every draw, and a world's compute passes theirs: a module per
             * draw or dispatch, and a pipeline each, compute's built while its dispatches were
             * skipped). */
            index_count += b.count;
        }
        else
        {
            UINT64 va = 0;
            UINT32 bytes = b.word_count * 4;
            if (host_count == ARRAYSIZE(hosts))
            {
                log_failure(h, "more than 8 host buffers", NULL);
                binding_failure_note(failure_id);
                goto done;
            }
            switch (b.role)
            {
            case ROLE_IMAGE_TABLE_MAP: /* the image tables' map (ImageTableAbi.hpp), host data like the SRT */
            case ROLE_FLATTENED_SRT:
            case ROLE_SHADER_DATA:
            {
                LONG64 t = ticks_now();
                va = b.word_count ? draw_buffer(b.words, bytes, NULL) : zero_buffer(bytes = 16);
                part_add(PART_HOST_BUFFERS, ticks_now() - t);
                break;
            }
            case ROLE_FAULT_BUFFER:
            {
                InitOnceExecuteOnce(&fault_buffer_once, init_fault_buffer, NULL, NULL);
                va = fault_buffer.gpu;
                bytes = fault_buffer.size;
                break;
            }
            case ROLE_GDS:
            {
                static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
                InitOnceExecuteOnce(&once, init_gds_buffer, NULL, NULL);
                va = gds_buffer.gpu;
                bytes = gds_buffer.size;
                break;
            }
            case ROLE_BDA_PAGETABLE:
            {
                LONG64 t = ticks_now();
                va = page_table(&bytes);
                part_add(PART_PAGE_TABLE, ticks_now() - t);
                break;
            }
            default:
            {
                char role[32];
                snprintf(role, sizeof(role), "role %u", b.role);
                log_failure(h, "unsupported host binding", role);
                binding_failure_note(failure_id);
                goto done;
            }
            }
            if (!va)
            {
                log_failure(h, "no GPU memory for a host buffer", NULL);
                goto done;
            }
            host_va[host_count] = va;
            /* Graphics stages share the push block (128 bytes each): buffers whose contents change per
             * draw (the flattened SRT, shader data, image table maps) are pushed after the stage's
             * push data; the process-wide ones (fault buffer, GDS, page table) are baked in. */
            if (graphics && (b.role == ROLE_IMAGE_TABLE_MAP || b.role == ROLE_FLATTENED_SRT || b.role == ROLE_SHADER_DATA))
            {
                if (pushed_count == GRAPHICS_PUSHED_MAX)
                {
                    log_failure(h, "more per-draw host buffers than a graphics stage has push room for", NULL);
                    binding_failure_note(failure_id);
                    goto done;
                }
                /* Their sizes change from draw to draw too: the length is in the draw block. */
                hosts[host_count] = (struct sa5_host_buffer){ b.set, b.binding, pushed_count, SA5_LENGTH_IN_DRAW_BLOCK,
                        0, in->push_offset + stage_push, 0 };
                draw_lengths[pushed_count] = bytes / 4;
                pushed_va[pushed_count++] = va;
            }
            else
            {
                /* A dispatch's per-dispatch buffers have their lengths in its draw block (a flattened
                 * SRT's size, baked in, made a module and a pipeline of a world's 350,000-word compute
                 * shaders for every size it came in); the process-wide ones' are constant. */
                BOOL per_dispatch = !graphics && host_count < SA5_DRAW_ADDRESSES
                        && (b.role == ROLE_IMAGE_TABLE_MAP || b.role == ROLE_FLATTENED_SRT || b.role == ROLE_SHADER_DATA);
                hosts[host_count] = (struct sa5_host_buffer){ b.set, b.binding, host_count,
                        per_dispatch ? SA5_LENGTH_IN_DRAW_BLOCK : bytes / 4, 0, 0, graphics ? va : 0 };
                if (per_dispatch)
                    draw_lengths[host_count] = bytes / 4;
                if (graphics)
                    key[0] = hash_bytes(key[0], &va, sizeof(va));
            }
            key[0] = hash_bytes(key[0], &hosts[host_count], sizeof(hosts[host_count]));
            host_count++;
        }
    }

    phase_end(&mark, PHASE_BINDINGS);
    if (!(module = find_module(key)) && !(module = module_wait(key)))
    {
        BOOL failed = sa5.finalize_draw(result, remaps, remap_count, hosts, host_count,
                                        graphics ? in->push_offset + stage_push : COMPUTE_DRAW_BLOCK, &words,
                                        &word_count, error, sizeof(error));
        if (!failed)
            module = add_module(key, words, word_count);
        module_built(key);
        if (failed)
        {
            log_failure(h, "fitting it to vkd3d-proton's bindings failed", error);
            binding_failure_note(failure_id);
            goto done;
        }
        if (!module)
            goto done;
        if (getenv("REBOX_DUMP_SPIRV"))
        {
            char path[MAX_PATH];
            FILE *f;
            snprintf(path, sizeof(path), "%s\\%016llx%016llx.spv", getenv("REBOX_DUMP_SPIRV"),
                     (unsigned long long)key[1], (unsigned long long)key[0]);
            if ((f = fopen(path, "wb")))
            {
                fwrite(module->words, 4, module->count, f);
                fclose(f);
            }
        }
        /* The key is what vkd3d-proton's breadcrumbs record before the dispatch or draw (patch 0008):
         * a module with the same SPIR-V as another goes by that one's key. */
        {
            UINT64 code_hash = hash_bytes(0xcbf29ce484222325ull, req.code, req.code_dwords * 4u);
            char same[48] = "";
            if (memcmp(module->vk_key, key, sizeof(key)))
                snprintf(same, sizeof(same), "; same SPIR-V as key %016llx", (unsigned long long)module->vk_key[0]);
            rebox_log("d3d12_xs: console shader %016llx: compiled a variant (%u SPIR-V words, %u guest bindings, "
                         "%u host buffers; key %016llx%s)", (unsigned long long)code_hash, module->count, remap_count,
                         host_count, (unsigned long long)key[0], same);
            if (variant_log())
                variant_note(code_hash, h->stage, identity, hosts, host_count, bind_hash, module);
        }
    }

    phase_end(&mark, PHASE_MODULE);
    memset(out, 0, sizeof(*out));
    out->spirv = module->words;
    out->spirv_words = module->count;
    out->subgroup_size = sa5.subgroup_size(result);
    memcpy(out->key, module->vk_key, sizeof(out->key));
    push = sa5.push_constants(result, &push_size);
    memcpy(out->push, push, min(push_size, graphics ? stage_push : 128u));
    if (!graphics)
    {
        /* A dispatch's draw block (DRAW_INDICES): its host buffers' addresses (by push slot, their index here),
         * their lengths, then its bindings' heap slots; its address at push byte COMPUTE_DRAW_BLOCK. */
        static __thread UINT32 block[DRAW_INDICES + MAX_ELEMENTS];
        UINT64 va;
        memset(block, 0, DRAW_INDICES * sizeof(UINT32));
        memcpy(block + 2 * SA5_DRAW_ADDRESSES, draw_lengths, sizeof(draw_lengths));
        memcpy(block, host_va, min(host_count, (UINT32)SA5_DRAW_ADDRESSES) * sizeof(UINT64));
        memcpy(block + DRAW_INDICES, indices, index_count * sizeof(UINT32));
        if (!(va = draw_buffer(block, (DRAW_INDICES + max(index_count, 1u)) * sizeof(UINT32), NULL)))
        {
            log_failure(h, "no GPU memory for the draw block", NULL);
            goto done;
        }
        memcpy(&out->push[COMPUTE_DRAW_BLOCK / 4], &va, sizeof(UINT64));
    }
    if (graphics)
    {
        /* The draw block: the per-draw host buffers' addresses and lengths, then every remapped binding's heap
         * slots (sa5_finalize_draw); its address follows the stage's push data. */
        static __thread UINT32 block[DRAW_INDICES + MAX_ELEMENTS];
        UINT64 va;
        memset(block, 0, DRAW_INDICES * sizeof(UINT32));
        memcpy(block + 2 * SA5_DRAW_ADDRESSES, draw_lengths, sizeof(draw_lengths));
        memcpy(block, pushed_va, pushed_count * sizeof(UINT64));
        memcpy(block + DRAW_INDICES, indices, index_count * sizeof(UINT32));
        if (!(va = draw_buffer(block, (DRAW_INDICES + max(index_count, 1u)) * sizeof(UINT32), &block_cpu)))
        {
            log_failure(h, "no GPU memory for the draw block", NULL);
            goto done;
        }
        memcpy(&out->push[stage_push / 4], &va, sizeof(UINT64));
        if (pending_count && block_cpu)
        {
            UINT noted = 0;
            AcquireSRWLockExclusive(&late_lock);
            for (UINT i = 0; i < pending_count && late_count < ARRAYSIZE(late); i++, noted++)
                late[late_count++] = (struct late_texture){
                        (UINT32 *)(block_cpu + (DRAW_INDICES + pending[i].position) * sizeof(UINT32)),
                        pending[i].source, pending[i].heap, 0, stage_id };
            ReleaseSRWLockExclusive(&late_lock);
            InterlockedAdd(&late_notes, noted);
        }
    }
    if (graphics && h->stage == 0x1)
    {
        out->vertex_input_count = min(sa5.vertex_input_count(result), 32u);
        for (UINT32 i = 0; i < out->vertex_input_count; i++)
            sa5.vertex_input(result, i, &out->vertex_inputs[i][0], &out->vertex_inputs[i][1], &out->vertex_inputs[i][2]);
        {
            static LONG shown;
            char line[256];
            int at = 0;
            for (UINT32 i = 0; i < out->vertex_input_count && at < 200; i++)
                at += snprintf(line + at, sizeof(line) - at, " %u:%u/%u", out->vertex_inputs[i][0],
                               out->vertex_inputs[i][1], out->vertex_inputs[i][2]);
            line[at] = 0;
            if (console_stage_watched(stage_id) || InterlockedIncrement(&shown) <= 12)
                rebox_log("d3d12_xs: console vertex shader %016llx: %u inputs (location:components/fetch)%s",
                             (unsigned long long)stage_id, out->vertex_input_count, line);
        }
    }
    hr = S_OK;
    binding_failure_clear(failure_id);
    {
        char detail[96];
        snprintf(detail, sizeof(detail), "%u SPIR-V words, %u guest bindings, %u host buffers", module->count,
                 remap_count, host_count);
        set_last(S_OK, words ? "compiled" : "cached", detail);
    }

done:
    if (started.QuadPart && stats)
    {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        InterlockedAdd64(&stats->ticks, now.QuadPart - started.QuadPart);
    }
    if (words)
    {
        void (*free_words)(UINT32 *) = (void *)GetProcAddress(GetModuleHandleA("rebox_anyps5.dll"), "sa5_free_words");
        if (free_words)
            free_words(words);
    }
    /* A rect-list draw's stages keep their result for ReboxConsoleRectStages. */
    if (result && module && in && in->rect_list && graphics)
    {
        AcquireSRWLockExclusive(&modules_lock);
        if (!module->result)
        {
            module->result = result;
            result = NULL;
        }
        ReleaseSRWLockExclusive(&modules_lock);
    }
    if (result)
        sa5.result_free(result);
    free(code);
    return hr;
}

/* ------------------------------------------------------------- rect lists */

/* The console's rect lists (topology 17: three corners of each rectangle, the fourth made by the
 * hardware) drawn by deferred console graphics: AnyPS5's rect-list shaders, a tessellation control
 * and evaluation pair over 3-vertex patches that makes the fourth corner after the vertex shader,
 * built for each pair of vertex and pixel stages (patch 0011). The control stage's fault buffer is
 * the process's. */
static struct rect_stages
{
    UINT64 vs[2], ps[2];
    struct module *control, *evaluation; /* NULL: they couldn't be built */
    struct rect_stages *next;
} *rect_cache;
static SRWLOCK rect_lock = SRWLOCK_INIT;

static struct module *rect_module(struct sa5_result *result, UINT64 salt, const UINT64 vs[2], const UINT64 ps[2])
{
    struct sa5_host_buffer hosts[4];
    UINT32 n = 0, *words = NULL, count = 0, bindings = sa5.binding_count(result);
    UINT64 key[2];
    struct module *m = NULL;
    char error[512];

    for (UINT32 i = 0; i < bindings && n < ARRAYSIZE(hosts); i++)
    {
        struct sa5_binding b;
        if (sa5.binding(result, i, &b))
            continue;
        if (b.role != ROLE_FAULT_BUFFER)
        {
            rebox_log("d3d12_xs: a rect-list stage binds role %u, which isn't handled", b.role);
            return NULL;
        }
        InitOnceExecuteOnce(&fault_buffer_once, init_fault_buffer, NULL, NULL);
        if (!fault_buffer.gpu)
            return NULL;
        hosts[n] = (struct sa5_host_buffer){ b.set, b.binding, n, fault_buffer.size / 4, 0, 0, fault_buffer.gpu };
        n++;
    }
    if (sa5.finalize(result, NULL, 0, hosts, n, &words, &count, error, sizeof(error)))
    {
        rebox_log("d3d12_xs: a rect-list stage couldn't be fitted to vkd3d-proton: %s", error);
        return NULL;
    }
    key[0] = hash_bytes(salt ^ vs[0] ^ (ps[0] * 0x9e3779b97f4a7c15ull), words, count * 4);
    key[1] = vs[1] ^ (ps[1] * 0xff51afd7ed558ccdull) ^ salt;
    m = add_module(key, words, count);
    {
        void (*free_words)(UINT32 *) = (void *)GetProcAddress(GetModuleHandleA("rebox_anyps5.dll"), "sa5_free_words");
        if (free_words)
            free_words(words);
    }
    return m;
}

HRESULT WINAPI ReboxConsoleRectStages(const struct rebox_compile_out *vs, const struct rebox_compile_out *ps,
                                      struct rebox_compile_out *control, struct rebox_compile_out *evaluation)
{
    struct sa5_result *vertex_result = NULL, *pixel_result = NULL, *c = NULL, *e = NULL;
    struct module *vm, *pm;
    struct rect_stages *r;
    static LONG logged, failures;
    char error[512];

    if (!vs || !ps || !control || !evaluation)
        return E_INVALIDARG;
    InitOnceExecuteOnce(&sa5.once, load_sa5, NULL, NULL);
    if (!sa5.rect_list_stages)
        return E_NOTIMPL;
    AcquireSRWLockShared(&rect_lock);
    for (r = rect_cache; r && (memcmp(r->vs, vs->key, 16) || memcmp(r->ps, ps->key, 16)); r = r->next)
        ;
    ReleaseSRWLockShared(&rect_lock);
    if (!r)
    {
        if (!(vm = find_module(vs->key)) || !(pm = find_module(ps->key)))
            return E_FAIL;
        AcquireSRWLockShared(&modules_lock);
        vertex_result = vm->result;
        pixel_result = pm->result;
        ReleaseSRWLockShared(&modules_lock);
        /* Not kept (compiled for another draw first): the next rect-list draw compiles them again. */
        if (!vertex_result || !pixel_result)
            return E_PENDING;
        if (!(r = calloc(1, sizeof(*r))))
            return E_OUTOFMEMORY;
        memcpy(r->vs, vs->key, 16);
        memcpy(r->ps, ps->key, 16);
        if (sa5.rect_list_stages(vertex_result, pixel_result, &c, &e, error, sizeof(error)))
        {
            if (InterlockedIncrement(&failures) <= 8)
                rebox_log("d3d12_xs: no rect-list stages for console vertex %016llx and pixel %016llx: %s",
                          (unsigned long long)vs->key[0], (unsigned long long)ps->key[0], error);
        }
        else
        {
            r->control = rect_module(c, 0x5443u, vs->key, ps->key);
            r->evaluation = r->control ? rect_module(e, 0x5445u, vs->key, ps->key) : NULL;
        }
        if (c)
            sa5.result_free(c);
        if (e)
            sa5.result_free(e);
        AcquireSRWLockExclusive(&rect_lock);
        r->next = rect_cache;
        rect_cache = r;
        ReleaseSRWLockExclusive(&rect_lock);
        if (r->control && r->evaluation && !InterlockedExchange(&logged, 1))
            rebox_log("d3d12_xs: console rect lists are drawn with AnyPS5's rect-list stages (tessellation: the "
                      "fourth corner made after the vertex shader)");
    }
    if (!r->control || !r->evaluation)
        return E_FAIL;
    memset(control, 0, sizeof(*control));
    memset(evaluation, 0, sizeof(*evaluation));
    control->spirv = r->control->words;
    control->spirv_words = r->control->count;
    memcpy(control->key, r->control->vk_key, sizeof(control->key));
    evaluation->spirv = r->evaluation->words;
    evaluation->spirv_words = r->evaluation->count;
    memcpy(evaluation->key, r->evaluation->vk_key, sizeof(evaluation->key));
    return S_OK;
}
