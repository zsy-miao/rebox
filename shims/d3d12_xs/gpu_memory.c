/* GPU-visible memory for xmem.dll and for command-list embedded data.
 *
 * Memory comes from CPU-visible custom heaps (WRITE_BACK, L0) on the shared
 * device, each spanned by one UAV-capable buffer, so vkd3d-proton knows the
 * whole range as a resource and resolves raw GPU VAs inside it. Small
 * requests are suballocated from 64 MiB chunks.
 *
 * Whether the returned CPU pointer equals its GPU VA depends on the Vulkan
 * driver and VK_LAYER_REBOX_gpuva_identity (docs/d3d12x.md). It is checked
 * once per chunk and logged. */

#define COBJMACROS
#include <windows.h>
#include <d3d12.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/rebox_log.h"
#include "../../layers/gpuva_mailbox.h"
#include "gpu_memory.h"
#include "title_textures.h"

#define CHUNK_SIZE (64u << 20)
#define DEDICATED_THRESHOLD (16u << 20)
#define HEAP_ALIGN (64u << 10)

struct range { UINT64 start, end; };

struct chunk
{
    ID3D12Heap *heap;
    ID3D12Resource *buffer;
    BYTE *cpu;
    UINT64 gpu, size, used;
    BOOL dedicated, dirty; /* dirty: something was freed, so ranges may hold old data */
    BOOL adopted; /* title memory made GPU-visible in place (gpu_memory_adopt): never freed */
    struct range *free;
    size_t free_count, free_cap;
    struct chunk *next;
};

struct allocation
{
    BYTE *cpu;
    struct chunk *chunk;
    UINT64 offset, size;
    struct allocation *next;
};

#define BUCKETS 4096

static SRWLOCK lock = SRWLOCK_INIT;
static struct chunk *chunks;
static volatile LONG chunks_generation; /* bumped under the lock after chunks changes */

/* The chunks by GPU VA, for lookups by address: every guest buffer a draw binds asked several
 * times, each a walk of every chunk (a world's adopted title memory makes hundreds), at thousands
 * of draws a frame. Made again under the lock, exclusive, when chunks_generation has moved on. */
struct span { UINT64 gpu, end; BOOL identity; };
static struct span *spans;
static UINT span_count, span_capacity;
static LONG spans_generation = -1;
static BOOL spans_overlap; /* chunks overlapping in GPU VA (never seen): the walks instead */

static int compare_spans(const void *a, const void *b)
{
    const struct span *x = a, *y = b;
    return x->gpu < y->gpu ? -1 : x->gpu > y->gpu;
}

/* Under the lock, exclusive. */
static void spans_make(void)
{
    UINT n = 0;
    for (struct chunk *c = chunks; c; c = c->next)
        n++;
    if (n > span_capacity)
    {
        struct span *more = realloc(spans, n * 2 * sizeof(*spans));
        if (!more)
        {
            span_count = 0;
            spans_overlap = TRUE; /* walks until it can be made */
            spans_generation = chunks_generation;
            return;
        }
        spans = more;
        span_capacity = n * 2;
    }
    n = 0;
    for (struct chunk *c = chunks; c; c = c->next)
        spans[n++] = (struct span){ c->gpu, c->gpu + c->size, (UINT64)(ULONG_PTR)c->cpu == c->gpu };
    qsort(spans, n, sizeof(*spans), compare_spans);
    spans_overlap = FALSE;
    for (UINT i = 1; i < n && !spans_overlap; i++)
        spans_overlap = spans[i].gpu < spans[i - 1].end;
    span_count = n;
    spans_generation = chunks_generation;
}

/* Takes the lock, shared, with the spans made for the chunks there are; FALSE if they can't be
 * used (the chunks are to be walked). */
static BOOL lock_spans(void)
{
    AcquireSRWLockShared(&lock);
    if (spans_generation != chunks_generation)
    {
        ReleaseSRWLockShared(&lock);
        AcquireSRWLockExclusive(&lock);
        if (spans_generation != chunks_generation)
            spans_make();
        ReleaseSRWLockExclusive(&lock);
        AcquireSRWLockShared(&lock);
    }
    /* chunks_generation moves only under the lock, exclusive: what was made is current. */
    return spans_generation == chunks_generation && !spans_overlap;
}

/* Under lock_spans: the span holding address, or NULL. */
static const struct span *span_at(UINT64 address)
{
    UINT lo = 0, hi = span_count;
    while (lo < hi) /* the first span starting past address */
    {
        UINT mid = lo + (hi - lo) / 2;
        if (spans[mid].gpu <= address)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo && address < spans[lo - 1].end ? &spans[lo - 1] : NULL;
}
static UINT64 bytes_in_use;
static struct allocation *table[BUCKETS];

static unsigned bucket(const void *p)
{
    ULONG_PTR h = (ULONG_PTR)p >> 6;
    return (unsigned)((h ^ (h >> 13)) % BUCKETS);
}

/* --------------------------------------------------------------- ranges */

/* Alignment applies to the CPU address (and so to the GPU VA when they are
 * equal), not just the offset: without the layer, mappings may only be
 * page-aligned. */
static BOOL range_alloc(struct chunk *c, UINT64 size, UINT64 align, UINT64 *out)
{
    const UINT64 base = (ULONG_PTR)c->cpu;

    for (size_t i = 0; i < c->free_count; i++)
    {
        struct range *r = &c->free[i];
        UINT64 start = ((base + r->start + align - 1) & ~(align - 1)) - base;

        if (start + size > r->end)
            continue;
        if (start > r->start && start + size < r->end)
        {
            if (c->free_count == c->free_cap)
                c->free = realloc(c->free, (c->free_cap *= 2) * sizeof(*c->free));
            r = &c->free[i];
            memmove(&c->free[i + 2], &c->free[i + 1], (c->free_count - i - 1) * sizeof(*c->free));
            c->free[i + 1] = (struct range){ start + size, r->end };
            r->end = start;
            c->free_count++;
        }
        else if (start > r->start)
            r->end = start;
        else if (start + size < r->end)
            r->start = start + size;
        else
        {
            memmove(&c->free[i], &c->free[i + 1], (c->free_count - i - 1) * sizeof(*c->free));
            c->free_count--;
        }
        *out = start;
        return TRUE;
    }
    return FALSE;
}

static void range_free(struct chunk *c, UINT64 start, UINT64 size)
{
    UINT64 end = start + size;
    size_t i;

    for (i = 0; i < c->free_count && c->free[i].start < start; i++)
        ;
    if (i > 0 && c->free[i - 1].end == start)
    {
        c->free[i - 1].end = end;
        if (i < c->free_count && c->free[i].start == end)
        {
            c->free[i - 1].end = c->free[i].end;
            memmove(&c->free[i], &c->free[i + 1], (c->free_count - i - 1) * sizeof(*c->free));
            c->free_count--;
        }
        return;
    }
    if (i < c->free_count && c->free[i].start == end)
    {
        c->free[i].start = start;
        return;
    }
    if (c->free_count == c->free_cap)
        c->free = realloc(c->free, (c->free_cap *= 2) * sizeof(*c->free));
    memmove(&c->free[i + 1], &c->free[i], (c->free_count - i) * sizeof(*c->free));
    c->free[i] = (struct range){ start, end };
    c->free_count++;
}

/* --------------------------------------------------------------- chunks */

static struct chunk *chunk_create(ID3D12Device *device, UINT64 size, BOOL dedicated, BOOL adopted)
{
    D3D12_HEAP_DESC hd = {
        .SizeInBytes = size,
        .Properties = { D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0, 0, 0 },
        .Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
        /* Adopted memory gets the title's bytes copied back; zeroing it first is wasted. */
        .Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | (adopted ? D3D12_HEAP_FLAG_CREATE_NOT_ZEROED : 0),
    };
    D3D12_RESOURCE_DESC rd = {
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = size, .Height = 1, .DepthOrArraySize = 1,
        .MipLevels = 1, .Format = DXGI_FORMAT_UNKNOWN, .SampleDesc = { 1, 0 },
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    D3D12_RANGE none = { 0, 0 };
    struct chunk *c = calloc(1, sizeof(*c));
    static LONG reported;
    HRESULT hr;

    if (FAILED(hr = ID3D12Device_CreateHeap(device, &hd, &IID_ID3D12Heap, (void **)&c->heap)))
    {
        rebox_log("d3d12_xs: CreateHeap(%#llx) for GPU memory failed: %#lx", (unsigned long long)size, hr);
        goto fail;
    }
    if (FAILED(hr = ID3D12Device_CreatePlacedResource(device, c->heap, 0, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                      NULL, &IID_ID3D12Resource, (void **)&c->buffer)))
    {
        rebox_log("d3d12_xs: CreatePlacedResource(%#llx) for GPU memory failed: %#lx", (unsigned long long)size, hr);
        goto fail;
    }
    if (FAILED(hr = ID3D12Resource_Map(c->buffer, 0, &none, (void **)&c->cpu)))
    {
        rebox_log("d3d12_xs: Map of GPU memory failed: %#lx", hr);
        goto fail;
    }
    c->gpu = ID3D12Resource_GetGPUVirtualAddress(c->buffer);
    c->size = size;
    c->dedicated = dedicated;
    c->free_cap = 8;
    c->free = calloc(c->free_cap, sizeof(*c->free));
    c->free[0] = (struct range){ 0, size };
    c->free_count = 1;

    if (adopted)
        return c;
    if ((UINT64)(ULONG_PTR)c->cpu != c->gpu)
    {
        if (!InterlockedExchange(&reported, 1))
            rebox_log("d3d12_xs: GPU memory is NOT identity-mapped (CPU %p, GPU VA %#llx): titles that use "
                         "GPU VAs as pointers will fault. Use .#radv-rebox and VK_LAYER_REBOX_gpuva_identity.",
                         c->cpu, (unsigned long long)c->gpu);
    }
    else if (!InterlockedExchange(&reported, 1))
        rebox_log("d3d12_xs: GPU memory is identity-mapped (first chunk at %p)", c->cpu);
    return c;

fail:
    if (c->buffer)
        ID3D12Resource_Release(c->buffer);
    if (c->heap)
        ID3D12Heap_Release(c->heap);
    free(c);
    return NULL;
}

static void chunk_destroy(struct chunk *c)
{
    ID3D12Resource_Unmap(c->buffer, 0, NULL);
    ID3D12Resource_Release(c->buffer);
    ID3D12Heap_Release(c->heap);
    free(c->free);
    free(c);
}

/* ---------------------------------------------------------------- API */

HRESULT WINAPI ReboxGpuAlloc(SIZE_T size, SIZE_T alignment, void **cpu, UINT64 *gpu)
{
    ID3D12Device *device = rebox_shared_device();
    struct allocation *a;
    struct chunk *c;
    UINT64 offset = 0;
    BOOL recycled;

    if (!cpu)
        return E_INVALIDARG;
    *cpu = NULL;
    if (!device)
        return E_FAIL;
    if (!size)
        size = 1;
    if (alignment < 256)
        alignment = 256; /* D3D12 constant-buffer alignment; cheap to always give */
    if (alignment & (alignment - 1))
        return E_INVALIDARG;

    AcquireSRWLockExclusive(&lock);
    if (size > DEDICATED_THRESHOLD || alignment > HEAP_ALIGN)
    {
        UINT64 chunk_size = (size + alignment + HEAP_ALIGN - 1) & ~(UINT64)(HEAP_ALIGN - 1);
        if ((c = chunk_create(device, chunk_size, TRUE, FALSE)) && range_alloc(c, size, alignment, &offset))
        {
            c->next = chunks;
            chunks = c;
            InterlockedIncrement(&chunks_generation);
        }
        else if (c)
        {
            chunk_destroy(c);
            c = NULL;
        }
    }
    else
    {
        for (c = chunks; c; c = c->next)
            if (!c->dedicated && range_alloc(c, size, alignment, &offset))
                break;
        if (!c && (c = chunk_create(device, CHUNK_SIZE, FALSE, FALSE)))
        {
            range_alloc(c, size, alignment, &offset);
            c->next = chunks;
            chunks = c;
            InterlockedIncrement(&chunks_generation);
        }
    }
    if (!c)
    {
        ReleaseSRWLockExclusive(&lock);
        return E_OUTOFMEMORY;
    }
    a = calloc(1, sizeof(*a));
    a->cpu = c->cpu + offset;
    recycled = c->dirty;
    a->chunk = c;
    a->offset = offset;
    a->size = size;
    a->next = table[bucket(a->cpu)];
    table[bucket(a->cpu)] = a;
    c->used += size;
    bytes_in_use += size;
    ReleaseSRWLockExclusive(&lock);

    /* New heaps are zero-filled; ranges reused within a chunk aren't. XMemAlloc
     * memory comes back zeroed, so clear them: anything from a chunk that has
     * had a free. */
    if (recycled)
        memset(a->cpu, 0, size);

    *cpu = a->cpu;
    if (gpu)
        *gpu = c->gpu + offset;
    return S_OK;
}

void WINAPI ReboxGpuFree(void *cpu)
{
    struct allocation **p, *a;
    struct chunk *c, **cp;

    if (!cpu)
        return;
    AcquireSRWLockExclusive(&lock);
    for (p = &table[bucket(cpu)]; *p && (*p)->cpu != cpu; p = &(*p)->next)
        ;
    if (!(a = *p))
    {
        ReleaseSRWLockExclusive(&lock);
        rebox_log("d3d12_xs: ReboxGpuFree(%p): not a GPU allocation", cpu);
        return;
    }
    *p = a->next;
    c = a->chunk;
    range_free(c, a->offset, a->size);
    c->used -= a->size;
    c->dirty = TRUE;
    bytes_in_use -= a->size;
    free(a);
    if (c->dedicated && !c->used)
    {
        for (cp = &chunks; *cp != c; cp = &(*cp)->next)
            ;
        *cp = c->next;
        InterlockedIncrement(&chunks_generation);
        chunk_destroy(c);
    }
    ReleaseSRWLockExclusive(&lock);
}

/* Whether cpu is a live ReboxGpuAlloc allocation (tests: what is freed when). */
BOOL WINAPI ReboxGpuAllocated(const void *cpu)
{
    struct allocation *a;

    AcquireSRWLockShared(&lock);
    for (a = table[bucket(cpu)]; a && a->cpu != cpu; a = a->next)
        ;
    ReleaseSRWLockShared(&lock);
    return a != NULL;
}

BOOL WINAPI ReboxGpuQuery(const void *address, void **base, SIZE_T *size)
{
    const BYTE *p = address;
    struct chunk *c;
    BOOL found = FALSE;

    AcquireSRWLockShared(&lock);
    for (c = chunks; c; c = c->next)
        if (p >= c->cpu && p < c->cpu + c->size)
        {
            if (base)
                *base = c->cpu;
            if (size)
                *size = c->size;
            found = TRUE;
            break;
        }
    ReleaseSRWLockShared(&lock);
    return found;
}

/* The end of the identity window's highest allocation (layers/gpuva_mailbox.h), or 0. */
UINT64 gpu_memory_window_end(void)
{
    static struct gpuva_mailbox *(WINAPI *get_mailbox)(void);
    static BOOL resolved;
    struct gpuva_mailbox *box;

    if (!resolved)
    {
        HMODULE xmem = GetModuleHandleA("xmem.dll");
        get_mailbox = xmem ? (void *)GetProcAddress(xmem, "ReboxGpuvaMailbox") : NULL;
        resolved = TRUE;
    }
    if (!get_mailbox || !(box = get_mailbox()) || box->magic != GPUVA_MAILBOX_MAGIC)
        return 0;
    return box->window_end;
}

/* Whether [gpu, gpu + size) is inside one of the chunks, by GPU VA. */
BOOL gpu_memory_holds_gpu(UINT64 gpu, UINT64 size)
{
    BOOL found = FALSE;

    if (lock_spans())
    {
        const struct span *s = span_at(gpu);
        found = s && gpu + size <= s->end;
    }
    else
        for (struct chunk *c = chunks; c && !found; c = c->next)
            found = gpu >= c->gpu && gpu + size <= c->gpu + c->size;
    ReleaseSRWLockShared(&lock);
    return found;
}

/* If gpu is inside one of the chunks (by GPU VA), the end of that chunk. */
BOOL gpu_memory_gpu_extent(UINT64 gpu, UINT64 *end)
{
    BOOL found = FALSE;

    AcquireSRWLockShared(&lock);
    for (struct chunk *c = chunks; c && !found; c = c->next)
        if (gpu >= c->gpu && gpu < c->gpu + c->size)
        {
            *end = c->gpu + c->size;
            found = TRUE;
        }
    ReleaseSRWLockShared(&lock);
    return found;
}

/* Bytes of title memory made GPU memory in place: the title's own pages stay behind them. */
UINT64 gpu_memory_adopted_bytes(void)
{
    UINT64 n = 0;

    AcquireSRWLockShared(&lock);
    for (struct chunk *c = chunks; c; c = c->next)
        if (c->adopted)
            n += c->size;
    ReleaseSRWLockShared(&lock);
    return n;
}

UINT64 WINAPI ReboxGpuBytesInUse(void)
{
    UINT64 n;

    AcquireSRWLockShared(&lock);
    n = bytes_in_use;
    ReleaseSRWLockShared(&lock);
    return n;
}

LONG gpu_memory_generation(void)
{
    return chunks_generation;
}

UINT gpu_memory_regions(struct gpu_memory_region *out, UINT max)
{
    UINT n = 0;

    AcquireSRWLockShared(&lock);
    for (struct chunk *c = chunks; c && n < max; c = c->next)
        out[n++] = (struct gpu_memory_region){ c->gpu, c->cpu, c->size };
    ReleaseSRWLockShared(&lock);
    return n;
}

/* Spans of title memory made GPU memory: 64 KiB-aligned (.#radv-rebox takes a lower-half address
 * with the page alignment), 2 MiB or more where there is room, and as much of the committed run
 * around the request as ADOPT_GRANULE allows, so neighbouring placements land in one chunk instead
 * of straddling two. Where there isn't room (between chunks, or in a short run) a span down to
 * 64 KiB fills what there is: vkd3d-proton suballocates buffer heaps under 2 MiB, so the layer
 * would never see the allocation, unless d3d12_xs asks it not to (ReboxHeapWhole, vkd3d-proton
 * patch 0033). Without that, such holes stayed out of GPU memory for good: a world's mesh draws'
 * root views (a title's per-draw data, 64 KiB apart from chunks on either side) read as empty,
 * 230 of 317 draws a frame. */
#define ADOPT_ALIGN (64ull << 10)
#define ADOPT_MIN (2ull << 20)

/* The thread making an adopted chunk's heap under 2 MiB (under the lock, exclusive), which
 * ReboxHeapWhole answers for; whether vkd3d-proton asked (it has patch 0033): -1 not known yet. */
static volatile DWORD whole_heap_thread;
static volatile LONG whole_heap_asked;
static volatile LONG whole_heaps = -1;

/* From vkd3d-proton (patch 0033), for a buffer heap under 2 MiB it is about to suballocate: TRUE
 * to give it memory of its own instead, as the heaps of title memory made GPU memory need. */
BOOL WINAPI ReboxHeapWhole(UINT64 size)
{
    if (whole_heap_thread != GetCurrentThreadId())
        return FALSE;
    InterlockedExchange(&whole_heap_asked, 1);
    return TRUE;
}

/* The least span adopt makes: 64 KiB unless vkd3d-proton is known not to keep such heaps whole,
 * or REBOX_ADOPT_SMALL=0. */
static UINT64 adopt_min_span(void)
{
    static int small = -1;
    if (small < 0)
    {
        const char *v = getenv("REBOX_ADOPT_SMALL");
        small = !v || strcmp(v, "0");
    }
    return small && whole_heaps ? ADOPT_ALIGN : ADOPT_MIN;
}
/* Between chunks: none. RADV maps a read-only page past each allocation on GPUs with the SMEM
 * out-of-bounds mitigation, which kept chunks 64 KiB apart, and the first 64 KiB of the next
 * granule out of GPU memory (a root UAV there faulted the GPU); patches/mesa-radv-low-va-no-pad.patch
 * leaves it out for allocations at title addresses. Adopted allocations have their exact size (the
 * layer). */
#define ADOPT_GAP 0ull
#define ADOPT_GRANULE (256ull << 20)

/* The run of committed, writable pages around address, as far as [lo, hi): [*start, *end). Walked
 * out from address's region both ways and no further than the window: a title's reservations hold
 * thousands of physical-page views, and a walk from the allocation's base took thousands of
 * VirtualQuery calls a draw. */
/* Read-only too with write tracking on: xmem arms textures' physical pages so (title_textures.c),
 * and a run stopping at them left buffers beside them out of GPU memory. */
static BOOL usable_region(const MEMORY_BASIC_INFORMATION *mbi)
{
    DWORD usable = PAGE_READWRITE | PAGE_EXECUTE_READWRITE | (title_textures_tracking() ? PAGE_READONLY : 0);
    return mbi->AllocationBase && mbi->State == MEM_COMMIT && (mbi->Protect & usable) && !(mbi->Protect & PAGE_GUARD);
}

static BOOL committed_run(UINT64 address, UINT64 lo, UINT64 hi, BOOL reach_back, UINT64 *start, UINT64 *end)
{
    MEMORY_BASIC_INFORMATION mbi;
    UINT64 run_start, run_end, reach, own_lo;
    BOOL mapped;

    if (!VirtualQuery((void *)(ULONG_PTR)address, &mbi, sizeof(mbi)) || !usable_region(&mbi))
        return FALSE;
    run_start = (UINT64)(ULONG_PTR)mbi.BaseAddress;
    run_end = run_start + mbi.RegionSize;
    /* Not back past the allocation's base, as the walk from the base never went; with reach_back,
     * at least ADOPT_MIN back. Physical pages are mapped as views of 64 KiB each, every one its own
     * allocation: a request near the end of a run of them stopped at its own view, found less than
     * ADOPT_MIN, and the data there (a mesh draw's root view) stayed out of GPU memory for good. */
    reach = address > ADOPT_MIN ? (address - ADOPT_MIN) & ~(ADOPT_ALIGN - 1) : 0;
    own_lo = max(lo, reach_back ? min((UINT64)(ULONG_PTR)mbi.AllocationBase, reach) : (UINT64)(ULONG_PTR)mbi.AllocationBase);
    /* Past that, from a physical page's view on down through the views below it (mapped, each its
     * own 64 KiB allocation), to lo: a chunk that began at its own view left the views just below
     * it a hole under 2 MiB between the run's start and GPU memory, and nothing in them could be
     * made GPU memory again (a world's root views and guest buffers there read as empty, a
     * thousand times a second). */
    mapped = mbi.Type == MEM_MAPPED;
    while (run_start > lo && VirtualQuery((void *)(ULONG_PTR)(run_start - 1), &mbi, sizeof(mbi))
           && usable_region(&mbi))
    {
        UINT64 below = (UINT64)(ULONG_PTR)mbi.BaseAddress;
        BOOL views = mapped && mbi.Type == MEM_MAPPED;
        if (run_start <= own_lo && !views)
            break;
        if (below < own_lo && !views)
        {
            run_start = own_lo;
            break;
        }
        mapped = mbi.Type == MEM_MAPPED;
        run_start = below;
    }
    while (run_end < hi && VirtualQuery((void *)(ULONG_PTR)run_end, &mbi, sizeof(mbi)) && usable_region(&mbi))
        run_end = (UINT64)(ULONG_PTR)mbi.BaseAddress + mbi.RegionSize;
    *start = max(run_start, lo);
    *end = min(run_end, hi);
    return TRUE;
}

/* Ranges that couldn't be adopted aren't tried again for a second, by their first 64 KiB page:
 * every root view bound asks, and each try is at least a VirtualQuery, slow in title memory full
 * of physical-page views (a world's frames spent their time there). Racy by design: a torn entry
 * only means another try. */
static struct { UINT64 page, end; ULONGLONG at; } adopt_failures[4096];

static UINT adopt_failure_slot(UINT64 page)
{
    return (UINT)((page * 0x9e3779b97f4a7c15ull) >> 52);
}

static BOOL adopt_failed_recently(UINT64 address, UINT64 end)
{
    UINT64 page = address >> 16;
    UINT slot = adopt_failure_slot(page);
    return adopt_failures[slot].page == page && end <= adopt_failures[slot].end
            && GetTickCount64() - adopt_failures[slot].at < 1000;
}

static void note_adopt_failed(UINT64 address, UINT64 end)
{
    UINT64 page = address >> 16;
    UINT slot = adopt_failure_slot(page);
    adopt_failures[slot].page = page;
    adopt_failures[slot].end = (end + 0xffff) & ~0xffffull;
    adopt_failures[slot].at = GetTickCount64();
}

/* Why requests weren't made GPU memory, since the last gpu_memory_adopt_failures_log: the first
 * logs were spent long before a world (8 in all), and a mesh draw's root view that was read as
 * zeros left the depth buffer at its clear value with no word of why. */
enum { ADOPT_FAIL_RECENT, ADOPT_FAIL_UNCOMMITTED, ADOPT_FAIL_PAST_RUN, ADOPT_FAIL_STRADDLE, ADOPT_FAIL_SPAN,
       ADOPT_FAIL_MAP, ADOPT_FAILS };
static const char *const adopt_fail_names[ADOPT_FAILS] = {
    "failed within the last second", "not committed memory", "running past the committed run",
    "straddling GPU memory", "too small a span between GPU memory or the run's ends", "the layer didn't map it" };
static volatile LONG adopt_fails[ADOPT_FAILS];
static volatile LONG64 adopt_fail_example[ADOPT_FAILS], adopt_fail_example_end[ADOPT_FAILS];
static volatile LONG adopt_trimmed;
/* What the last request not in committed memory found at its address (VirtualQuery's state,
 * protection and type): not committed, read-only, or physical pages' views not mapped yet. */
static volatile DWORD uncommitted_state, uncommitted_protect, uncommitted_type;

static void adopt_fail(UINT reason, UINT64 address, UINT64 end)
{
    InterlockedIncrement(&adopt_fails[reason]);
    InterlockedExchange64(&adopt_fail_example[reason], (LONG64)address);
    InterlockedExchange64(&adopt_fail_example_end[reason], (LONG64)end);
}

void gpu_memory_adopt_failures_log(void)
{
    char line[1024];
    int at = 0;
    LONG trimmed = InterlockedExchange(&adopt_trimmed, 0);

    for (UINT r = 0; r < ADOPT_FAILS && at < (int)sizeof(line) - 160; r++)
    {
        LONG n = InterlockedExchange(&adopt_fails[r], 0);
        if (n)
            at += snprintf(line + at, sizeof(line) - at, "%s %ld %s (the last %#llx-%#llx)", at ? ";" : "", n,
                           adopt_fail_names[r], (unsigned long long)adopt_fail_example[r],
                           (unsigned long long)adopt_fail_example_end[r]);
        if (n && r == ADOPT_FAIL_UNCOMMITTED && at < (int)sizeof(line) - 80)
            at += snprintf(line + at, sizeof(line) - at, " (state %#lx, protection %#lx, type %#lx there)",
                           uncommitted_state, uncommitted_protect, uncommitted_type);
    }
    line[at] = 0;
    if (at || trimmed)
        rebox_log("d3d12_xs:   title memory not made GPU memory since the last census:%s; %ld requests of a "
                  "guessed size cut to what could be", at ? line : " none", trimmed);
}

static BOOL adopt(UINT64 address, UINT64 size, BOOL trim);

/* Makes [address, address + size) of the title's own memory GPU-visible at the same address: a
 * heap whose memory VK_LAYER_REBOX_gpuva_identity puts at that device address, fills with the
 * title's bytes and maps over the title's pages in one go (asked through the mailbox,
 * layers/gpuva_mailbox.h). TRUE if the range is GPU memory now. */
BOOL gpu_memory_adopt(UINT64 address, UINT64 size)
{
    return adopt(address, size, FALSE);
}

/* For a request whose size is a guess (a root view's, a guest buffer's 64 KiB): as much of it from
 * address on as can be, the end cut to the committed run's and to GPU memory above it. TRUE if
 * some of it, from address on, is GPU memory now. */
BOOL gpu_memory_adopt_some(UINT64 address, UINT64 size)
{
    return adopt(address, size, TRUE);
}

static BOOL adopt(UINT64 address, UINT64 size, BOOL trim)
{
    static struct gpuva_mailbox *(WINAPI *get_mailbox)(void);
    static LONG logged;
    ID3D12Device *device = rebox_shared_device();
    UINT64 end = address + size, run_start, run_end, base, limit, bytes, asked, asked_end;
    struct gpuva_mailbox *box;
    struct chunk *c;
    BOOL ok = FALSE, held_head, straddles = FALSE;
    static int enabled = -1;

    /* REBOX_ADOPT=0: none, to tell problems of memory replaced in place from others. */
    if (enabled < 0)
    {
        const char *v = getenv("REBOX_ADOPT");
        enabled = !v || strcmp(v, "0");
    }
    if (!enabled)
        return FALSE;
    if (!get_mailbox)
    {
        HMODULE xmem = GetModuleHandleA("xmem.dll");
        get_mailbox = xmem ? (void *)GetProcAddress(xmem, "ReboxGpuvaMailbox") : NULL;
    }
    if (!device || !size || !get_mailbox || !(box = get_mailbox()) || box->magic != GPUVA_MAILBOX_MAGIC)
        return FALSE;
    asked = address;
    asked_end = end;
    /* Already GPU memory, the usual case (every root view bound asks): before committed_run, whose
     * VirtualQuery calls made a frame's bindings take seconds once a world streams in. */
    if (lock_spans())
    {
        /* As the walks below, by binary search. */
        const struct span *s = span_at(address);
        BOOL held = s && end <= s->end;
        if (!held)
            while (address < end && (s = span_at(address)))
                address = s->end;
        ReleaseSRWLockShared(&lock);
        if (held || address >= end)
            return TRUE;
        c = NULL;
        goto adopt;
    }
    for (c = chunks; c && !(address >= c->gpu && end <= c->gpu + c->size); c = c->next)
        ;
    /* A request that starts in GPU memory and runs past it: only the rest is to be made so (a
     * resource straddling a chunk's end was refused, and the GPU read its tail unmapped). */
    if (!c)
    {
        BOOL moved;
        do
        {
            moved = FALSE;
            for (struct chunk *k = chunks; k; k = k->next)
                if (address >= k->gpu && address < k->gpu + k->size)
                {
                    address = k->gpu + k->size;
                    moved = TRUE;
                }
        } while (moved && address < end);
    }
    ReleaseSRWLockShared(&lock);
    if (c || address >= end)
        return TRUE;
adopt:
    size = end - address;
    /* The span: the granule-aligned neighbourhood, inside the run, 2 MiB-aligned inwards. */
    base = address & ~(ADOPT_GRANULE - 1);
    limit = (end + ADOPT_GRANULE - 1) & ~(ADOPT_GRANULE - 1);
    if (limit - end < ADOPT_MIN) /* near the granule's end: into the next one too */
        limit += ADOPT_GRANULE;
    /* With trim, FALSE only if nothing from the address asked for on is GPU memory. */
    held_head = trim && address > asked;
    if (adopt_failed_recently(address, end))
    {
        adopt_fail(ADOPT_FAIL_RECENT, address, end);
        return held_head;
    }
    /* No further than the GPU memory on either side (the walk through physical pages' views is a
     * VirtualQuery each 64 KiB): the span stops there anyway, below. */
    AcquireSRWLockShared(&lock);
    for (c = chunks; c; c = c->next)
    {
        if (c->gpu + c->size <= address && c->gpu + c->size > base)
            base = c->gpu + c->size;
        if (c->gpu >= end && c->gpu < limit)
            limit = c->gpu;
    }
    ReleaseSRWLockShared(&lock);
    if (!committed_run(address, base, limit, FALSE, &run_start, &run_end))
    {
        MEMORY_BASIC_INFORMATION mbi;
        adopt_fail(ADOPT_FAIL_UNCOMMITTED, address, end);
        note_adopt_failed(address, end);
        if (VirtualQuery((void *)(ULONG_PTR)address, &mbi, sizeof(mbi)))
        {
            uncommitted_state = mbi.State;
            uncommitted_protect = mbi.Protect;
            uncommitted_type = mbi.Type;
        }
        return held_head;
    }
    if (end > run_end && trim && run_end > address)
    {
        end = run_end;
        InterlockedIncrement(&adopt_trimmed);
    }
    if (end > run_end)
    {
        adopt_fail(ADOPT_FAIL_PAST_RUN, address, end);
        note_adopt_failed(address, end);
        return held_head;
    }
    /* Too little around it from its own allocation on (a physical page's view): back past it. */
    if (min(limit, run_end) - max(base, run_start) < ADOPT_MIN)
    {
        UINT64 back_start, back_end;
        if (committed_run(address, base, limit, TRUE, &back_start, &back_end) && back_start < run_start)
            run_start = back_start;
    }
    if (base < run_start)
        base = run_start;
    if (limit > run_end)
        limit = run_end;
    base = (base + ADOPT_ALIGN - 1) & ~(ADOPT_ALIGN - 1);
    limit &= ~(ADOPT_ALIGN - 1);

    AcquireSRWLockExclusive(&lock);
    /* With trim: up to GPU memory above, rather than straddling it (a root view's guessed 64 KiB
     * ran into the next chunk, and none of it was made GPU memory). */
    if (trim)
        for (c = chunks; c; c = c->next)
            if (c->gpu > address && c->gpu < end)
            {
                end = c->gpu;
                InterlockedIncrement(&adopt_trimmed);
            }
    /* Already GPU memory, wholly; otherwise keep clear of the chunks there are. */
    for (c = chunks; c; c = c->next)
    {
        if (address >= c->gpu && end <= c->gpu + c->size)
        {
            ReleaseSRWLockExclusive(&lock);
            return TRUE;
        }
        /* Touching counts: ADOPT_GAP apart (none with the patched RADV). */
        if (c->gpu <= limit && base <= c->gpu + c->size)
        {
            if (c->gpu + c->size <= address)
                base = ((c->gpu + c->size + ADOPT_ALIGN - 1) & ~(ADOPT_ALIGN - 1)) + ADOPT_GAP;
            else if (c->gpu >= end)
                limit = (c->gpu & ~(ADOPT_ALIGN - 1)) - ADOPT_GAP;
            else
            {
                limit = base; /* the request straddles a chunk: nothing to do */
                straddles = TRUE;
            }
        }
    }
    if (base > address || limit < end || limit - base < adopt_min_span())
    {
        ReleaseSRWLockExclusive(&lock);
        adopt_fail(straddles ? ADOPT_FAIL_STRADDLE : ADOPT_FAIL_SPAN, address, end);
        note_adopt_failed(address, asked_end);
        if (InterlockedIncrement(&logged) <= 8)
            rebox_log("d3d12_xs: title memory %#llx+%#llx can't be made GPU-visible in place (committed run "
                      "%#llx-%#llx, span %#llx-%#llx)", (unsigned long long)address, (unsigned long long)size,
                      (unsigned long long)run_start, (unsigned long long)run_end, (unsigned long long)base,
                      (unsigned long long)limit);
        return held_head;
    }
    bytes = limit - base;
    /* The layer copies the title's bytes in before it maps the memory over them. */
    box->result = 0;
    box->size = bytes;
    MemoryBarrier();
    box->address = base;
    if (bytes < ADOPT_MIN)
    {
        whole_heap_asked = 0;
        whole_heap_thread = GetCurrentThreadId();
    }
    c = chunk_create(device, bytes, TRUE, TRUE);
    box->address = 0;
    if (bytes < ADOPT_MIN)
    {
        whole_heap_thread = 0;
        if (whole_heaps < 0 || (whole_heaps && !whole_heap_asked))
        {
            /* Not asked: vkd3d-proton without patch 0033, which suballocated the heap; no more
             * spans under 2 MiB. */
            whole_heaps = whole_heap_asked;
            rebox_log("d3d12_xs: title memory spans under 2 MiB %s (%#llx+%#llx)",
                      whole_heaps ? "are made GPU memory as heaps of their own"
                                  : "can't be made GPU memory: vkd3d-proton suballocates such heaps (it lacks "
                                    "patch 0033)", (unsigned long long)base, (unsigned long long)bytes);
        }
    }
    if (c && (UINT64)(ULONG_PTR)c->cpu == base && c->gpu == base && box->result == base)
    {
        c->adopted = TRUE;
        c->used = bytes;
        c->free_count = 0; /* nothing to suballocate */
        c->next = chunks;
        chunks = c;
        InterlockedIncrement(&chunks_generation);
        ok = TRUE;
    }
    else if (c) /* mapped elsewhere: the title's pages are untouched */
        chunk_destroy(c);
    ReleaseSRWLockExclusive(&lock);
    /* Physical pages mapped there are a copy in the heap now, not views of their pages: xmem
     * writes what DirectStorage puts in those pages here too. */
    if (ok)
    {
        static void (WINAPI *adopted)(ULONGLONG, ULONGLONG);
        static BOOL resolved;
        if (!resolved)
        {
            HMODULE xmem = GetModuleHandleA("xmem.dll");
            adopted = xmem ? (void *)GetProcAddress(xmem, "ReboxPhysicalAdopted") : NULL;
            resolved = TRUE;
        }
        if (adopted)
            adopted(base, bytes);
    }
    if (InterlockedIncrement(&logged) <= 8)
        rebox_log("d3d12_xs: title memory %#llx+%#llx %s GPU-visible in place (for %#llx+%#llx)",
                  (unsigned long long)base, (unsigned long long)bytes, ok ? "made" : "could not be made",
                  (unsigned long long)address, (unsigned long long)size);
    if (!ok)
        adopt_fail(ADOPT_FAIL_MAP, address, end);
    return ok || held_head;
}

BOOL gpu_memory_identity_extent(UINT64 address, UINT64 *end)
{
    BOOL found = FALSE;

    if (lock_spans())
    {
        const struct span *s = span_at(address);
        if (s && s->identity)
        {
            *end = s->end;
            found = TRUE;
        }
        ReleaseSRWLockShared(&lock);
        return found;
    }
    for (struct chunk *c = chunks; c; c = c->next)
        if ((UINT64)(ULONG_PTR)c->cpu == c->gpu && address >= c->gpu && address < c->gpu + c->size)
        {
            *end = c->gpu + c->size;
            found = TRUE;
            break;
        }
    ReleaseSRWLockShared(&lock);
    return found;
}

BOOL gpu_memory_identity_run_end(UINT64 address, UINT64 *end)
{
    BOOL found = FALSE, moved;

    if (lock_spans())
    {
        const struct span *s;
        while ((s = span_at(address)) && s->identity)
        {
            address = s->end;
            found = TRUE;
        }
        ReleaseSRWLockShared(&lock);
        if (found)
            *end = address;
        return found;
    }
    do
    {
        moved = FALSE;
        for (struct chunk *c = chunks; c; c = c->next)
            if ((UINT64)(ULONG_PTR)c->cpu == c->gpu && address >= c->gpu && address < c->gpu + c->size)
            {
                address = c->gpu + c->size;
                found = moved = TRUE;
            }
    } while (moved);
    ReleaseSRWLockShared(&lock);
    if (found)
        *end = address;
    return found;
}

BOOL gpu_memory_heap_at(UINT64 address, ID3D12Heap **heap, UINT64 *offset)
{
    BOOL found = FALSE;

    AcquireSRWLockShared(&lock);
    for (struct chunk *c = chunks; c; c = c->next)
        if (address >= c->gpu && address < c->gpu + c->size)
        {
            ID3D12Heap_AddRef(c->heap);
            *heap = c->heap;
            *offset = address - c->gpu;
            found = TRUE;
            break;
        }
    ReleaseSRWLockShared(&lock);
    return found;
}
