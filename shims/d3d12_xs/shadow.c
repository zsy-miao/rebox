/* Console-format shadow heaps (docs/anyps5.md).
 *
 * AnyPS5 reads descriptors as the console wrote them (RDNA2 T#/V#/S#), while
 * vkd3d-proton's heaps hold the host driver's own format. So every CBV/SRV/UAV
 * and sampler heap gets a shadow: the same number of descriptors at the same
 * stride (the increment the title gets from GetDescriptorHandleIncrementSize),
 * filled in console format (console_desc.c) whenever the title writes a
 * descriptor through the API (Create*View, CreateSampler, CopyDescriptors*).
 * A heap placed in title memory (shadow_place_heap) keeps its shadow there, as the console keeps
 * descriptors in the heap's memory: bytes the title writes into it itself are seen too. For the
 * GPU, a shader-visible one's pages that are GPU memory where they are (gpu_memory_adopt) are read
 * there; the rest from a mirror refreshed before each submission (shadow_sync_mirrors).
 *
 * Shader-visible heaps also get a title address: a 32-bit address of our
 * choosing (from 0x10000000, 64 KiB aligned) that descriptor-table pointers
 * are built from, and that ReboxShadowRead resolves for the recompiler.
 * Heap slot 10 reports it (proxy.c), and vkd3d-proton packs console SGPRs
 * with it. REBOX_HEAP_VA=real or virtual turns title addresses off; the
 * shadows are still kept.
 * Their shadows live in GPU-visible memory (ReboxGpuAlloc) so shaders can
 * later read them directly; other heaps' shadows are plain memory.
 *
 * AnyPS5 returns each binding by value (the descriptor words it resolved).
 * ReboxShadowFind maps those words back to the vkd3d-proton descriptor
 * that was written alongside them, through a hash of every shadow slot. */

#define COBJMACROS
#define WIDL_C_INLINE_WRAPPERS
#include <windows.h>
#include <d3d12.h>
#include <stdlib.h>
#include <string.h>

#include "../common/rebox_log.h"
#include "console_desc.h"
#include "gpu_memory.h"
#include "shadow.h"

#define TITLE_BASE 0x10000000ull
#define TITLE_ALIGN 0x10000ull
#define BUCKETS (1u << 18)
#define INDEXES 3 /* exact; loose (T#/S# with rewritable fields masked); loose V# (its first 3 words) */

static volatile LONG placed_heaps; /* heaps with title_cpu: no translation while 0 */

struct shadow_heap
{
    ID3D12DescriptorHeap *inner;   /* not referenced: only compared */
    D3D12_DESCRIPTOR_HEAP_TYPE type;
    BOOL shader_visible, gpu_memory;
    BOOL data_in_title;            /* data is title_cpu's memory: not ours to free */
    BYTE *mirror;                  /* GPU memory for the pages of it that aren't GPU memory, or NULL */
    UINT64 mirror_va;
    SIZE_T cpu_start;              /* vkd3d-proton's CPU handle for descriptor 0 */
    SIZE_T title_cpu;              /* the title's CPU handle for it, when its memory is the title's
                                    * (shadow_place_heap), else 0 */
    ULONGLONG title_attributes;    /* what title_cpu was allocated with */
    UINT stride, count;
    BYTE *data;                    /* count * stride bytes, console format */
    BYTE *views;                   /* count bytes: enum shadow_view of each slot */
    struct uav_texture *uavs;      /* count entries, made on the first texture UAV write */
    UINT64 title_va, gpu_va;       /* title_va 0 for heaps that aren't shader-visible */
    UINT reserved, reserved_used;  /* hidden slots after count in vkd3d-proton's heap */
    UINT32 (*reserved_words)[4];   /* the S# each used one holds */
    UINT *reserved_head, *reserved_next; /* reserved_words by hash (reserved_find), or NULL: walked */
    UINT reserved_buckets;
    ID3D12Resource **res;          /* CBV/SRV/UAV: count entries, the resource each slot's view is of
                                    * (not referenced), made on the first view of one */
    ID3D12Resource **reserved_res; /* the same for the reserved tail (shadow_reserve_srv) */
    struct shadow_heap *next;
};

/* A texture UAV's resource and view, so a sampled view of it can be made (AnyPS5 binds a
 * storage image as a sampled image too, for size queries). The resource isn't referenced: a slot
 * is only used while the title keeps its resource alive. */
struct uav_texture
{
    ID3D12Resource *res;
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc;
    BOOL has_desc;
};

struct entry
{
    UINT32 key[8];
    struct shadow_heap *heap;
    UINT index;
    struct entry *next;
};

static SRWLOCK lock = SRWLOCK_INIT;
static struct shadow_heap *heaps;

/* Resources whose views in descriptor slots are counted (shadow_track): a texture made for views
 * of memory is retired when the title puts another texture over its memory (proxy.c), and released
 * only once no slot holds a view of it. Released while a slot still did, a shader that read that
 * slot faulted the GPU. An open-addressed table under the lock: NULL empty, TOMBSTONE removed. */
#define TOMBSTONE ((ID3D12Resource *)1)
static struct tracked { ID3D12Resource *res; LONG refs; } *tracked;
static SIZE_T tracked_capacity, tracked_count, tracked_used;
static struct shadow_heap *heap_for_cpu(SIZE_T cpu, UINT *index);

static SIZE_T tracked_hash(const ID3D12Resource *res)
{
    return (SIZE_T)(((UINT64)(ULONG_PTR)res >> 4) * 0x9e3779b97f4a7c15ull >> 20);
}

/* Under the lock. */
static struct tracked *tracked_find(const ID3D12Resource *res)
{
    if (!tracked_count || !res)
        return NULL;
    for (SIZE_T i = tracked_hash(res) & (tracked_capacity - 1), n = 0; n < tracked_capacity;
            i = (i + 1) & (tracked_capacity - 1), n++)
        if (tracked[i].res == res)
            return &tracked[i];
        else if (!tracked[i].res)
            return NULL;
    return NULL;
}

/* Under the lock, exclusive: a slot's view is of res now, not old. */
static void ref_change(ID3D12Resource *old, ID3D12Resource *res)
{
    struct tracked *t;
    if (old == res || !tracked_count)
        return;
    if ((t = tracked_find(old)))
        t->refs--;
    if ((t = tracked_find(res)))
        t->refs++;
}

/* Under the lock, exclusive. */
static void slot_set_res(struct shadow_heap *h, UINT index, ID3D12Resource *res)
{
    ID3D12Resource *old = h->res ? h->res[index] : NULL;
    if (old == res || h->type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        return;
    if (!h->res && !(h->res = calloc(h->count, sizeof(*h->res))))
        return;
    ref_change(old, res);
    h->res[index] = res;
}

void shadow_track(ID3D12Resource *res)
{
    if (!res)
        return;
    AcquireSRWLockExclusive(&lock);
    if ((tracked_used + 1) * 2 > tracked_capacity)
    {
        SIZE_T n = tracked_capacity ? tracked_capacity : 1024;
        struct tracked *grown, *old = tracked;
        while ((tracked_count + 1) * 4 > n)
            n *= 2;
        if ((grown = calloc(n, sizeof(*grown))))
        {
            SIZE_T old_capacity = tracked_capacity;
            tracked = grown;
            tracked_capacity = n;
            tracked_used = tracked_count;
            for (SIZE_T i = 0; i < old_capacity; i++)
                if (old[i].res && old[i].res != TOMBSTONE)
                {
                    SIZE_T j = tracked_hash(old[i].res) & (n - 1);
                    while (tracked[j].res)
                        j = (j + 1) & (n - 1);
                    tracked[j] = old[i];
                }
            free(old);
        }
    }
    if (tracked_capacity && !tracked_find(res) && tracked_used < tracked_capacity - 1)
    {
        SIZE_T i = tracked_hash(res) & (tracked_capacity - 1);
        while (tracked[i].res && tracked[i].res != TOMBSTONE)
            i = (i + 1) & (tracked_capacity - 1);
        if (!tracked[i].res)
            tracked_used++;
        tracked[i].res = res;
        tracked[i].refs = 0;
        tracked_count++;
        /* Views of it written before it was tracked (none: proxy.c tracks it as it is made). */
    }
    ReleaseSRWLockExclusive(&lock);
}

BOOL shadow_untrack_unreferenced(ID3D12Resource *res)
{
    struct tracked *t;
    BOOL done = TRUE;

    AcquireSRWLockExclusive(&lock);
    if ((t = tracked_find(res)))
    {
        if (t->refs > 0)
            done = FALSE;
        else
        {
            t->res = TOMBSTONE;
            tracked_count--;
        }
    }
    ReleaseSRWLockExclusive(&lock);
    return done;
}

LONG shadow_view_refs(ID3D12Resource *res)
{
    struct tracked *t;
    LONG refs = -1;

    AcquireSRWLockShared(&lock);
    if ((t = tracked_find(res)))
        refs = t->refs;
    ReleaseSRWLockShared(&lock);
    return refs;
}

/* For tests: the resource the view in the slot at cpu is of. */
ID3D12Resource *WINAPI ReboxShadowSlotResource(SIZE_T cpu)
{
    ID3D12Resource *res = NULL;
    struct shadow_heap *h;
    UINT index;

    cpu = shadow_cpu_to_inner(cpu);
    AcquireSRWLockShared(&lock);
    if ((h = heap_for_cpu(cpu, &index)) && h->res)
        res = h->res[index];
    ReleaseSRWLockShared(&lock);
    return res;
}
static struct entry *buckets[INDEXES][BUCKETS];
static UINT64 next_title_va = TITLE_BASE;

/* REBOX_HEAP_VA: title (default), real or virtual; REBOX_REAL_HEAP_VA=0
 * is the older spelling of virtual. */
static BOOL CALLBACK read_heap_va_mode(INIT_ONCE *once, void *param, void **ctx)
{
    char *mode = rebox_getenv("REBOX_HEAP_VA"), *legacy = rebox_getenv("REBOX_REAL_HEAP_VA");
    enum shadow_heap_va *out = param;

    *out = SHADOW_HEAP_VA_TITLE;
    if (legacy && legacy[0] == '0')
        *out = SHADOW_HEAP_VA_VIRTUAL;
    if (mode && !strcmp(mode, "real"))
        *out = SHADOW_HEAP_VA_REAL;
    else if (mode && !strcmp(mode, "virtual"))
        *out = SHADOW_HEAP_VA_VIRTUAL;
    else if (mode && !strcmp(mode, "title"))
        *out = SHADOW_HEAP_VA_TITLE;
    free(mode);
    free(legacy);
    return TRUE;
}

enum shadow_heap_va shadow_heap_va_mode(void)
{
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    static enum shadow_heap_va mode;

    InitOnceExecuteOnce(&once, read_heap_va_mode, &mode, NULL);
    return mode;
}

BOOL shadow_title(SIZE_T cpu_start, UINT64 *title_va, UINT64 *size)
{
    BOOL found = FALSE;

    AcquireSRWLockShared(&lock);
    for (struct shadow_heap *h = heaps; h && !found; h = h->next)
        if (h->cpu_start == cpu_start && h->title_va)
        {
            *title_va = h->title_va;
            *size = (UINT64)h->count * h->stride;
            found = TRUE;
        }
    ReleaseSRWLockShared(&lock);
    return found;
}

static UINT desc_size(const struct shadow_heap *h)
{
    return h->type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ? 16 : 32;
}

/* The key for a slot: its first 8 dwords; samplers (4 dwords) are padded with
 * 0xffffffff so an S# never matches a V# with zero padding. */
static void make_key(const UINT32 *words, UINT word_count, BOOL sampler, UINT32 key[8])
{
    memset(key, sampler ? 0xff : 0, 8 * sizeof(*key));
    memcpy(key, words, (word_count < 8 ? word_count : 8) * sizeof(*key));
}

static UINT hash_key(const UINT32 key[8])
{
    UINT32 h = 2166136261u;
    for (int i = 0; i < 8; i++)
        h = (h ^ key[i]) * 16777619u;
    return h % BUCKETS;
}

static void slot_key(const struct shadow_heap *h, UINT index, UINT32 key[8])
{
    make_key((const UINT32 *)(h->data + (SIZE_T)index * h->stride), desc_size(h) / 4,
             h->type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, key);
}

static BOOL key_is_null(const UINT32 key[8])
{
    for (int i = 0; i < 8; i++)
        if (key[i] && key[i] != 0xffffffffu)
            return FALSE;
    return TRUE;
}

/* Caller holds the lock exclusively. */
/* The key of index `which`: 0 exact; 1 with the fields AnyPS5 may rewrite masked (a T#'s DST_SEL
 * in word 3, an S#'s word 2 bits 20-27: loose_match's); 2 a buffer descriptor's first 3 words (a
 * 4-word V# query ignores word 3). */
static void index_key(const UINT32 key[8], BOOL sampler, int which, UINT32 out[8])
{
    memcpy(out, key, 8 * sizeof(*out));
    if (which == 1 && sampler)
        out[2] &= ~(0xffu << 20);
    else if (which == 1)
        out[3] &= ~0xfffu;
    else if (which == 2)
        memset(out + 3, 0, 5 * sizeof(*out));
}

static void unindex(struct shadow_heap *h, UINT index)
{
    BOOL sampler = h->type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    UINT32 key[8], k[8];
    struct entry **p;

    slot_key(h, index, key);
    if (key_is_null(key))
        return;
    for (int which = 0; which < (sampler ? 2 : INDEXES); which++)
    {
        index_key(key, sampler, which, k);
        for (p = &buckets[which][hash_key(k)]; *p; p = &(*p)->next)
            if ((*p)->heap == h && (*p)->index == index)
            {
                struct entry *e = *p;
                *p = e->next;
                free(e);
                break;
            }
    }
}

static void index_slot(struct shadow_heap *h, UINT index)
{
    BOOL sampler = h->type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    UINT32 key[8];

    slot_key(h, index, key);
    if (key_is_null(key))
        return;
    for (int which = 0; which < (sampler ? 2 : INDEXES); which++)
    {
        struct entry *e;
        UINT b;
        if (!(e = malloc(sizeof(*e))))
            return;
        index_key(key, sampler, which, e->key);
        e->heap = h;
        e->index = index;
        b = hash_key(e->key);
        e->next = buckets[which][b];
        buckets[which][b] = e;
    }
}

/* Bumped under the lock after each change to the heap list or a heap's placement. */
static volatile LONG regions_generation;

LONG shadow_regions_generation(void)
{
    return regions_generation;
}

static void heap_free(struct shadow_heap *h)
{
    InterlockedIncrement(&regions_generation);
    for (UINT i = 0; i < h->count; i++)
        unindex(h, i);
    for (UINT i = 0; h->res && i < h->count; i++)
        ref_change(h->res[i], NULL);
    for (UINT i = 0; h->reserved_res && i < h->reserved_used; i++)
        ref_change(h->reserved_res[i], NULL);
    free(h->res);
    free(h->reserved_res);
    if (h->mirror)
        ReboxGpuFree(h->mirror);
    if (h->data_in_title)
        ;
    else if (h->gpu_memory)
        ReboxGpuFree(h->data);
    else
        free(h->data);
    free(h->reserved_words);
    free(h->reserved_head);
    free(h->reserved_next);
    free(h->views);
    free(h->uavs);
    free(h);
}

/* Caller holds the lock. */
static struct shadow_heap *heap_for_cpu(SIZE_T cpu, UINT *index)
{
    for (struct shadow_heap *h = heaps; h; h = h->next)
        if (cpu >= h->cpu_start && cpu - h->cpu_start < (SIZE_T)h->count * h->stride)
        {
            *index = (UINT)((cpu - h->cpu_start) / h->stride);
            return h;
        }
    return NULL;
}

void shadow_heap_created(ID3D12DescriptorHeap *inner, const D3D12_DESCRIPTOR_HEAP_DESC *desc, UINT increment,
        UINT reserved)
{
    struct shadow_heap *heap, **p;
    SIZE_T size;

    if (!desc || (desc->Type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && desc->Type != D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)
            || !desc->NumDescriptors || !(heap = calloc(1, sizeof(*heap))))
        return;
    heap->inner = inner;
    heap->type = desc->Type;
    heap->shader_visible = !!(desc->Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
    heap->cpu_start = ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(inner).ptr;
    heap->count = desc->NumDescriptors;
    heap->stride = increment;
    heap->reserved = reserved;
    if (increment < desc_size(heap))
    {
        static LONG logged;
        if (!InterlockedExchange(&logged, 1))
            rebox_log("d3d12_xs: descriptor increment %u is too small for console descriptors (%u bytes); "
                         "no shadow heaps", increment, desc_size(heap));
        free(heap);
        return;
    }
    size = (SIZE_T)heap->count * heap->stride;
    if (heap->shader_visible && SUCCEEDED(ReboxGpuAlloc(size, 256, (void **)&heap->data, &heap->gpu_va)))
    {
        heap->gpu_memory = TRUE;
        memset(heap->data, 0, size);
    }
    else if (!(heap->data = calloc(1, size)))
        ;
    if (heap->data && !(heap->views = calloc(1, heap->count)))
    {
        if (heap->gpu_memory)
            ReboxGpuFree(heap->data);
        else
            free(heap->data);
        heap->data = NULL;
    }
    if (!heap->data)
    {
        rebox_log("d3d12_xs: no memory for a %u-descriptor shadow heap", heap->count);
        free(heap);
        return;
    }

    AcquireSRWLockExclusive(&lock);
    /* A destroyed heap's object or CPU range can come back: forget it. */
    for (p = &heaps; *p;)
    {
        struct shadow_heap *h = *p;
        if (h->inner == inner || (h->cpu_start < heap->cpu_start + size
                                  && heap->cpu_start < h->cpu_start + (SIZE_T)h->count * h->stride))
        {
            *p = h->next;
            heap_free(h);
        }
        else
            p = &h->next;
    }
    if (heap->shader_visible && shadow_heap_va_mode() == SHADOW_HEAP_VA_TITLE)
    {
        heap->title_va = next_title_va;
        next_title_va = (next_title_va + size + TITLE_ALIGN - 1) & ~(TITLE_ALIGN - 1);
        if (next_title_va > 0xffffffffull)
        {
            rebox_log("d3d12_xs: shadow heap title addresses passed 4 GiB; wrapping");
            next_title_va = TITLE_BASE;
        }
    }
    heap->next = heaps;
    heaps = heap;
    InterlockedIncrement(&regions_generation);
    ReleaseSRWLockExclusive(&lock);
}

void shadow_heap_destroyed(ID3D12DescriptorHeap *inner)
{
    SIZE_T title_cpu = 0;
    ULONGLONG attributes = 0;

    AcquireSRWLockExclusive(&lock);
    for (struct shadow_heap **p = &heaps; *p; p = &(*p)->next)
        if ((*p)->inner == inner)
        {
            struct shadow_heap *h = *p;
            *p = h->next;
            title_cpu = h->title_cpu;
            attributes = h->title_attributes;
            if (title_cpu)
                InterlockedDecrement(&placed_heaps);
            heap_free(h);
            break;
        }
    ReleaseSRWLockExclusive(&lock);
    if (title_cpu)
    {
        void (WINAPI *xmem_free)(void *, ULONGLONG) =
                (void *)GetProcAddress(GetModuleHandleA("xmem.dll"), "XMemFree");
        if (xmem_free)
            xmem_free((void *)title_cpu, attributes);
    }
}

/* Descriptor heaps in the title's memory. On the console the runtime allocates a heap's memory
 * with XMemAlloc (object type 4, the D3D allocator, 128), so the title's allocation hooks can hand
 * it memory of its own: the target title registers an address for its main heap and asserts that
 * the heap's CPU start is that address. Here the heap gets title memory the same way and that is
 * its CPU start; the title's CPU handles are translated to vkd3d-proton's at each call
 * (shadow_cpu_to_inner). Only titles with allocation hooks (exe export __XMemAllocationHooks__). */
/* xcl's D3D_DESCRIPTOR_HEAP: object type 4, 64 KiB pages and alignment, the D3D allocator (128),
 * graphics memory. The title's handler compares the object type and the size. */
#define XALLOC_DESCRIPTOR_HEAP (4ull | 1ull << 14 | 0x80ull << 16 | 16ull << 24 | 1ull << 29)

void shadow_place_heap(ID3D12DescriptorHeap *inner, UINT count, UINT increment)
{
    static int hooks = -1;
    void *(WINAPI *xmem_alloc)(SIZE_T, ULONGLONG);
    SIZE_T size = (SIZE_T)count * increment;
    BOOL adopted;
    void *memory;

    if (hooks < 0)
        hooks = GetProcAddress(GetModuleHandleA(NULL), "__XMemAllocationHooks__") != NULL;
    if (!hooks || !size || !(xmem_alloc = (void *)GetProcAddress(GetModuleHandleA("xmem.dll"), "XMemAlloc"))
            || !(memory = xmem_alloc(size, XALLOC_DESCRIPTOR_HEAP)))
        return;
    {
        static LONG logged;
        if (InterlockedIncrement(&logged) <= 32)
            rebox_log("d3d12_xs: a descriptor heap of %u descriptors (%llu bytes) is in the title's memory at %p",
                      count, (unsigned long long)size, memory);
    }
    /* A shader-visible heap's memory is made GPU memory where it is, as far as it can be (outside
     * the lock: that makes D3D12 objects). */
    {
        BOOL visible = FALSE;
        AcquireSRWLockShared(&lock);
        for (struct shadow_heap *h = heaps; h; h = h->next)
            if (h->inner == inner)
                visible = h->shader_visible;
        ReleaseSRWLockShared(&lock);
        adopted = FALSE;
        if (visible && !(adopted = gpu_memory_adopt((UINT64)(ULONG_PTR)memory, size)))
        {
            /* Part of it may be GPU memory already (a buffer the title placed over its pool): the
             * rest as far as it goes, then the remainder through a mirror. */
            UINT64 at = (UINT64)(ULONG_PTR)memory, end = at + size, chunk_end;
            while (at < end)
            {
                UINT64 run = at;
                if (gpu_memory_identity_extent(at, &chunk_end))
                {
                    at = chunk_end;
                    continue;
                }
                while (run < end && !gpu_memory_identity_extent(run, &chunk_end))
                    run += 0x10000;
                if (run > end)
                    run = end;
                gpu_memory_adopt(at, run - at);
                at = run;
            }
        }
    }
    AcquireSRWLockExclusive(&lock);
    for (struct shadow_heap *h = heaps; h; h = h->next)
        if (h->inner == inner)
        {
            h->title_cpu = (SIZE_T)memory;
            h->title_attributes = XALLOC_DESCRIPTOR_HEAP;
            InterlockedIncrement(&placed_heaps);
            /* The shadow becomes the heap's memory (GPU-visible where it is, for a shader-visible
             * heap), so descriptors the title writes there itself are read as well. */
            if (h->stride == increment)
            {
                static LONG logged;
                memcpy(memory, h->data, size);
                /* Not wholly GPU memory: the old GPU copy becomes the mirror. */
                if (h->shader_visible && !adopted && h->gpu_memory)
                {
                    h->mirror = h->data;
                    h->mirror_va = h->gpu_va;
                }
                else if (h->gpu_memory)
                    ReboxGpuFree(h->data);
                else
                    free(h->data);
                h->data = memory;
                h->data_in_title = TRUE;
                if (h->shader_visible && adopted)
                {
                    h->gpu_memory = TRUE;
                    h->gpu_va = (UINT64)(ULONG_PTR)memory;
                }
                InterlockedIncrement(&regions_generation);
                if (InterlockedIncrement(&logged) <= 8)
                    rebox_log("d3d12_xs: the %u-descriptor heap's console descriptors are its memory at %p%s", count,
                              memory, !h->shader_visible ? "" : adopted ? " (GPU memory there)"
                              : h->mirror ? " (the GPU reads a mirror where it isn't GPU memory)" : " (not seen by the GPU)");
            }
            memory = NULL;
            break;
        }
    ReleaseSRWLockExclusive(&lock);
    if (memory)
    {
        void (WINAPI *xmem_free)(void *, ULONGLONG) =
                (void *)GetProcAddress(GetModuleHandleA("xmem.dll"), "XMemFree");
        if (xmem_free)
            xmem_free(memory, XALLOC_DESCRIPTOR_HEAP);
        return;
    }
}

BOOL shadow_title_heaps_placed(void)
{
    return placed_heaps > 0;
}

SIZE_T shadow_title_cpu_start(SIZE_T cpu_start)
{
    SIZE_T title = 0;

    if (!placed_heaps)
        return 0;
    AcquireSRWLockShared(&lock);
    for (struct shadow_heap *h = heaps; h && !title; h = h->next)
        if (h->cpu_start == cpu_start)
            title = h->title_cpu;
    ReleaseSRWLockShared(&lock);
    return title;
}

SIZE_T shadow_cpu_to_inner(SIZE_T cpu)
{
    SIZE_T inner = cpu;

    if (!placed_heaps)
        return cpu;
    AcquireSRWLockShared(&lock);
    for (struct shadow_heap *h = heaps; h; h = h->next)
        if (h->title_cpu && cpu - h->title_cpu < (SIZE_T)h->count * h->stride)
        {
            inner = h->cpu_start + (cpu - h->title_cpu);
            break;
        }
    ReleaseSRWLockShared(&lock);
    return inner;
}

/* What kind of view a write made: is_buffer -1 means "from the resource" (no view desc). */
static enum shadow_view view_kind(ID3D12Resource *res, int is_buffer, DXGI_FORMAT format, BOOL raw, BOOL uav)
{
    if (!res)
        return SHADOW_VIEW_NONE;
    if (is_buffer < 0)
    {
        D3D12_RESOURCE_DESC rd = ID3D12Resource_GetDesc(res);
        is_buffer = rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
        raw = is_buffer;
    }
    if (!is_buffer)
        return uav ? SHADOW_VIEW_UAV_TEXTURE : SHADOW_VIEW_SRV_TEXTURE;
    if (raw || format == DXGI_FORMAT_UNKNOWN)
        return uav ? SHADOW_VIEW_UAV_RAW : SHADOW_VIEW_SRV_RAW;
    return uav ? SHADOW_VIEW_UAV_TYPED : SHADOW_VIEW_SRV_TYPED;
}

/* GPU address ranges of buffers the title made views of, for AnyPS5's BDA page table: memory a
 * shader may reach by address. Kept after the views go: a buffer's range only matters while some
 * descriptor or pointer still names it. */
static SRWLOCK ranges_lock = SRWLOCK_INIT;
static struct { UINT64 begin, end; } *buffer_ranges;
static UINT buffer_range_count, buffer_range_capacity;

static volatile LONG ranges_generation;

LONG shadow_buffer_ranges_generation(void)
{
    return ranges_generation;
}

static void note_buffer_range(UINT64 va, UINT64 size)
{
    if (!va || !size)
        return;
    AcquireSRWLockExclusive(&ranges_lock);
    for (UINT i = 0; i < buffer_range_count; i++)
        if (buffer_ranges[i].begin == va)
        {
            if (buffer_ranges[i].end < va + size)
            {
                buffer_ranges[i].end = va + size;
                InterlockedIncrement(&ranges_generation);
            }
            ReleaseSRWLockExclusive(&ranges_lock);
            return;
        }
    if (buffer_range_count == buffer_range_capacity)
    {
        UINT capacity = buffer_range_capacity ? buffer_range_capacity * 2 : 256;
        void *grown = realloc(buffer_ranges, capacity * sizeof(*buffer_ranges));
        if (!grown)
        {
            ReleaseSRWLockExclusive(&ranges_lock);
            return;
        }
        buffer_ranges = grown;
        buffer_range_capacity = capacity;
    }
    buffer_ranges[buffer_range_count].begin = va;
    buffer_ranges[buffer_range_count++].end = va + size;
    InterlockedIncrement(&ranges_generation);
    ReleaseSRWLockExclusive(&ranges_lock);
}

static void note_buffer_view(ID3D12Resource *res)
{
    D3D12_RESOURCE_DESC rd;
    if (!res)
        return;
    rd = ID3D12Resource_GetDesc(res);
    if (rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
        note_buffer_range(ID3D12Resource_GetGPUVirtualAddress(res), rd.Width);
}

UINT shadow_buffer_ranges(UINT64 (*out)[2], UINT max)
{
    UINT n;

    AcquireSRWLockShared(&ranges_lock);
    n = min(buffer_range_count, max);
    for (UINT i = 0; i < n; i++)
    {
        out[i][0] = buffer_ranges[i].begin;
        out[i][1] = buffer_ranges[i].end;
    }
    ReleaseSRWLockShared(&ranges_lock);
    return n;
}

/* The first writes into shader-visible heaps, by title address: do the slots the console tables
 * point at get written, and before the draws that read them? */
static void note_visible_write(const struct shadow_heap *h, UINT index, const char *how)
{
    static LONG shown;
    if (h->title_va && InterlockedIncrement(&shown) <= 64)
        rebox_log("d3d12_xs: shadow write (%s) at %#llx: view %u, words %08x %08x %08x %08x", how,
                     (unsigned long long)(h->title_va + (UINT64)index * h->stride), h->views[index],
                     ((const UINT32 *)(h->data + (SIZE_T)index * h->stride))[0],
                     ((const UINT32 *)(h->data + (SIZE_T)index * h->stride))[1],
                     ((const UINT32 *)(h->data + (SIZE_T)index * h->stride))[2],
                     ((const UINT32 *)(h->data + (SIZE_T)index * h->stride))[3]);
}

static void write_slot(SIZE_T cpu, const UINT32 *words, UINT size, enum shadow_view view, ID3D12Resource *res)
{
    struct shadow_heap *h;
    UINT index;

    AcquireSRWLockExclusive(&lock);
    if ((h = heap_for_cpu(cpu, &index)))
    {
        unindex(h, index);
        memset(h->data + (SIZE_T)index * h->stride, 0, h->stride);
        memcpy(h->data + (SIZE_T)index * h->stride, words, size < h->stride ? size : h->stride);
        h->views[index] = view;
        slot_set_res(h, index, res);
        index_slot(h, index);
        note_visible_write(h, index, "view");
    }
    ReleaseSRWLockExclusive(&lock);
}

/* Whether [va, va + size) can go in the BDA page table as identity-mapped memory: a title address
 * (4 GiB to 2^46, outside the GPU window, as proxy.c's ia_adopt) only if it is GPU memory or can be
 * made so; anything else is vkd3d-proton's own. A constant buffer in title memory that isn't GPU
 * memory (pages of a render-target pool) was mapped anyway, and shaders reading through the table
 * faulted the GPU. */
static BOOL buffer_range_mappable(UINT64 va, UINT64 size)
{
    UINT64 end;
    if (va < 0x100000000ull || va + size > 0x400000000000ull
            || (va + size > 0x200000000000ull && va < 0x210000000000ull))
        return TRUE;
    if (gpu_memory_identity_extent(va, &end) && end >= va + size)
        return TRUE;
    return gpu_memory_adopt(va, size);
}

void shadow_write_cbv(SIZE_T cpu, const D3D12_CONSTANT_BUFFER_VIEW_DESC *desc)
{
    UINT32 w[8];
    if (desc && desc->BufferLocation && buffer_range_mappable(desc->BufferLocation, desc->SizeInBytes))
        note_buffer_range(desc->BufferLocation, desc->SizeInBytes);
    console_encode_cbv(desc, w);
    write_slot(cpu, w, sizeof(w), desc && desc->BufferLocation ? SHADOW_VIEW_CBV : SHADOW_VIEW_NONE, NULL);
}

void shadow_write_srv(SIZE_T cpu, ID3D12Resource *res, const D3D12_SHADER_RESOURCE_VIEW_DESC *desc)
{
    UINT32 w[8];
    note_buffer_view(res);
    console_encode_srv(res, desc, w);
    write_slot(cpu, w, sizeof(w), view_kind(res, desc ? desc->ViewDimension == D3D12_SRV_DIMENSION_BUFFER : -1,
               desc ? desc->Format : DXGI_FORMAT_UNKNOWN, desc && desc->ViewDimension == D3D12_SRV_DIMENSION_BUFFER
               ? (desc->Buffer.StructureByteStride || (desc->Buffer.Flags & D3D12_BUFFER_SRV_FLAG_RAW)) : 0, FALSE),
               res);
}

void shadow_write_uav(SIZE_T cpu, ID3D12Resource *res, const D3D12_UNORDERED_ACCESS_VIEW_DESC *desc)
{
    struct shadow_heap *h;
    UINT32 w[8];
    UINT index;

    AcquireSRWLockExclusive(&lock);
    if ((h = heap_for_cpu(cpu, &index)) && h->shader_visible && res
            && (h->uavs || (h->uavs = calloc(h->count, sizeof(*h->uavs)))))
    {
        h->uavs[index].res = res;
        h->uavs[index].has_desc = desc != NULL;
        if (desc)
            h->uavs[index].desc = *desc;
    }
    ReleaseSRWLockExclusive(&lock);
    note_buffer_view(res);
    console_encode_uav(res, desc, w);
    write_slot(cpu, w, sizeof(w), view_kind(res, desc ? desc->ViewDimension == D3D12_UAV_DIMENSION_BUFFER : -1,
               desc ? desc->Format : DXGI_FORMAT_UNKNOWN, desc && desc->ViewDimension == D3D12_UAV_DIMENSION_BUFFER
               ? (desc->Buffer.StructureByteStride || (desc->Buffer.Flags & D3D12_BUFFER_UAV_FLAG_RAW)) : 0, TRUE),
               res);
}

void shadow_write_sampler(SIZE_T cpu, const D3D12_SAMPLER_DESC *desc)
{
    UINT32 w[4];
    if (!desc)
        return;
    console_encode_sampler(desc, w);
    write_slot(cpu, w, sizeof(w), SHADOW_VIEW_SAMPLER, NULL);
}

void shadow_copy(SIZE_T dst, SIZE_T src, UINT count)
{
    AcquireSRWLockExclusive(&lock);
    for (UINT i = 0; i < count; i++)
    {
        struct shadow_heap *d, *s;
        UINT di, si;

        if (!(d = heap_for_cpu(dst, &di)))
            break;
        dst += d->stride;
        s = heap_for_cpu(src, &si);
        src += s ? s->stride : d->stride;
        unindex(d, di);
        if (s && s->stride == d->stride)
        {
            memcpy(d->data + (SIZE_T)di * d->stride, s->data + (SIZE_T)si * s->stride, d->stride);
            d->views[di] = s->views[si];
            slot_set_res(d, di, s->res ? s->res[si] : NULL);
        }
        else
        {
            memset(d->data + (SIZE_T)di * d->stride, 0, d->stride);
            d->views[di] = SHADOW_VIEW_NONE;
            slot_set_res(d, di, NULL);
        }
        index_slot(d, di);
        note_visible_write(d, di, "copy");
    }
    ReleaseSRWLockExclusive(&lock);
}

void shadow_copy_ranges(UINT dst_count, const D3D12_CPU_DESCRIPTOR_HANDLE *dst_starts, const UINT *dst_sizes,
        UINT src_count, const D3D12_CPU_DESCRIPTOR_HANDLE *src_starts, const UINT *src_sizes)
{
    UINT di = 0, si = 0, d_off = 0, s_off = 0;

    /* Walk both range lists one run at a time, as CopyDescriptors does. */
    while (di < dst_count && si < src_count)
    {
        UINT d_size = dst_sizes ? dst_sizes[di] : 1, s_size = src_sizes ? src_sizes[si] : 1;
        UINT n = min(d_size - d_off, s_size - s_off);
        struct shadow_heap *dh, *sh;
        UINT unused;

        AcquireSRWLockShared(&lock);
        dh = heap_for_cpu(dst_starts[di].ptr, &unused);
        sh = heap_for_cpu(src_starts[si].ptr, &unused);
        ReleaseSRWLockShared(&lock);
        if (dh && n)
            shadow_copy(dst_starts[di].ptr + (SIZE_T)d_off * dh->stride,
                        src_starts[si].ptr + (SIZE_T)s_off * (sh ? sh->stride : dh->stride), n);
        if ((d_off += n) == d_size) { di++; d_off = 0; }
        if ((s_off += n) == s_size) { si++; s_off = 0; }
    }
}

/* ------------------------------------------------- for console_compile.c */

UINT shadow_regions(struct shadow_region *out, UINT max)
{
    UINT n = 0;

    AcquireSRWLockShared(&lock);
    for (struct shadow_heap *h = heaps; h && n < max; h = h->next)
        if (h->title_va && h->mirror)
        {
            /* Runs of pages: GPU memory where they are, or the mirror. The mirrored ones are also
             * at the heap's own address: the console heap's GPU address is its memory's, and a
             * title computes descriptor addresses from it (bindless lookups); the rest is GPU
             * memory there already. */
            UINT64 base = (UINT64)(ULONG_PTR)h->data, size = (UINT64)h->count * h->stride, off = 0, end;
            while (off < size && n < max)
            {
                UINT64 next = off;
                BOOL there = gpu_memory_identity_extent(base + off, &end);
                if (there)
                    next = end - base < size ? end - base : size;
                else
                    while (next < size && !gpu_memory_identity_extent(base + next, &end))
                        next = ((base + next + 0x10000) & ~0xffffull) - base;
                if (next > size)
                    next = size;
                out[n].title_va = h->title_va + off;
                out[n].gpu_va = there ? base + off : h->mirror_va + off;
                out[n].data = h->data + off;
                out[n].size = (SIZE_T)(next - off);
                n++;
                if (!there && n < max)
                {
                    out[n] = out[n - 1];
                    out[n].title_va = base + off;
                    n++;
                }
                off = next;
            }
        }
        else if (h->title_va)
        {
            out[n].title_va = h->title_va;
            out[n].gpu_va = h->gpu_memory ? h->gpu_va : 0;
            out[n].data = h->data;
            out[n].size = (SIZE_T)h->count * h->stride;
            n++;
        }
    ReleaseSRWLockShared(&lock);
    return n;
}

/* Before a submission: the mirrored pages of heaps placed in title memory, from that memory. */
void shadow_sync_mirrors(void)
{
    AcquireSRWLockShared(&lock);
    for (struct shadow_heap *h = heaps; h; h = h->next)
        if (h->mirror)
        {
            UINT64 base = (UINT64)(ULONG_PTR)h->data, size = (UINT64)h->count * h->stride, off = 0, end;
            while (off < size)
            {
                UINT64 next;
                if (gpu_memory_identity_extent(base + off, &end))
                {
                    off = end - base;
                    continue;
                }
                next = ((base + off + 0x10000) & ~0xffffull) - base;
                if (next > size)
                    next = size;
                memcpy(h->mirror + off, h->data + off, (SIZE_T)(next - off));
                off = next;
            }
        }
    ReleaseSRWLockShared(&lock);
}

/* Fields AnyPS5 may rewrite in the words it binds: a T#'s DST_SEL (word 3 bits 0-11, made
 * identity when the view swizzle takes it over), an S#'s filters (word 2 bits 20-27, for the
 * point-filtered twin of each sampler) and a V#'s word 3 (format and DST_SEL, zeroed for buffers
 * it reads raw, such as constant buffers read with s_buffer_load; base, stride and size in words
 * 0-2 still identify the view). */
static BOOL loose_match(const UINT32 *slot, const UINT32 *words, UINT n, BOOL sampler)
{
    for (UINT i = 0; i < n; i++)
    {
        UINT32 mask = ~0u;
        if (sampler && i == 2)
            mask = ~(0xffu << 20);
        else if (!sampler && n == 8 && i == 3)
            mask = ~0xfffu;
        else if (!sampler && n == 4 && i == 3)
            mask = 0;
        if ((slot[i] ^ words[i]) & mask)
            return FALSE;
    }
    return TRUE;
}

BOOL shadow_find_index(SIZE_T heap_cpu, const UINT32 *words, UINT n, BOOL sampler, BOOL loose, UINT *index)
{
    return shadow_find_index_view(heap_cpu, words, n, sampler, loose, 0, index);
}

/* A slot's view is one of views (a mask of 1 << enum shadow_view; 0 takes any). */
static BOOL view_wanted(const struct entry *e, UINT views)
{
    return !views || (e->index < e->heap->count && (views & (1u << e->heap->views[e->index])));
}

BOOL shadow_find_index_view(SIZE_T heap_cpu, const UINT32 *words, UINT n, BOOL sampler, BOOL loose, UINT views,
                            UINT *index)
{
    UINT32 key[8];
    BOOL found = FALSE, wanted = FALSE;

    if (!words || !n || n > 8)
        return FALSE;
    make_key(words, n, sampler, key);
    AcquireSRWLockShared(&lock);
    /* The first slot with these words, or one holding a view of the wanted kind: a texture's
     * single-level SRV and UAV have the same T#, and a storage image bound to the SRV's slot (or a
     * sampled one to the UAV's) was refused. */
    for (struct entry *e = buckets[0][hash_key(key)]; e && !wanted; e = e->next)
        if (e->heap->cpu_start == heap_cpu && !memcmp(e->key, key, sizeof(key)) && (!found || view_wanted(e, views)))
        {
            *index = e->index;
            found = TRUE;
            wanted = view_wanted(e, views);
        }
    /* Loose: through the masked index, then loose_match on the candidates (heaps hold up to a
     * million descriptors; scanning one per binding made compiles slow). */
    if (loose && !wanted)
    {
        UINT32 k[8];
        index_key(key, sampler, !sampler && n == 4 ? 2 : 1, k);
        for (struct entry *e = buckets[!sampler && n == 4 ? 2 : 1][hash_key(k)]; e && !wanted; e = e->next)
            if (e->heap->cpu_start == heap_cpu && !memcmp(e->key, k, sizeof(k)) && (!found || view_wanted(e, views))
                    && loose_match((const UINT32 *)(e->heap->data + (SIZE_T)e->index * e->heap->stride), words, n,
                                   sampler))
            {
                *index = e->index;
                found = TRUE;
                wanted = view_wanted(e, views);
            }
    }
    ReleaseSRWLockShared(&lock);
    return found;
}

enum shadow_view shadow_slot_view(SIZE_T heap_cpu, UINT index)
{
    enum shadow_view view = SHADOW_VIEW_NONE;

    AcquireSRWLockShared(&lock);
    for (struct shadow_heap *h = heaps; h; h = h->next)
        if (h->cpu_start == heap_cpu)
        {
            if (index < h->count)
                view = h->views[index];
            else if (index < h->count + h->reserved_used)
                view = h->type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ? SHADOW_VIEW_SAMPLER : SHADOW_VIEW_SRV_TEXTURE;
            break;
        }
    ReleaseSRWLockShared(&lock);
    return view;
}

/* The reserved tail's slots by their words: a draw's T#s the title wrote itself, its dummy views and
 * the samplers made from literals each looked through every used slot, under the lock exclusive,
 * at every draw (thousands of slots once a world's textures are viewed there). */
static UINT reserved_hash(const struct shadow_heap *h, const UINT32 key[4])
{
    UINT64 a = key[0] | (UINT64)key[1] << 32, b = key[2] | (UINT64)key[3] << 32;
    UINT64 x = a * 0x9e3779b97f4a7c15ull ^ b * 0xc2b2ae3d27d4eb4full;
    x ^= x >> 29;
    return (UINT)((x * 0x9e3779b97f4a7c15ull) >> 32) & (h->reserved_buckets - 1);
}

/* Under the lock: the first used reserved slot holding these words, or ~0u. */
static UINT reserved_find(const struct shadow_heap *h, const UINT32 key[4])
{
    UINT found = ~0u;
    if (!h->reserved_words)
        return ~0u;
    if (!h->reserved_head)
    {
        for (UINT i = 0; i < h->reserved_used; i++)
            if (!memcmp(h->reserved_words[i], key, 16))
                return i;
        return ~0u;
    }
    /* A slot's words can change after it went in (shadow_forget_reserved): compared again. */
    for (UINT i = h->reserved_head[reserved_hash(h, key)]; i != ~0u; i = h->reserved_next[i])
        if (i < found && i < h->reserved_used && !memcmp(h->reserved_words[i], key, 16))
            found = i;
    return found;
}

/* Under the lock, exclusive: reserved slot i, its words just written, is found by them. The index is
 * made with the first slot; without memory for it, the slots are walked. */
static void reserved_note(struct shadow_heap *h, UINT i)
{
    UINT b;
    if (!i && !h->reserved_head)
    {
        UINT buckets = 16;
        while (buckets < h->reserved && buckets < (1u << 30))
            buckets <<= 1;
        h->reserved_head = malloc(buckets * sizeof(*h->reserved_head));
        h->reserved_next = malloc(h->reserved * sizeof(*h->reserved_next));
        if (!h->reserved_head || !h->reserved_next)
        {
            free(h->reserved_head);
            free(h->reserved_next);
            h->reserved_head = h->reserved_next = NULL;
        }
        else
        {
            memset(h->reserved_head, 0xff, buckets * sizeof(*h->reserved_head));
            h->reserved_buckets = buckets;
        }
    }
    if (!h->reserved_head || i >= h->reserved)
        return;
    b = reserved_hash(h, h->reserved_words[i]);
    h->reserved_next[i] = h->reserved_head[b];
    h->reserved_head[b] = i;
}

/* The reserved slot of the heap (of type) at heap_cpu holding these words, looked for under the lock
 * shared: the usual case at a draw, which then doesn't queue the other recording threads behind it. */
static BOOL reserved_lookup(SIZE_T heap_cpu, D3D12_DESCRIPTOR_HEAP_TYPE type, const UINT32 key[4], UINT *index)
{
    BOOL found = FALSE;
    AcquireSRWLockShared(&lock);
    for (struct shadow_heap *h = heaps; h; h = h->next)
        if (h->cpu_start == heap_cpu && h->type == type && h->reserved)
        {
            UINT i = reserved_find(h, key);
            if (i != ~0u)
            {
                *index = h->count + i;
                found = TRUE;
            }
            break;
        }
    ReleaseSRWLockShared(&lock);
    return found;
}

/* Whether a heap's reserved tail has room; the first times it doesn't, logged: what would have gone
 * there binds nothing. Under lock. */
static BOOL reserve_room(const struct shadow_heap *h)
{
    static LONG logged;
    if (h->reserved_used < h->reserved)
        return TRUE;
    if (h->reserved && InterlockedIncrement(&logged) <= 8)
        rebox_log("d3d12_xs: a shader-visible heap's %u reserved slots are all used: a descriptor the host makes "
                  "binds nothing", h->reserved);
    return FALSE;
}

UINT shadow_reserved(SIZE_T cpu_start)
{
    UINT reserved = 0;

    AcquireSRWLockShared(&lock);
    for (struct shadow_heap *h = heaps; h; h = h->next)
        if (h->cpu_start == cpu_start)
            reserved = h->reserved;
    ReleaseSRWLockShared(&lock);
    return reserved;
}

/* An SRV equivalent to a texture UAV's view. */
static void srv_from_uav(const struct uav_texture *u, D3D12_SHADER_RESOURCE_VIEW_DESC *srv)
{
    D3D12_RESOURCE_DESC rd = ID3D12Resource_GetDesc(u->res);
    const D3D12_UNORDERED_ACCESS_VIEW_DESC *d = &u->desc;
    D3D12_UAV_DIMENSION dim = u->has_desc ? d->ViewDimension
            : rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D ? (rd.DepthOrArraySize > 1 ? D3D12_UAV_DIMENSION_TEXTURE1DARRAY : D3D12_UAV_DIMENSION_TEXTURE1D)
            : rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? D3D12_UAV_DIMENSION_TEXTURE3D
            : rd.DepthOrArraySize > 1 ? D3D12_UAV_DIMENSION_TEXTURE2DARRAY : D3D12_UAV_DIMENSION_TEXTURE2D;

    memset(srv, 0, sizeof(*srv));
    srv->Format = u->has_desc && d->Format != DXGI_FORMAT_UNKNOWN ? d->Format : rd.Format;
    srv->Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    switch (dim)
    {
    case D3D12_UAV_DIMENSION_TEXTURE1D:
        srv->ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
        srv->Texture1D.MostDetailedMip = u->has_desc ? d->Texture1D.MipSlice : 0;
        srv->Texture1D.MipLevels = 1;
        break;
    case D3D12_UAV_DIMENSION_TEXTURE1DARRAY:
        srv->ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
        srv->Texture1DArray.MostDetailedMip = u->has_desc ? d->Texture1DArray.MipSlice : 0;
        srv->Texture1DArray.MipLevels = 1;
        srv->Texture1DArray.FirstArraySlice = u->has_desc ? d->Texture1DArray.FirstArraySlice : 0;
        srv->Texture1DArray.ArraySize = u->has_desc ? d->Texture1DArray.ArraySize : rd.DepthOrArraySize;
        break;
    case D3D12_UAV_DIMENSION_TEXTURE2DARRAY:
        srv->ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        srv->Texture2DArray.MostDetailedMip = u->has_desc ? d->Texture2DArray.MipSlice : 0;
        srv->Texture2DArray.MipLevels = 1;
        srv->Texture2DArray.FirstArraySlice = u->has_desc ? d->Texture2DArray.FirstArraySlice : 0;
        srv->Texture2DArray.ArraySize = u->has_desc ? d->Texture2DArray.ArraySize : rd.DepthOrArraySize;
        break;
    case D3D12_UAV_DIMENSION_TEXTURE3D:
        srv->ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        srv->Texture3D.MostDetailedMip = u->has_desc ? d->Texture3D.MipSlice : 0;
        srv->Texture3D.MipLevels = 1;
        break;
    default:
        srv->ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv->Texture2D.MostDetailedMip = u->has_desc ? d->Texture2D.MipSlice : 0;
        srv->Texture2D.MipLevels = 1;
        break;
    }
}

BOOL shadow_reserve_srv_for_uav(SIZE_T heap_cpu, UINT uav_index, UINT *index)
{
    ID3D12Device *device = rebox_shared_device();
    ID3D12Resource *res = NULL;
    struct shadow_heap *heap = NULL;
    BOOL found = FALSE;
    UINT32 key[4];

    AcquireSRWLockExclusive(&lock);
    for (struct shadow_heap *h = heaps; h && !heap; h = h->next)
        if (h->cpu_start == heap_cpu && h->type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && h->reserved)
            heap = h;
    if (heap && uav_index < heap->count && heap->uavs && heap->uavs[uav_index].res
            && heap->views[uav_index] == SHADOW_VIEW_UAV_TEXTURE)
    {
        const struct uav_texture *u = &heap->uavs[uav_index];
        /* Keyed by the UAV's resource and view, in the slot's first words. */
        UINT64 r = (UINT64)(ULONG_PTR)u->res;
        key[0] = (UINT32)r;
        key[1] = (UINT32)(r >> 32);
        key[2] = u->has_desc ? u->desc.Format : 0;
        key[3] = u->has_desc ? (UINT32)u->desc.ViewDimension | u->desc.Texture2DArray.MipSlice << 8
                 | u->desc.Texture2DArray.FirstArraySlice << 16 : ~0u;
        {
            UINT i = reserved_find(heap, key);
            if (i != ~0u)
            {
                *index = heap->count + i;
                found = TRUE;
            }
        }
        if (!found && device && reserve_room(heap)
                && (heap->reserved_words || (heap->reserved_words = calloc(heap->reserved, 16))))
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv;
            D3D12_CPU_DESCRIPTOR_HANDLE cpu;
            UINT i = heap->reserved_used++;

            srv_from_uav(u, &srv);
            memcpy(heap->reserved_words[i], key, 16);
            reserved_note(heap, i);
            *index = heap->count + i;
            cpu.ptr = heap->cpu_start + (SIZE_T)*index * heap->stride;
            ID3D12Device_CreateShaderResourceView(device, u->res, &srv, cpu);
            found = TRUE;
        }
    }
    /* A UAV the slot got by a copy from a CPU-only heap, which keeps no UAV descriptions: an SRV of
     * the whole texture it views. */
    if (!found && heap && uav_index < heap->count && heap->views[uav_index] == SHADOW_VIEW_UAV_TEXTURE && heap->res)
        res = heap->res[uav_index];
    ReleaseSRWLockExclusive(&lock);
    if (!found && res)
        found = shadow_reserve_srv(heap_cpu, res, index);
    return found;
}

/* A typed format for a texture's default view (typeless resources need one). */
static DXGI_FORMAT view_format(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_B8G8R8X8_TYPELESS: return DXGI_FORMAT_B8G8R8X8_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R8G8_TYPELESS: return DXGI_FORMAT_R8G8_UNORM;
    case DXGI_FORMAT_R16_TYPELESS: return DXGI_FORMAT_R16_FLOAT;
    case DXGI_FORMAT_R8_TYPELESS: return DXGI_FORMAT_R8_UNORM;
    case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R24G8_TYPELESS: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R32G8X24_TYPELESS: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_BC1_TYPELESS: return DXGI_FORMAT_BC1_UNORM;
    case DXGI_FORMAT_BC2_TYPELESS: return DXGI_FORMAT_BC2_UNORM;
    case DXGI_FORMAT_BC3_TYPELESS: return DXGI_FORMAT_BC3_UNORM;
    case DXGI_FORMAT_BC4_TYPELESS: return DXGI_FORMAT_BC4_UNORM;
    case DXGI_FORMAT_BC5_TYPELESS: return DXGI_FORMAT_BC5_UNORM;
    case DXGI_FORMAT_BC7_TYPELESS: return DXGI_FORMAT_BC7_UNORM;
    default: return f;
    }
}

/* A shader resource view of res in the heap's reserved tail, made once per resource: for a T# the
 * title wrote itself (not through the API), found by the texture's address (console_texture_of). */
BOOL shadow_reserve_srv(SIZE_T heap_cpu, ID3D12Resource *res, UINT *index)
{
    ID3D12Device *device = rebox_shared_device();
    struct shadow_heap *heap = NULL;
    BOOL found = FALSE;
    UINT32 key[4];
    UINT64 r = (UINT64)(ULONG_PTR)res;

    if (!res)
        return FALSE;
    key[0] = (UINT32)r;
    key[1] = (UINT32)(r >> 32);
    key[2] = 0x53525621; /* "!VRS": not the srv_for_uav keys, whose word 2 is a format */
    key[3] = 0;
    if (reserved_lookup(heap_cpu, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, key, index))
        return TRUE;
    AcquireSRWLockExclusive(&lock);
    for (struct shadow_heap *h = heaps; h && !heap; h = h->next)
        if (h->cpu_start == heap_cpu && h->type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && h->reserved)
            heap = h;
    if (heap)
    {
        UINT i = reserved_find(heap, key);
        if (i != ~0u)
        {
            *index = heap->count + i;
            found = TRUE;
        }
    }
    if (heap && !found && device && reserve_room(heap)
            && (heap->reserved_words || (heap->reserved_words = calloc(heap->reserved, 16))))
    {
        D3D12_RESOURCE_DESC rd;
        D3D12_SHADER_RESOURCE_VIEW_DESC srv = { 0 };
        D3D12_CPU_DESCRIPTOR_HANDLE cpu;
        UINT i;

        ((D3D12_RESOURCE_DESC *(STDMETHODCALLTYPE *)(void *, D3D12_RESOURCE_DESC *))(*(void ***)res)[10])(res, &rd);
        srv.Format = view_format(rd.Format);
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        /* The T#'s channel select is applied in the shader, to the texel as the console's hardware
         * reads it, in memory order. The console has no A8 format: an A8 texture's T# is an 8-bit
         * one taking alpha from its first channel, which the A8 view returns as 0, so a glyph
         * atlas drew nothing. Its view returns the byte first instead. */
        if (rd.Format == DXGI_FORMAT_A8_UNORM)
            srv.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(3,
                    D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0,
                    D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1);
        if (rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && rd.DepthOrArraySize > 1)
        {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            srv.Texture2DArray.MipLevels = rd.MipLevels;
            srv.Texture2DArray.ArraySize = rd.DepthOrArraySize;
        }
        else if (rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
        {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            srv.Texture3D.MipLevels = rd.MipLevels;
        }
        else if (rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Texture2D.MipLevels = rd.MipLevels;
        }
        if (srv.ViewDimension)
        {
            i = heap->reserved_used++;
            memcpy(heap->reserved_words[i], key, 16);
            reserved_note(heap, i);
            *index = heap->count + i;
            cpu.ptr = heap->cpu_start + (SIZE_T)*index * heap->stride;
            ID3D12Device_CreateShaderResourceView(device, res, &srv, cpu);
            if (heap->reserved_res || (heap->reserved_res = calloc(heap->reserved, sizeof(*heap->reserved_res))))
            {
                heap->reserved_res[i] = res;
                ref_change(NULL, res);
            }
            found = TRUE;
        }
    }
    ReleaseSRWLockExclusive(&lock);
    return found;
}

/* res is going: its views in reserved tails (shadow_reserve_srv) become null ones and aren't found
 * again, so a resource made later at its address doesn't get them. */
void shadow_forget_reserved(ID3D12Resource *res)
{
    ID3D12Device *device = rebox_shared_device();

    if (!res || !device)
        return;
    AcquireSRWLockExclusive(&lock);
    for (struct shadow_heap *h = heaps; h; h = h->next)
        for (UINT i = 0; h->reserved_res && i < h->reserved_used; i++)
            if (h->reserved_res[i] == res)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC srv = { DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_SRV_DIMENSION_TEXTURE2D,
                                                        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING };
                D3D12_CPU_DESCRIPTOR_HANDLE cpu = { h->cpu_start + (SIZE_T)(h->count + i) * h->stride };
                srv.Texture2D.MipLevels = 1;
                ID3D12Device_CreateShaderResourceView(device, NULL, &srv, cpu);
                memset(h->reserved_words[i], 0xff, 16);
                ref_change(res, NULL);
                h->reserved_res[i] = NULL;
            }
    ReleaseSRWLockExclusive(&lock);
}

/* A view of a 1x1 zeroed dummy texture (SRV, or UAV for a storage image) in the heap's reserved
 * tail, made once: what a console shader gets for a descriptor the title left empty or wrote
 * outside our heaps (a UE5 title's T# with base 0x9200, a dummy), which reads zero there as a null
 * descriptor does. A real texture, not a null view: lavapipe crashed on a null storage image. */
static ID3D12Resource *dummy_texture(ID3D12Device *device)
{
    static ID3D12Resource *texture;
    D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC desc = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 1, 1, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                                 { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };

    if (!texture)
        ID3D12Device_CreateCommittedResource(device, &heap, D3D12_HEAP_FLAG_NONE, &desc,
                                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL, &IID_ID3D12Resource,
                                             (void **)&texture);
    return texture;
}
BOOL shadow_reserve_null(SIZE_T heap_cpu, BOOL storage, UINT *index)
{
    ID3D12Device *device = rebox_shared_device();
    struct shadow_heap *heap = NULL;
    const UINT32 key[4] = { 0x4c4c554e /* NULL */, storage, 0, 0 };
    BOOL found = FALSE;

    if (reserved_lookup(heap_cpu, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, key, index))
        return TRUE;
    AcquireSRWLockExclusive(&lock);
    for (struct shadow_heap *h = heaps; h && !heap; h = h->next)
        if (h->cpu_start == heap_cpu && h->type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && h->reserved)
            heap = h;
    if (heap)
    {
        UINT i = reserved_find(heap, key);
        if (i != ~0u)
        {
            *index = heap->count + i;
            found = TRUE;
        }
    }
    if (heap && !found && device && dummy_texture(device) && reserve_room(heap)
            && (heap->reserved_words || (heap->reserved_words = calloc(heap->reserved, 16))))
    {
        ID3D12Resource *texture = dummy_texture(device);
        D3D12_CPU_DESCRIPTOR_HANDLE cpu;
        UINT i = heap->reserved_used++;

        memcpy(heap->reserved_words[i], key, 16);
        reserved_note(heap, i);
        *index = heap->count + i;
        cpu.ptr = heap->cpu_start + (SIZE_T)*index * heap->stride;
        if (storage)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav = { DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_UAV_DIMENSION_TEXTURE2D };
            ID3D12Device_CreateUnorderedAccessView(device, texture, NULL, &uav, cpu);
        }
        else
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv = { DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_SRV_DIMENSION_TEXTURE2D,
                                                    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING };
            srv.Texture2D.MipLevels = 1;
            ID3D12Device_CreateShaderResourceView(device, texture, &srv, cpu);
        }
        found = TRUE;
    }
    ReleaseSRWLockExclusive(&lock);
    return found;
}

BOOL shadow_reserve_sampler(SIZE_T heap_cpu, const UINT32 words[4], UINT *index)
{
    ID3D12Device *device = rebox_shared_device();
    struct shadow_heap *heap = NULL;
    BOOL found = FALSE;

    if (reserved_lookup(heap_cpu, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, words, index))
        return TRUE;
    AcquireSRWLockExclusive(&lock);
    for (struct shadow_heap *h = heaps; h && !heap; h = h->next)
        if (h->cpu_start == heap_cpu && h->type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER && h->reserved)
            heap = h;
    if (heap)
    {
        UINT i = reserved_find(heap, words);
        if (i != ~0u)
        {
            *index = heap->count + i;
            found = TRUE;
        }
    }
    if (heap && !found && device && reserve_room(heap)
            && (heap->reserved_words || (heap->reserved_words = calloc(heap->reserved, 16))))
    {
        D3D12_SAMPLER_DESC desc;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu;
        UINT i = heap->reserved_used++;

        console_decode_sampler(words, &desc);
        memcpy(heap->reserved_words[i], words, 16);
        reserved_note(heap, i);
        *index = heap->count + i;
        cpu.ptr = heap->cpu_start + (SIZE_T)*index * heap->stride;
        ID3D12Device_CreateSampler(device, &desc, cpu);
        found = TRUE;
    }
    ReleaseSRWLockExclusive(&lock);
    return found;
}

/* ------------------------------------------------------------- exports */

/* The shadow of the heap whose descriptor 0 has vkd3d-proton CPU handle
 * cpu_start: its title address (0 if not shader-visible), contents, stride
 * and size. */
HRESULT WINAPI ReboxShadowHeap(SIZE_T cpu_start, UINT64 *title_va, void **data, UINT *stride, UINT *count)
{
    HRESULT hr = E_INVALIDARG;

    cpu_start = shadow_cpu_to_inner(cpu_start); /* a heap placed in title memory: its title start */
    AcquireSRWLockShared(&lock);
    for (struct shadow_heap *h = heaps; h; h = h->next)
        if (h->cpu_start == cpu_start)
        {
            if (title_va) *title_va = h->title_va;
            if (data) *data = h->data;
            if (stride) *stride = h->stride;
            if (count) *count = h->count;
            hr = S_OK;
            break;
        }
    ReleaseSRWLockShared(&lock);
    return hr;
}

/* Copies size bytes at a title address out of the shadow heaps. Fails unless
 * the whole range lies in one heap. */
BOOL WINAPI ReboxShadowRead(UINT64 address, void *out, SIZE_T size)
{
    BOOL ok = FALSE;

    AcquireSRWLockShared(&lock);
    for (struct shadow_heap *h = heaps; h; h = h->next)
    {
        UINT64 end = h->title_va + (UINT64)h->count * h->stride;
        if (h->title_va && address >= h->title_va && address < end && size <= end - address)
        {
            memcpy(out, h->data + (address - h->title_va), size);
            ok = TRUE;
            break;
        }
    }
    ReleaseSRWLockShared(&lock);
    return ok;
}

/* Finds the descriptor whose console words are these (4 or 8 dwords; sampler
 * for S#) and returns vkd3d-proton's CPU handle for it, preferring
 * shader-visible heaps. Several slots holding the same words hold the same
 * view, so any of them will do. */
BOOL WINAPI ReboxShadowFind(const UINT32 *words, UINT word_count, BOOL sampler, SIZE_T *cpu)
{
    struct entry *best = NULL;
    UINT32 key[8];

    if (!words || !word_count)
        return FALSE;
    make_key(words, word_count, sampler, key);
    AcquireSRWLockShared(&lock);
    for (struct entry *e = buckets[0][hash_key(key)]; e; e = e->next)
        if (!memcmp(e->key, key, sizeof(key)) && (sampler == (e->heap->type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER))
                && (!best || (e->heap->shader_visible && !best->heap->shader_visible)))
            best = e;
    if (best && cpu) /* the title's handle for a heap in its memory, as the API gives it */
        *cpu = (best->heap->title_cpu ? best->heap->title_cpu : best->heap->cpu_start)
                + (SIZE_T)best->index * best->heap->stride;
    ReleaseSRWLockShared(&lock);
    return best != NULL;
}

/* For tests: vkd3d-proton's CPU handle for a handle as the title has it (the heap_cpu a console compile
 * is given). */
SIZE_T WINAPI ReboxShadowInnerCpu(SIZE_T cpu)
{
    return shadow_cpu_to_inner(cpu);
}

/* For tests: the index of a slot of the heap at cpu (a handle as the title has it) holding these
 * words, preferring one whose view is in views (shadow_find_index_view); ~0u for none. */
UINT WINAPI ReboxShadowFindIndex(SIZE_T cpu, const UINT32 *words, UINT word_count, UINT views)
{
    UINT index;
    return shadow_find_index_view(shadow_cpu_to_inner(cpu), words, word_count, FALSE, FALSE, views, &index) ? index : ~0u;
}

/* For tests: the slot of an SRV for the texture UAV in slot index of the heap at cpu, as a sampled
 * binding of it gets (shadow_reserve_srv_for_uav); ~0u for none. */
UINT WINAPI ReboxShadowSrvForUav(SIZE_T cpu, UINT index)
{
    UINT srv;
    return shadow_reserve_srv_for_uav(shadow_cpu_to_inner(cpu), index, &srv) ? srv : ~0u;
}
