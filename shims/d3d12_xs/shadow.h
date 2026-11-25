#pragma once

#include <windows.h>
#include <d3d12.h>

/* The kind of view each shadow slot holds, so a binding is never pointed at a vkd3d-proton
 * descriptor of another type (mutable sets don't check). */
enum shadow_view
{
    SHADOW_VIEW_NONE, SHADOW_VIEW_SRV_TEXTURE, SHADOW_VIEW_SRV_TYPED, SHADOW_VIEW_SRV_RAW, SHADOW_VIEW_UAV_TEXTURE,
    SHADOW_VIEW_UAV_TYPED, SHADOW_VIEW_UAV_RAW, SHADOW_VIEW_CBV, SHADOW_VIEW_SAMPLER,
};

/* Console-format shadow heaps; see shadow.c. */
/* Shader-visible CBV/SRV/UAV and sampler heaps get slots past the title's count in vkd3d-proton's
 * heap (patch 0005 raises its sampler-heap limit to match), for descriptors the host makes itself:
 * samplers that shaders build from literals (SHADOW_RESERVED), views of textures the title wrote T#s
 * for itself (SHADOW_RESERVED_VIEWS: a world brings thousands, and with 256 the rest bound nothing). */
#define SHADOW_RESERVED 256
#define SHADOW_RESERVED_VIEWS 65536
void shadow_heap_created(ID3D12DescriptorHeap *inner, const D3D12_DESCRIPTOR_HEAP_DESC *desc, UINT increment,
        UINT reserved);
UINT shadow_reserved(SIZE_T cpu_start);
/* A slot in the reserved tail of the sampler heap at heap_cpu holding the sampler these S# words
 * describe, made on first use (one per distinct S#). */
BOOL shadow_reserve_sampler(SIZE_T heap_cpu, const UINT32 words[4], UINT *index);
/* A slot in the reserved tail of the CBV/SRV/UAV heap at heap_cpu holding an SRV of the texture UAV
 * in slot uav_index (same resource, format, mip and slices), made on first use. */
BOOL shadow_reserve_srv_for_uav(SIZE_T heap_cpu, UINT uav_index, UINT *index);
/* A null texture SRV (or UAV, storage) in the reserved tail, made once per heap. */
BOOL shadow_reserve_null(SIZE_T heap_cpu, BOOL storage, UINT *index);
void shadow_heap_destroyed(ID3D12DescriptorHeap *inner);
/* The heap's CPU memory from the title's allocation hooks (XMemAlloc, object type 4, allocator 128),
 * for titles that have them; then its CPU handles are the title's (shadow.c). */
void shadow_place_heap(ID3D12DescriptorHeap *inner, UINT count, UINT increment);
/* The title's CPU start for the heap at vkd3d-proton's cpu_start, or 0. */
SIZE_T shadow_title_cpu_start(SIZE_T cpu_start);
/* A title CPU handle in a placed heap as vkd3d-proton's; any other handle unchanged. */
SIZE_T shadow_cpu_to_inner(SIZE_T cpu);
BOOL shadow_title_heaps_placed(void);
/* usable: shadow_buffer_range_usable of the view's range (only that much is noted as a buffer). */
void shadow_write_cbv(SIZE_T cpu, const D3D12_CONSTANT_BUFFER_VIEW_DESC *desc, UINT64 usable);
/* How many bytes from va of [va, va + size) the GPU can read (title memory made GPU memory in place
 * if it can be; below 4 GiB and in the GPU memory window, only known buffers and Rebox's chunks, up
 * to their end); 0 if it can't read va. */
UINT64 shadow_buffer_range_usable(UINT64 va, UINT64 size);
void shadow_write_srv(SIZE_T cpu, ID3D12Resource *res, const D3D12_SHADER_RESOURCE_VIEW_DESC *desc);
void shadow_write_uav(SIZE_T cpu, ID3D12Resource *res, const D3D12_UNORDERED_ACCESS_VIEW_DESC *desc);
void shadow_write_sampler(SIZE_T cpu, const D3D12_SAMPLER_DESC *desc);
void shadow_copy(SIZE_T dst, SIZE_T src, UINT count);
void shadow_copy_ranges(UINT dst_count, const D3D12_CPU_DESCRIPTOR_HANDLE *dst_starts, const UINT *dst_sizes,
        UINT src_count, const D3D12_CPU_DESCRIPTOR_HANDLE *src_starts, const UINT *src_sizes);

/* Shader-visible shadows, for the recompiler's guest memory and the BDA page table. */
struct shadow_region { UINT64 title_va, gpu_va; const void *data; SIZE_T size; };
UINT shadow_regions(struct shadow_region *out, UINT max);
/* Changes whenever what shadow_regions returns may have (a heap made, freed or placed). */
LONG shadow_regions_generation(void);
/* Before a submission: refreshes the GPU mirrors of heaps placed in title memory (shadow.c). */
void shadow_sync_mirrors(void);
/* The slot of the heap starting at heap_cpu that holds these console words (4 or 8); with loose,
 * also allowing the fields AnyPS5 rewrites in bound descriptors. */
BOOL shadow_find_index(SIZE_T heap_cpu, const UINT32 *words, UINT n, BOOL sampler, BOOL loose, UINT *index);

/* What heap slot 10 reports (shadow.c). */
enum shadow_heap_va { SHADOW_HEAP_VA_TITLE, SHADOW_HEAP_VA_REAL, SHADOW_HEAP_VA_VIRTUAL };
enum shadow_heap_va shadow_heap_va_mode(void);
/* The title address and byte size of the shader-visible heap whose descriptor 0 is at cpu_start. */
BOOL shadow_title(SIZE_T cpu_start, UINT64 *title_va, UINT64 *size);
enum shadow_view shadow_slot_view(SIZE_T heap_cpu, UINT index);
/* shadow_find_index, preferring a slot whose view is one of views (a mask of 1 << enum shadow_view;
 * 0: the first slot found): the same words can be in slots of different views. */
BOOL shadow_find_index_view(SIZE_T heap_cpu, const UINT32 *words, UINT n, BOOL sampler, BOOL loose, UINT views,
                            UINT *index);
/* GPU address ranges [begin, end) of buffers the title made views of or asked the GPU address of,
 * sorted, merged where they overlap or touch; returns the count. */
UINT shadow_buffer_ranges(UINT64 (*out)[2], UINT max);
/* Notes a buffer's range [va, va + size) (proxy.c's GetGPUVirtualAddress). */
void shadow_note_buffer_range(UINT64 va, UINT64 size);
/* The noted range holding address, if any. */
BOOL shadow_buffer_range_at(UINT64 address, UINT64 *begin, UINT64 *end);
/* Changes whenever a buffer range is added or grows. */
LONG shadow_buffer_ranges_generation(void);

/* A shader resource view of res in the heap's reserved tail (made once per resource), for a T# the
 * title wrote itself; its index in *index. */
BOOL shadow_reserve_srv(SIZE_T heap_cpu, ID3D12Resource *res, UINT *index);

/* Views counted per resource (shadow.c): shadow_track starts counting the descriptor slots holding a
 * view of res; shadow_untrack_unreferenced stops, if none does (TRUE), else returns FALSE;
 * shadow_view_refs gives the count (-1 untracked). shadow_forget_reserved makes res's views in the
 * reserved tails null ones. */
void shadow_track(ID3D12Resource *res);
BOOL shadow_untrack_unreferenced(ID3D12Resource *res);
LONG shadow_view_refs(ID3D12Resource *res);
void shadow_forget_reserved(ID3D12Resource *res);
