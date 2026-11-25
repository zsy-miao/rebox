#pragma once

#include <windows.h>
#include <d3d12.h>

/* The one device the process uses. Console titles have a single device, and
 * GPU memory handed out by xmem must live on the device the title renders
 * with, since vkd3d-proton resolves GPU VAs per device. NULL if creation
 * failed. Not reference-counted for the caller. */
ID3D12Device *rebox_shared_device(void);

/* GPU-visible memory, suballocated from CPU-visible D3D12 heaps on the shared
 * device. With VK_LAYER_REBOX_gpuva_identity the CPU pointer and the GPU
 * VA are equal; both are returned so callers can tell. */
HRESULT WINAPI ReboxGpuAlloc(SIZE_T size, SIZE_T alignment, void **cpu, UINT64 *gpu);
void WINAPI ReboxGpuFree(void *cpu);

/* If address lies in memory from ReboxGpuAlloc, returns its chunk's range. */
BOOL WINAPI ReboxGpuQuery(const void *address, void **base, SIZE_T *size);

/* Bytes handed out by ReboxGpuAlloc and not yet freed. */
UINT64 WINAPI ReboxGpuBytesInUse(void);

/* Every chunk's GPU VA, CPU pointer and size, for the shader recompiler's guest memory
 * (console_compile.c). Returns the count, at most max. */
struct gpu_memory_region { UINT64 gpu; const void *cpu; SIZE_T size; };
UINT gpu_memory_regions(struct gpu_memory_region *out, UINT max);
/* Changes whenever a chunk is made or freed (what gpu_memory_regions and
 * gpu_memory_identity_extent see). */
LONG gpu_memory_generation(void);
/* The chunk heap holding a GPU address, and the address's offset in it (the heap gets a
 * reference); FALSE if the address isn't in GPU memory from ReboxGpuAlloc. */
BOOL gpu_memory_heap_at(UINT64 address, ID3D12Heap **heap, UINT64 *offset);
/* Makes the title's own memory [address, address + size) GPU-visible at the same address (widened
 * to 64 KiB pages), keeping its contents; TRUE if it is GPU memory now (or already was). Needs
 * VK_LAYER_REBOX_gpuva_identity's mailbox (REBOX_GPUVA_MAILBOX=1). */
BOOL gpu_memory_adopt(UINT64 address, UINT64 size);
/* The same for a size that is a guess (a root view's): as much of it from address on as can be,
 * cut at the committed run's end and at GPU memory above; TRUE if some of it, from address on, is
 * GPU memory now. */
BOOL gpu_memory_adopt_some(UINT64 address, UINT64 size);
/* For vkd3d-proton (patch 0033): whether the buffer heap under 2 MiB it is making is to be memory
 * of its own rather than a piece of a chunk (the heap of title memory being made GPU memory). */
BOOL WINAPI ReboxHeapWhole(UINT64 size);
/* Logs, and clears, the counts of requests not made GPU memory by reason (with the census). */
void gpu_memory_adopt_failures_log(void);
/* If address is identity-mapped GPU memory (CPU address == GPU VA), the end of its chunk. */
BOOL gpu_memory_identity_extent(UINT64 address, UINT64 *end);
/* The end of the identity window's highest allocation, or 0 (no mailbox, or none yet). */
UINT64 gpu_memory_window_end(void);
/* Whether [gpu, gpu + size) is inside one of the chunks, by GPU VA. */
BOOL gpu_memory_holds_gpu(UINT64 gpu, UINT64 size);
/* If gpu is inside one of the chunks (by GPU VA), the end of that chunk. */
BOOL gpu_memory_gpu_extent(UINT64 gpu, UINT64 *end);
/* Bytes of title memory made GPU memory in place (gpu_memory_adopt). */
UINT64 gpu_memory_adopted_bytes(void);
/* The same through chunks that touch: the end of the identity-mapped GPU memory that runs on
 * unbroken from address. */
BOOL gpu_memory_identity_run_end(UINT64 address, UINT64 *end);
