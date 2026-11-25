#pragma once

#include <windows.h>
#include <d3d12.h>

/* D3D12.X proxies: objects with the Xbox vtable layouts that forward to
 * vkd3d-proton. See docs/d3d12x.md for the slot tables. */

/* Returns the device proxy for the shared device, as riid. */
HRESULT proxy_query_device(ID3D12Device *inner, REFIID riid, void **out);
/* The SDK version from D3D12XBOX_CREATE_DEVICE_PARAMETERS: picks the queue layout. */
void proxy_set_sdk_version(UINT32 version);

/* For work d3d12_xs records into a title's command list: the wrapped list,
 * plus the descriptor heaps the title last set on it (up to 2), so they can
 * be restored. A list that isn't a proxy is returned as is, with no heaps. */
ID3D12GraphicsCommandList *proxy_list(void *list, ID3D12DescriptorHeap **heaps, UINT *heap_count);

/* REBOX_SLOT_STATS: called once per PresentX; logs per-slot call counts
 * every N frames. Slots are marked c (our code), T (trap), or neither
 * (forwarded to the PC slot). */
void proxy_frame_presented(void);
void proxy_log_slot_stats(void);
/* Logs, and clears, the counts of root views read as empty by pipeline and kind (with the census). */
void proxy_root_views_census_log(void);

/* The QueryPerformanceCounter time before which every submission (ExecuteCommandLists through the
 * proxies) has finished on the GPU, as last seen (every 20 ms); 0 before the first. Memory the GPU
 * may read is reused or freed only once this passes the time it was last handed to the GPU. */
LONG64 proxy_gpu_finished_before(void);

/* Proxy -> vkd3d-proton object; anything else is returned unchanged. */
void *proxy_unwrap(void *object);

/* The texture the title knows at this address (a T# it wrote itself holds it), or NULL. */
ID3D12Resource *proxy_texture_at(UINT64 address);

/* Before a texture is uploaded into on queue: one made without memory (REBOX_VIEW_TEXTURE_DEFER)
 * gets it, mapped on that queue. FALSE if it has none and can't have it. */
BOOL proxy_texture_memory(ID3D12CommandQueue *queue, ID3D12Resource *res);
