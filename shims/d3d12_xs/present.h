#pragma once

#include <windows.h>
#include <d3d12.h>

/* PresentX: shows the presented plane in a window through a DXGI swapchain
 * on the queue, and optionally dumps frames (REBOX_DUMP_FRAMES). */

struct present_queue;

HRESULT present_x(ID3D12CommandQueue *queue, struct present_queue **state, UINT planes, const void *plane_params,
                  const void *params);
void present_queue_destroy(struct present_queue *state);

/* Resource state tracking, so the copy out of the plane starts from the
 * state the title left it in. Fed from resource creation and from the
 * barriers the title records. */
void present_track_state(void *resource, D3D12_RESOURCE_STATES state);
BOOL present_tracked_state(void *resource, D3D12_RESOURCE_STATES *state);
/* Either kind: *kind 1 for a legacy state, 2 for an enhanced-barrier layout, in *value. */
BOOL present_tracked(void *resource, UINT *kind, UINT *value);
void present_track_barriers(UINT count, const D3D12_RESOURCE_BARRIER *barriers);
void present_track_enhanced(UINT32 count, const D3D12_BARRIER_GROUP *groups);
