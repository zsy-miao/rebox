#pragma once

#include <windows.h>
#include <d3d12.h>

/* REBOX_DEBUG_VIEW: one of the title's render targets or depth buffers shown in place of the
 * presented plane; see debug_view.c. */

/* A command list's binding, as debug_view_bind left it (in struct proxy; zeroed is "none"). */
struct debug_view_list
{
    int rt, depth;               /* 1 + the bound targets' table indices, or 0 */
    void *rt_res, *depth_res;    /* what those entries held when bound */
    void *snapshot_res;          /* the chosen target, bound: copied when the pass ends */
    int probe;                   /* 1 + the probe slot of the draw being recorded, or 0 */
};

BOOL debug_view_on(void);

/* From the proxy's command lists (proxy.c). list is the inner list (a copy is recorded into it when
 * the pass that drew the chosen target ends). display: a display-sized target is among rts. */
void debug_view_bind(ID3D12GraphicsCommandList *list, struct debug_view_list *l, ID3D12Resource *const *rts,
                     UINT count, ID3D12Resource *depth, BOOL display);
/* Before a draw, mesh dispatch or ExecuteIndirect (args and count: an ExecuteIndirect's buffers, or
 * NULL): in a probed frame, queries begin around it if it draws into the target shown. */
void debug_view_draw_begin(ID3D12GraphicsCommandList *list, struct debug_view_list *l, ID3D12Resource *args,
                           UINT64 args_offset, ID3D12Resource *count, UINT64 count_offset);
/* After it: fate is what became of it (DRAW_FATE_*). */
void debug_view_draw(ID3D12GraphicsCommandList *list, struct debug_view_list *l, BOOL mesh, UINT fate);
void debug_view_close(ID3D12GraphicsCommandList *list, struct debug_view_list *l);

/* From PresentX (present.c): the target to show this present, referenced; FALSE for none. *snapshot:
 * res is our copy, in COMMON; else it is the title's, in whatever state its barriers left it. */
struct debug_view_pick
{
    ID3D12Resource *res;
    D3D12_RESOURCE_DESC desc;  /* of the title's target */
    BOOL depth, snapshot;
};
BOOL debug_view_pick(struct debug_view_pick *pick);
/* Records the view of pick->res (readable as a shader resource by now) into all of dst (RGBA8, in
 * UNORDERED_ACCESS), dst_w x dst_h. */
BOOL debug_view_record(ID3D12GraphicsCommandList *list, const struct debug_view_pick *pick, ID3D12Resource *dst,
                       UINT dst_w, UINT dst_h);
