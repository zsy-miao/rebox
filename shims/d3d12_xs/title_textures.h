#pragma once

#include <windows.h>
#include <d3d12.h>

/* Textures placed at title addresses, uploaded from title memory; see title_textures.c. */
void title_texture_register(ID3D12Resource *resource, UINT64 address, const D3D12_RESOURCE_DESC *console_desc,
                            const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES state);
void title_textures_upload(ID3D12CommandQueue *queue);
UINT64 title_texture_size(const D3D12_RESOURCE_DESC *desc);
/* Whether a texture is filled from title memory: its address and how often it was uploaded. */
/* A GPU copy writes the texture (recorded): from then on its texels are the GPU's, and title memory
 * is no longer uploaded over them. */
void title_texture_gpu_written(ID3D12Resource *resource);
BOOL title_texture_info(ID3D12Resource *resource, UINT64 *address, UINT *uploads);
/* The texture registered at this title address, or NULL. */
ID3D12Resource *title_texture_at(UINT64 address);
BOOL title_texture_at_nowait(UINT64 address, ID3D12Resource **res);
void title_texture_unregister(ID3D12Resource *resource);
/* Whether it can be released and made again from title memory alone; *filled: uploaded yet. */
BOOL title_texture_refillable(ID3D12Resource *resource, BOOL *filled);
void WINAPI ReboxTitleMemoryWritten(UINT64 address, UINT64 size);
/* Whether textures in physical-page views are write-tracked (REBOX_TEXTURE_TRACKING, default on). */
BOOL title_textures_tracking(void);

/* For the memory log: textures registered (dirty ones), and uploads in flight (buffers, bytes). */
struct title_texture_stats { UINT textures, dirty, uploads; UINT64 upload_bytes; };
void title_textures_stats(struct title_texture_stats *s);
/* Whether a texture's title memory (GPU address, console size) can't be read yet. */
BOOL title_texture_memory_unreadable(UINT64 address, UINT64 size);
