#pragma once

#include <windows.h>
#include <d3d12.h>

/* Console-format (RDNA2) descriptors for D3D12 views; see console_desc.c.
 * Resource descriptors fill 8 dwords (T# or V# plus zeros), samplers 4. */
void console_encode_srv(ID3D12Resource *res, const D3D12_SHADER_RESOURCE_VIEW_DESC *desc, UINT32 out[8]);
void console_encode_uav(ID3D12Resource *res, const D3D12_UNORDERED_ACCESS_VIEW_DESC *desc, UINT32 out[8]);
void console_encode_cbv(const D3D12_CONSTANT_BUFFER_VIEW_DESC *desc, UINT32 out[8]);
void console_encode_sampler(const D3D12_SAMPLER_DESC *desc, UINT32 out[4]);
/* The D3D12 sampler an S# describes (console_encode_sampler's inverse, for samplers shaders build
 * themselves). */
void console_decode_sampler(const UINT32 words[4], D3D12_SAMPLER_DESC *desc);
/* The texture a T# names by its synthetic address (console_encode_srv/uav), or NULL. */
ID3D12Resource *console_texture_of(const UINT32 words[2]);
/* console_texture_of for the T#s of views made through the API only (by their texture's id, not a
 * title address): no lookups in title memory, whose locks an upload pass holds. */
ID3D12Resource *console_texture_by_id(const UINT32 words[2]);
/* res is being released: its synthetic texture address is forgotten. */
void console_texture_forget(ID3D12Resource *res);
