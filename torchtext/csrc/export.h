#pragma once

// Define the visibility of symbols.
// The original logic and background can be found here.
// https://github.com/pytorch/pytorch/blob/bcc02769bef1d7b89bec724223284958b7c5b564/c10/macros/Export.h#L49-L55
//
// In the context of torchtext, the logic is simpler at the moment.
//
// The torchtext custom operations are implemented in
// `torchtext/lib/libtorchtext.[so|pyd]`. Some symbols are referred from
// `torchtext._torchtext`.
//
// In Windows, default visibility of dynamically library are hidden, while in
// Linux/macOS, they are visible.
//
// At the moment we do not expect torchtext libraries to be built/linked
// statically. We assume they are always shared.

#ifdef _WIN32
#define TORCHTEXT_EXPORT __declspec(dllexport)
#define TORCHTEXT_IMPORT __declspec(dllimport)
#else // _WIN32
#if defined(__GNUC__)
#define TORCHTEXT_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define TORCHTEXT_EXPORT
#endif // defined(__GNUC__)
#define TORCHTEXT_IMPORT TORCHTEXT_EXPORT
#endif // _WIN32

#ifdef TORCHTEXT_BUILD_MAIN_LIB
#define TORCHTEXT_API TORCHTEXT_EXPORT
#else
#define TORCHTEXT_API TORCHTEXT_IMPORT
#endif
