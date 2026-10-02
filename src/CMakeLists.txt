# 2026-10-02: FLUX.2 klein 9B on the Hexagon NPU (see flux_stream.cpp). Built from the ggml-hexagon tree like
# zimage_dit and ltx_video.
set(TARGET flux-stream)
add_executable(${TARGET} flux_stream.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE _USE_MATH_DEFINES)
if(WIN32)
    # ANSI code page = UTF-8 (model or work folders with letters outside ASCII), merged into the linker's own manifest
    target_link_options(${TARGET} PRIVATE "LINKER:/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/utf8.manifest")
endif()

# 2026-10-02: the text encoder (Qwen3 8B, taps after layers 9/18/27, padded to 512) - see flux_encode.cpp
add_executable(flux-encode flux_encode.cpp)
target_link_libraries(flux-encode PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(flux-encode PRIVATE cxx_std_17)
if(WIN32)
    target_link_options(flux-encode PRIVATE "LINKER:/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/utf8.manifest")
endif()

# 2026-10-02: the VAE decoder in stages (see flux_vae.cpp)
add_executable(flux-vae flux_vae.cpp)
target_link_libraries(flux-vae PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(flux-vae PRIVATE cxx_std_17)
if(WIN32)
    target_link_options(flux-vae PRIVATE "LINKER:/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/utf8.manifest")
endif()

# 2026-10-02: a QNN context binary (the VAE decoder / encoder as ONNX Runtime compiled them) run with QnnHtp.dll alone
# - see flux_qnn.cpp. Needs only the QNN headers; QnnHtp.dll and QnnSystem.dll are loaded at run time.
set(FLUX_QNN_INCLUDE "" CACHE PATH "QNN headers: <QAIRT SDK>/include/QNN (needed for flux-make and flux-qnn)")
if(EXISTS "${FLUX_QNN_INCLUDE}/QnnInterface.h")
    add_executable(flux-qnn flux_qnn.cpp)
    add_executable(flux-make flux_make.cpp)
    target_include_directories(flux-make PRIVATE ${FLUX_QNN_INCLUDE})
    target_compile_features(flux-make PRIVATE cxx_std_17)
    target_link_libraries(flux-make PRIVATE windowscodecs ole32 gdiplus)
    target_include_directories(flux-qnn PRIVATE ${FLUX_QNN_INCLUDE})
    target_compile_features(flux-qnn PRIVATE cxx_std_17)
    if(WIN32)
        target_link_options(flux-qnn PRIVATE "LINKER:/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/utf8.manifest")
        target_link_options(flux-make PRIVATE "LINKER:/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/utf8.manifest")
    endif()
endif()

# 2026-10-02: the window (flux_gui.cpp) - Dear ImGui + DX11, runs flux-make.exe. Nothing of ggml is linked.
set(FLUX_IMGUI_DIR "" CACHE PATH "Dear ImGui root (needed for the window, pulsex-image)")
if(WIN32 AND EXISTS "${FLUX_IMGUI_DIR}/imgui.cpp")
    add_executable(pulsex-image
        flux_gui.cpp
        ${FLUX_IMGUI_DIR}/imgui.cpp
        ${FLUX_IMGUI_DIR}/imgui_draw.cpp
        ${FLUX_IMGUI_DIR}/imgui_tables.cpp
        ${FLUX_IMGUI_DIR}/imgui_widgets.cpp
        ${FLUX_IMGUI_DIR}/backends/imgui_impl_win32.cpp
        ${FLUX_IMGUI_DIR}/backends/imgui_impl_dx11.cpp
    )
    target_include_directories(pulsex-image PRIVATE ${FLUX_IMGUI_DIR} ${FLUX_IMGUI_DIR}/backends)
    target_compile_features(pulsex-image PRIVATE cxx_std_17)
    target_compile_definitions(pulsex-image PRIVATE NOMINMAX)
    target_link_libraries(pulsex-image PRIVATE d3d11 dxgi d3dcompiler comdlg32 windowscodecs ole32 shell32 dwmapi)
    target_link_options(pulsex-image PRIVATE "LINKER:/SUBSYSTEM:WINDOWS" "LINKER:/ENTRY:mainCRTStartup"
                        "LINKER:/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/gui.manifest")
endif()
