# Third-party notices

The PulseX code is MIT (see LICENSE). It is built with the following, each under its own license.

## Included in this repository (as binaries in `bin/`)

- **llama.cpp / ggml** - MIT License, the ggml authors. `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`,
  `ggml-hexagon.dll`, `llama.dll` and the NPU modules `libggml-htp-v7x.so` are built from the PulseX fork of
  llama.cpp with a Hexagon NPU backend. The license text is in `bin/LICENSE-llama.cpp-ggml.txt`.
- **Dear ImGui** - MIT License, Omar Cornut and contributors. Compiled into `pulsex-image.exe`. The license text is in
  `bin/LICENSE-imgui.txt`.

## Not included

- **FLUX.2 [klein] 9B** (Black Forest Labs) - the model this program runs. Its weights, and everything made from
  them, are under the FLUX Non-Commercial License and are not in this repository.
- **Qwen3 8B** (Alibaba Cloud) - the text encoder of FLUX.2 [klein]. Not in this repository.
- **Qualcomm AI runtime (QAIRT)** - `flux-make.exe` is compiled against the QNN headers of Qualcomm's QAIRT SDK and
  loads `QnnHtp.dll` and `QnnSystem.dll` at run time. Neither the headers nor the runtime are in this repository;
  both are under Qualcomm's own license.
- **QuickSRNet-Large** (Qualcomm AI Hub; BSD 3-Clause License) - the upscaler for enlarging. Not in this repository;
  the compiled graph is with the model files on Hugging Face, with its license text.

The table of Swedish words (`bin/flux_cues.tsv`) is PulseX's own, generated from `cues.json` of pulsex-zimage (MIT).

No warranty. Nothing you type or generate leaves the computer: the program makes no network connection.
