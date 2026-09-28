# Third-party materials

The MIT license applies to new code and documentation in this repository. The VLCoT manuscript, reported tables, original figures, pretrained weights, benchmark data, and simulator assets are not relicensed here and are not bundled.

The implementation uses the public interfaces of:

- [OpenVLA](https://github.com/openvla/openvla) and the [OpenVLA-7B model](https://huggingface.co/openvla/openvla-7b): the model adapter loads the checkpoint's Python implementation at a pinned revision. Model/base-model terms remain applicable, including upstream Llama-related requirements where applicable.
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft), MIT: continuous action chunking and L1 regression motivate the action interface. This repository uses its own attention-based action head and the stock causal OpenVLA language backbone, not OFT's globally bidirectional Transformers fork.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), MIT: imported separately for simulator replay and evaluation.
- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), MIT: imported separately at the stable 2.0 revision. Assets and dependencies may have separate terms.
- [LIBERO-RECOVER](https://github.com/liulin815/libero-recover): referenced protocol, not redistributed implementation or data. The similarly named custom-scene robustness repository is not silently substituted for its four-level benchmark.

Dependencies are installed separately and retain their notices. Consult their repositories and dataset/model cards before redistribution. No blanket license is asserted over the paper's research materials.
