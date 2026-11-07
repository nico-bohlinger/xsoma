# Third-party notices

## Holosoma

XSoma reimplements the G1 velocity-tracking task of Holosoma (https://github.com/amazon-far/holosoma, commit `79ba714`) and
redistributes one of its files:

- `assets/g1_29dof.xml` is Holosoma's G1 MJCF (`src/holosoma/holosoma/data/robots/g1/g1_29dof.xml`), unmodified. Its header
  records where it comes from: Unitree's G1 description (see below), with actuators added by Juyue Chen and the groups of some
  visual geoms changed by Younggyo Seo.
- `xsoma/task.py`, `xsoma/sim.py`, `xsoma/scene.py` and `xsoma/terrain.py` reimplement, in JAX and with MuJoCo Warp, the
  task's observations, actions, commands, gait clock, rewards, terminations, resets, randomization, curriculum and scene and
  its `terrain_locomotion_mix` generator (`docs/holosoma_task.md` cites the source of every part). Changes: rewritten as pure
  JAX functions for batched simulation and for differentiation in imagination; the terrain is one heightfield per tile, not
  stretched; the explicit sole contact pairs are equivalent priority contacts; friction randomization optionally reaches the
  soles (each listed in the files' docstrings and the README).
- `xsoma/evaluate.py` runs the actor networks of Holosoma's PPO and FastSAC, `tools/holosoma_ckpt.py` reads their checkpoints,
  and `tools/holosoma_dump.py` runs Holosoma's environment to record `tests/data/holosoma_g1_ppo_plane.*`.

These parts are used under the Apache License, Version 2.0 ([`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)).
Holosoma's NOTICE file reads:

```
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
```

## Unitree G1 description

The G1 model in `assets/g1_29dof.xml` derives from Unitree's G1 description. `tools/download_meshes.sh` downloads its meshes
(from Holosoma's repository at the same commit) for rendering; training needs none, and they are not included here.
Unitree's description is distributed under the BSD 3-Clause License:

```
BSD 3-Clause License

Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## XMimic, BlinkMimic, InstantTrack and supermimic_vibed

The world model, policy network, replay buffer, optimizers and rotation utilities (`xsoma/nets.py`, `xsoma/buffer.py`,
`xsoma/optim.py`, `xsoma/rot.py`) and the structure of the trainer (`xsoma/train.py`) come from XMimic
(https://github.com/nico-bohlinger/xmimic, MIT, same author as XSoma), whose code is in large part a port of BlinkMimic
(https://github.com/fuyichen0725/blinkmimic), which derives from InstantTrack (https://github.com/Huy-Quang-Dao/instantTrack)
and supermimic_vibed (https://github.com/zita-ch/supermimic_vibed). Those portions are used under their MIT Licenses:

```
MIT License

Copyright (c) 2026 Fuyi Chen
Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
MIT License

Copyright (c) 2026 Huy-Quang-Dao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dependencies

JAX, MuJoCo, MJX, MuJoCo Warp and NVIDIA Warp (Apache-2.0), NumPy, Matplotlib and Pillow are installed from their own
distributions under their own licenses and are not redistributed here. Holosoma itself (for the baselines and
`tools/holosoma_dump.py`) is installed from its own repository.
