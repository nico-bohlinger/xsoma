"""Forward kinematics of the G1 in the compiled scene: a free root (the pelvis) and at most one hinge per body, anchored at
the body origin (checked at construction).

Holosoma reads body poses after a forward-kinematics refresh of the post-step state (`refresh_rigid_body_states`), so the
task sees the bodies of the final substep's qpos; MuJoCo's xpos after a step is one substep older. This is that refresh, used
by the simulator and by imagination alike. Only the bodies on the paths from the root to the requested ones are evaluated.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import mujoco
import numpy as np

from .rot import axis_angle_to_mat, mm3, mv3, quat_to_mat


class Kinematics:
    """Forward kinematics for a fixed list of bodies: `forward` takes the root pose and the joint angles and returns the world
    poses of exactly those bodies."""

    def __init__(self, m: mujoco.MjModel, bodies: Sequence[str]):
        assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[0] == 1, "joint 0 must be the root's free joint"
        self.parent = [int(b) for b in m.body_parentid]
        self.pos = np.asarray(m.body_pos, np.float32)
        self.rot = np.asarray(quat_to_mat(np.asarray(m.body_quat, np.float32)), np.float32)
        self.axis = [None] * m.nbody
        self.col = [-1] * m.nbody  # the hinge's column in q = qpos[7:]
        for j in range(1, m.njnt):
            b = int(m.jnt_bodyid[j])
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and self.col[b] < 0, "one hinge per body"
            assert np.allclose(m.jnt_pos[j], 0), "hinges must be anchored at the body origin"
            self.axis[b], self.col[b] = np.asarray(m.jnt_axis[j], np.float32), int(m.jnt_qposadr[j]) - 7
        self.out = [m.body(n).id for n in bodies]
        need = set()
        for b in self.out:
            while b > 1:
                need.add(b)
                b = self.parent[b]
        self.order = sorted(need)  # MuJoCo numbers parents before their children

    def forward(self, p, R, q):
        """World positions (..., n, 3) and rotations (..., n, 3, 3) of the requested bodies, given the root pose (p, R)."""
        P, Rs = {1: p}, {1: R}  # by body id: body 1 is the pelvis (the free root), body 0 the world
        for b in self.order:
            par = self.parent[b]
            P[b] = P[par] + mv3(Rs[par], self.pos[b])
            Rb = mm3(Rs[par], self.rot[b])
            Rs[b] = Rb if self.col[b] < 0 else mm3(Rb, axis_angle_to_mat(self.axis[b], q[..., self.col[b]]))
        return jnp.stack([P[b] for b in self.out], -2), jnp.stack([Rs[b] for b in self.out], -3)
