"""
Trajectory stitching routines for reconnecting broken trajectory segments.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from flowtracks.trajectory import Trajectory


def stitch_trajectories(
    trajs,
    fps=1.0,
    max_gap=5,
    max_distance=5.0,
    max_vel_diff=10.0,
    dt=None,
):
    """Stitch (relink) broken trajectory segments across short frame gaps.

    Parameters
    ----------
    trajs : list of Trajectory
        List of trajectory objects to be stitched.
    fps : float, optional
        Frames per second (or sampling frequency). Default is 1.0.
    max_gap : int, optional
        Maximum number of missing frames allowed between track end and start.
        e.g., max_gap=5 means a gap of 1 to 5 missing frames can be bridged.
    max_distance : float, optional
        Maximum spatial distance between the extrapolated end of Track A and
        start of Track B.
    max_vel_diff : float, optional
        Maximum magnitude of velocity difference between Track A end and Track B start.
    dt : float, optional
        Time step delta between consecutive frames. If None, derived as 1.0 / fps.

    Returns
    -------
    stitched_trajs : list of Trajectory
        List of stitched Trajectory objects with gaps interpolated.
    """
    if dt is None:
        dt = 1.0 / fps

    if not trajs:
        return []

    active_trajs = list(trajs)

    while True:
        N = len(active_trajs)
        if N < 2:
            break

        ends = []
        starts = []

        for tr in active_trajs:
            t_tr = tr.time()
            pos_tr = tr.pos()
            vel_tr = tr.velocity()

            ends.append({
                't': t_tr[-1],
                'pos': pos_tr[-1],
                'vel': vel_tr[-1],
            })

            starts.append({
                't': t_tr[0],
                'pos': pos_tr[0],
                'vel': vel_tr[0],
            })

        candidates = []
        for i in range(N):
            t_end = ends[i]['t']
            p_end = ends[i]['pos']
            v_end = ends[i]['vel']

            for j in range(N):
                if i == j:
                    continue

                t_start = starts[j]['t']
                p_start = starts[j]['pos']
                v_start = starts[j]['vel']

                gap_frames = int(round(t_start - t_end - 1))
                if not (0 <= gap_frames <= max_gap):
                    continue

                delta_t = (gap_frames + 1) * dt

                p_extrap_fwd = p_end + v_end * delta_t
                p_extrap_bwd = p_start - v_start * delta_t

                dist_fwd = np.linalg.norm(p_extrap_fwd - p_start)
                dist_bwd = np.linalg.norm(p_extrap_bwd - p_end)
                dist = 0.5 * (dist_fwd + dist_bwd)

                if dist > max_distance:
                    continue

                vel_diff = np.linalg.norm(v_end - v_start)
                if vel_diff > max_vel_diff:
                    continue

                cost = dist + 0.1 * vel_diff
                candidates.append((i, j, cost))

        if not candidates:
            break

        BIG_COST = 1e9
        cost_matrix = np.full((N, N), BIG_COST, dtype=np.float64)

        for i, j, c in candidates:
            cost_matrix[i, j] = c

        row_ind, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        for r, c in zip(row_ind, col_indices):
            cost_val = cost_matrix[r, c]
            if cost_val < BIG_COST / 2:
                matches.append((r, c))

        if not matches:
            break

        merged_mask = np.zeros(N, dtype=bool)
        new_active_trajs = []

        for i, j in matches:
            merged_mask[i] = True
            merged_mask[j] = True

            trA = active_trajs[i]
            trB = active_trajs[j]

            tA = trA.time()
            tB = trB.time()

            t_end = tA[-1]
            t_start = tB[0]
            gap_count = int(round(t_start - t_end - 1))

            if gap_count > 0:
                gap_times = np.arange(t_end + 1, t_start)
                p_end = trA.pos()[-1]
                p_start = trB.pos()[0]
                v_end = trA.velocity()[-1]
                v_start = trB.velocity()[0]

                s = np.linspace(0, 1, gap_count + 2)[1:-1]
                s2 = s**2
                s3 = s**3

                h00 = 2 * s3 - 3 * s2 + 1
                h10 = s3 - 2 * s2 + s
                h01 = -2 * s3 + 3 * s2
                h11 = s3 - s2

                gap_dt = (t_start - t_end) * dt
                gap_pos = (
                    h00[:, None] * p_end
                    + h10[:, None] * (v_end * gap_dt)
                    + h01[:, None] * p_start
                    + h11[:, None] * (v_start * gap_dt)
                )

                dh00 = 6 * s2 - 6 * s
                dh10 = 3 * s2 - 4 * s + 1
                dh01 = -6 * s2 + 6 * s
                dh11 = 3 * s2 - 2 * s

                gap_vel = (
                    dh00[:, None] * p_end
                    + dh10[:, None] * (v_end * gap_dt)
                    + dh01[:, None] * p_start
                    + dh11[:, None] * (v_start * gap_dt)
                ) / gap_dt

                merged_pos = np.concatenate([trA.pos(), gap_pos, trB.pos()], axis=0)
                merged_vel = np.concatenate([trA.velocity(), gap_vel, trB.velocity()], axis=0)
                merged_time = np.concatenate([tA, gap_times, tB], axis=0)
            else:
                merged_pos = np.concatenate([trA.pos(), trB.pos()], axis=0)
                merged_vel = np.concatenate([trA.velocity(), trB.velocity()], axis=0)
                merged_time = np.concatenate([tA, tB], axis=0)

            merged_traj = Trajectory(merged_pos, merged_vel, merged_time, trA.trajid())

            for k in trA.as_dict():
                if k not in ('pos', 'velocity', 'time', 'trajid') and trB.has_property(k):
                    propA = trA.as_dict()[k]
                    propB = trB.as_dict()[k]
                    if gap_count > 0:
                        prop_end = propA[-1]
                        prop_start = propB[0]
                        gap_prop = np.array([prop_end + (prop_start - prop_end) * si for si in s])
                        merged_prop = np.concatenate([propA, gap_prop, propB], axis=0)
                    else:
                        merged_prop = np.concatenate([propA, propB], axis=0)
                    merged_traj.create_property(k, merged_prop)

            new_active_trajs.append(merged_traj)

        for idx in range(N):
            if not merged_mask[idx]:
                new_active_trajs.append(active_trajs[idx])

        active_trajs = new_active_trajs

    return active_trajs
