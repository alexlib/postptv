"""
HDF5 / pytables reader benchmarks (SPEEDUP_PLAN Phase 0 / Phase 2).

Run with::

    uv run pytest benchmarks/ --benchmark-save=<phase>

Reads the real ``data/tracers/ptv_is.*`` corpus, then exercises the HDF5
readers both as text-ingest and as batched ``Scene`` iteration.
"""

import tempfile
import numpy as np
import pytest
from flowtracks import io
from flowtracks.scene import Scene

PTVIS = 'data/tracers/ptv_is.%d'
FIRST, LAST = 10001, 10115


@pytest.fixture(scope='module')
def trajectories():
    return list(io.iter_trajectories_ptvis(PTVIS, first=FIRST, last=LAST))


@pytest.fixture(scope='module')
def h5_path(trajectories, tmp_path_factory):
    d = tmp_path_factory.mktemp('bench')
    fname = str(d / 'trajs.h5')
    io.save_particles_table(fname, trajectories)
    return fname


def test_iter_trajectories_ptvis(benchmark):
    benchmark(lambda: list(
        io.iter_trajectories_ptvis(PTVIS, first=FIRST, last=LAST)))


def test_trajectories_table(benchmark, h5_path):
    benchmark(lambda: io.trajectories_table(h5_path))


def test_scene_iter_trajectories(benchmark, h5_path):
    scene = Scene(h5_path)
    benchmark(lambda: list(scene.iter_trajectories()))


def test_scene_iter_frames(benchmark, h5_path):
    scene = Scene(h5_path)
    benchmark(lambda: list(scene.iter_frames()))
