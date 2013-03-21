# -*- coding: utf-8 -*-

"""
Contains functions for reading frame-by-frame flow data and trajectories in 
various formats.
"""
import os, os.path
from ConfigParser import SafeConfigParser
import re

import numpy as np
from scipy import io

from .particle import Particle
from .trajectory import Trajectory

def collect_particles(fname_tmpl, frame, path_seg=False):
    """
    Going backwards over trajAcc files [2], starting from a given frame,
    collect the data for all particles whose path begins in earlier frames and
    go as far as the given frame.
    
    Arguments:
    fname_tmpl - a format-string with one %d where the frame number should be
        inserted.
    frame - the frame number.
    path_seg - if True, find for each particle also the particle matching it in
        the next time step, so that acceleration can be calculated. Discarts 
        unmatched particles.
    
    Returns:
    a table with columns 0-5,33 from the files, combined from all lines in all
    files that belong to particles in frame ``frame``. If path_seg is True, the
    table has two layers (a 2,n,7 array), the first is the particles in the 
    given frame, the second is their matches in the next time step.
    """
    selected = []
    cur_frame = frame
    fname_tmpl = os.path.expanduser(fname_tmpl)
    
    while os.path.exists(fname_tmpl % cur_frame):
        table = np.loadtxt(fname_tmpl % cur_frame, usecols=(0,1,2,3,4,5,33))
        path_age = frame - cur_frame
        
        if path_seg is True:
            segs = np.nonzero((table[:,-1] == path_age) & \
                (np.roll(table[:,-1], -1) == path_age + 1))[0]
            in_frame = np.concatenate(
                (table[segs,:][None,...], table[segs + 1,:][None,...]), axis=0)
        else:
            in_frame = table[table[:,-1] == path_age]
        
        # When no previous path is long enough to reach ``frame``:
        if in_frame.shape[0] == 0:
            break
        
        selected.append(in_frame)
        cur_frame -= 1
    
    if path_seg is True:
        all_rows = np.concatenate(selected, axis=1)
        return all_rows[:,mark_unique_rows(all_rows[0])]
    else:
        all_rows = np.vstack(selected)
        return all_rows[mark_unique_rows(all_rows)]

def mark_unique_rows(all_rows):
    """
    Filter out rows whose position columns represent a particle that already
    appears, so that each particle position appears only once.
    
    Arguments:
    all_rows - an array with n rows and at least 3 columns for position.
    
    Returns:
    an array with the indices of rows to take from the input such that in the
    result, the first 3 columns form a unique combination.
    """
    # Remove duplicates (particles occupying same position):
    srt = np.lexsort(all_rows[:,:3].T)
    diff = np.diff(all_rows[srt,:3], axis=0).any(axis=1)
    uniq = np.r_[srt[0], srt[1:][diff]]
    uniq.sort()
        
    return uniq

def trajectories_mat(fname):
    data = io.loadmat(os.path.expanduser(fname))
    # Get the workspace variable holding the trajectories:
    data_name = [s for s in data.keys() \
        if (not s.startswith('__')) and (not s == 'directory')][0]
    raw = data[data_name][:,0]
    
    trajects = []
    for traj in raw:
        # also convert data from mm to m.
        pos = np.hstack((traj['xf'], traj['yf'], traj['zf']))/1000.
        vel = np.hstack((traj['uf'], traj['vf'], traj['wf']))/1000.
        t = traj['t'].squeeze()
        trajid = traj['trajid'][0,0]
        trajects.append(Trajectory(pos, vel, t, trajid))
    
    return trajects

def trajectories_acc(fname, first=None, last=None):
    """
    Extract all trajectories in a directory of trajAcc files.
    
    Arguments:
    fname - a template file name representing all trajAcc files in the
        directory, with exactly one '%d' representing the frame number.
    first, last - inclusive range of frames to read, rel. filename numbering.
    
    Returns:
    a list of Trajectory objects.
    """
    trajects = []
    dirname, basename = os.path.split(os.path.expanduser(fname))
    is_data_file = re.compile(basename.replace('%d', '(\d+)', 1))
    
    for fname in os.listdir(dirname):
        match = is_data_file.match(fname)
        if match is None: continue
        frame = int(match.group(1))
        
        if first is not None and frame < first: continue
        if last is not None and frame >= last: break
        
        table = np.loadtxt(os.path.join(dirname, fname),
            usecols=(0,1,2,3,4,5,33))
        traj_starts = np.nonzero(table[:,-1] == 0)[0]
        traj_ends = np.r_[traj_starts[1:], table.shape[0]]
        
        for s, e in zip(traj_starts, traj_ends):
            trajects.append(Trajectory(
                table[s:e,0:3], table[s:e,3:6], table[s:e,6] + frame,
                len(trajects) ))
    
    return trajects

def trajectories_ptvis(fname, first=None, last=None, frate=1., xuap=False):
    """
    Extract all trajectories in a directory of ptv_is files, as generated by
    programs in the 3d-ptv/pyptv family.
    
    Arguments:
    fname - a template file name representing all trajAcc files in the
        directory, with exactly one '%d' representing the frame number.
    first, last - inclusive range of frames to read, rel. filename numbering.
    frate - frame rate, used for calculating velocities by backward 
        derivative.
    xuap - The format is extended with colums for velocity and acceleration.
    
    Returns:
    a list of Trajectory objects.
    """
    fname = os.path.expanduser(fname)
    dirname, basename = os.path.split(fname)
    is_data_file = re.compile(basename.replace('%d', '(\d+)', 1))
    
    # Collect existing frames. This is necessary to ensure that frames are
    # processed in the correct order, which in this format is important.
    frame_nums = []
    for name in os.listdir(dirname):
        match = is_data_file.match(name)
        if match is None: continue
        frame = int(match.group(1))
        
        if first is not None and frame < first: continue
        if last is not None and frame > last: break
        # Note that we're reading one extra frame, otherwise the last frame
        # has 0 path segments.
        
        frame_nums.append(frame)
    
    # Process frames in order.
    frame_nums.sort()
    
    if xuap:
        fmt = np.dtype([('prev', 'i4'), ('next', 'i4'), ('pos', '3f8'),
                        ('pos_int', '3f8'), ('vel', '3f8'), ('acc', '3f8')])
        skip = 0
        count_base = 1
    else:
        fmt = np.dtype([('prev', 'i4'), ('next', 'i4'), ('pos', '3f8')])
        skip = 1
        count_base = 0
    
    # In the first frame, every particle starts a trajectory.
    table = np.loadtxt(fname % frame_nums[0], dtype=fmt, skiprows=skip)
    
    frames = []
    
    pos = table['pos']
    if not xuap: pos /=1000.
    
    if 'vel' in fmt.fields:
        vel = table['vel']
    else:
        vel = np.zeros_like(pos)
    
    frame = np.hstack((pos, vel, np.ones((table.shape[0], 1))*frame_nums[0],
        np.arange(table.shape[0])[:,None]))
    max_traj = table.shape[0]
    frames.append(frame)
    
    # Assign trajectory numbers:
    for fix, frame_num in enumerate(frame_nums[1:]):
        table = np.loadtxt(fname % frame_num, dtype=fmt, skiprows=skip)
        
        # Continue existing trajectories into this frame:
        cont = table['prev'] - count_base > -1
        traj = np.empty(table['prev'].shape)
        prev_ix = table['prev'][cont] - count_base
        traj[cont] = frames[fix][:,-1][prev_ix]
        
        # Start new trajectories:
        num_new_traj = np.sum(~cont)
        traj[~cont] = np.arange(max_traj, max_traj + num_new_traj)
        max_traj += num_new_traj
        
        # Consolidate into frame table.
        pos = table['pos']
        if not xuap: pos /= 1000.
        t = np.ones((table.shape[0], 1))*frame_num
        
        if 'vel' in fmt.fields:
            vel = table['vel']
        else:
            vel = np.zeros_like(pos)
        
        frame = np.hstack((pos, vel, t, traj[:,None]))
        if 'vel' not in fmt.fields:
            frames[fix][prev_ix,3:6] = \
                (pos[cont] - frames[fix][prev_ix,:3]) * frate
        frames.append(frame)
    
    # From time series to list of trajectories:
    trajects = [[] for tr in xrange(max_traj)]
    for frame in frames:
        for tix in np.unique(frame[:,-1]):
            trajects[int(tix)].append(frame[frame[:,-1] == tix][0])
    
    trajects = [np.array(traj) for traj in trajects]
    return [Trajectory(traj[:,:3], traj[:,3:6], traj[:,6], traj[0,7]) \
        for traj in trajects]
    
def trajectories(fname, first, last, frate, fmt=None):
    """
    Extract all trajectories in a given target location. The location format
    is interpreted based on the format of the data files, in the respective 
    trajectories_* functions.
    
    Trajectories of one frame are filtered out.
    
    Arguments:
    fname - a template file name, as needed by the appropriate suboridinate
        function.
    first, last - inclusive range of frames to read, rel. filename numbering.
    frate - frame rate under which the film was shot - needed for ptvis 
        trajectories.
    
    Returns:
    a list of Trajectory objects.
    """
    # Infer format:
    if fmt is None:
        fmt = infer_format(fname)
    
    if fmt == 'mat':
        traj = trajectories_mat(fname)
    elif fmt == 'acc':
        traj = trajectories_acc(fname, first, last)
    elif fmt == 'ptvis':
        traj = trajectories_ptvis(fname, first, last, frate)
    elif fmt == 'xuap':
        traj = trajectories_ptvis(fname, first, last, frate, xuap=True)
    
    return [tr for tr in traj if len(tr) > 1]
        
def infer_format(fname):
    """
    Try to guess the format of a particles data file by its name.
    
    Arguments:
    fname - the file name from which to guess the format.
    
    Returns:
    A string marking the format. Currently one of 'acc', 'mat' or 'ptvis'.
    """
    if fname.endswith('mat'):
        return 'mat'
    elif 'ptv_is' in fname:
        return 'ptvis'
    elif 'xuap' in fname:
        return 'xuap'
    else:
        return 'acc'

def collect_particles_mat(fname, frame, path_seg=False):
    """
    The same as collect_particles, but uses mat files as generated by the PTV
    post-processing code.
    """
    trajects = trajectories_mat(fname)
    return collect_particles_generic(trajects, frame, path_seg)
    
def collect_particles_generic(trajects, frame, path_seg=False):
    """
    Collect from a list of trajectories the particles appearing in a given
    frame.
    
    Arguments:
    trajects - a list of Trajectory objects.
    frame - the frame number.
    path_seg - if True, find for each particle also the particle matching it in
        the next time step, so that acceleration can be calculated. Discarts 
        unmatched particles.
    
    Returns:
    a table with columns 0-2 for position, 3-5 for velocity, 6 for frame
    number and 7 for trajectory id. If path_seg is True, the table has two
    layers (a 2,n,7 array), the first is the particles in the given frame,
    the second is their matches in the next time step.
    """
    selected = []
    for traj in trajects:
        if path_seg is True:
            t = np.nonzero((traj.time() == frame) & \
                (np.roll(traj.time(), -1) == frame + 1))[0]
            if len(t) == 0: continue
            
            t = t[0]
            sel = traj[t : t + 2].reshape(2, 1, -1)
            
        else:
            t = np.nonzero(traj.time() == frame)[0]
            if len(t) == 0: continue
            sel = traj[t[0]]
        
        selected.append(sel)
    
    if len(selected) == 0:
        return np.empty((2,0,7))
    
    if path_seg is True:
        all_rows = np.concatenate(selected, axis=1)
        return all_rows[:,mark_unique_rows(all_rows[0])]
    else:
        all_rows = np.vstack(selected)
        return all_rows[mark_unique_rows(all_rows)]


def read_frame_data(conf_fname):
    """
    Read a configuration file in INI format, which specifies the locations 
    where particle positions and velocities should be read from, and directly
    stores some scalar frame values, like particle densidy etc.
    
    Arguments:
    conf_fname - name of the config file
    
    Returns:
    particle - a Particle object holding particle properties.
    frate - the frame rate at which the scene was shot.
    part_segs - particle segents, a (2,n,7) array with the properties of each
        particle at the first and last frame of a path segment.
    tracer_segs - same as part_segs but for tracer files.
    """
    parser = SafeConfigParser()
    parser.read(conf_fname)
    
    particle = Particle(
        parser.getfloat("Particle", "diameter"),
        parser.getfloat("Particle", "density"))
    
    data = [None]*2
    titles = ["part_file", "tracer_file"]
    frame = parser.getint("Scene", "frame")
    frate = parser.getfloat("Scene", "frame rate")
    
    for dix in xrange(2):
        fname = parser.get("Scene", titles[dix])
        traj = trajectories(fname, frame, frame + 1, frate, None)
        data[dix] = collect_particles_generic(traj, frame, True)
    
    return particle, frate, data[0], data[1]