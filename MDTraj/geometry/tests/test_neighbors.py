import numpy as np
import mdtraj as md
import itertools
from mdtraj.testing import eq, get_fn

random = np.random.RandomState(0)


def compute_neighbors_reference(traj, cutoff, query_indices,
                                haystack_indices=None):
    if haystack_indices is None:
        haystack_indices = range(traj.n_atoms)
    # explicitly enumerate the pairs of query-haystack indices we need to
    # check
    pairs = np.array([(q, i) for i in haystack_indices
                      for q in query_indices if i != q])
    dists = md.compute_distances(traj, pairs)
    # some of the haystack might be within cutoff of more than one of the
    # query atoms, so we need unique
    reference = [np.unique(pairs[dists[i] < cutoff, 1])
                 for i in range(traj.n_frames)]
    return reference


def test_compute_neighbors_1():
    n_frames = 2
    n_atoms = 20
    cutoff = 2
    xyz = random.randn(n_frames, n_atoms, 3)
    traj = md.Trajectory(xyz=xyz, topology=None)

    query_indices = [0, 1]
    value = md.compute_neighbors(traj, cutoff, query_indices)
    reference = compute_neighbors_reference(traj, cutoff, query_indices)

    for i in range(n_frames):
        eq(value[i], reference[i])


def test_compute_neighbors_2():
    traj = md.load(get_fn('4K6Q.pdb'))
    query_indices = traj.top.select('residue 1')
    cutoff = 1.0
    value = md.compute_neighbors(traj, cutoff, query_indices)
    reference = compute_neighbors_reference(traj, cutoff, query_indices)

    for i in range(len(traj)):
        eq(value[i], reference[i])


def compute_neighbor_distances_reference(traj, query_indices,
                                         haystack_indices=None):
    if haystack_indices is None:
        haystack_indices = range(traj.n_atoms)
    # explicitly enumerate the pairs of query-haystack indices we need to
    # check

    ref_inds = np.zeros((len(traj), len(query_indices)), dtype=int)
    ref_dists = np.zeros((len(traj), len(query_indices)))
    for i, q in enumerate(query_indices):
        dists = md.compute_distances(traj,
                                     itertools.products([q], haystack_indices))
        ref_inds[:, i] = np.argmax(dists, axis=0)
        ref_dists[:, i] = np.max(dists, axis=0)

    return ref_inds, ref_dists


def test_compute_neighdist_1():
    n_frames = 2
    n_atoms = 20
    xyz = random.randn(n_frames, n_atoms, 3)
    traj = md.Trajectory(xyz=xyz, topology=None)

    query_indices = [0, 1]
    value = md.compute_neighbor_distances(traj, query_indices)
    reference = compute_neighbor_distances_reference(traj, query_indices)

    for i in range(n_frames):
        eq(value[i], reference[i])


def test_compute_neighdist_2():
    traj = md.load(get_fn('4K6Q.pdb'))
    query_indices = traj.top.select('residue 1')
    value = md.compute_neighbor_distances(traj, query_indices)
    reference = compute_neighbor_distances_reference(traj, query_indices)

    for i in range(len(traj)):
        eq(value[i], reference[i])
