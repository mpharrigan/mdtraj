# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2017 Stanford University and the Authors
#
# Authors: Matthew Harrigan
# Contributors:

# MDTraj is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MDTraj. If not, see <http://www.gnu.org/licenses/>.

import os
from copy import deepcopy
from collections import Iterable
import numpy as np
import functools

from ..formats import (TOPOLOGY_LOADERS,
                       TRAJECTORY_FILEOBJECTS,
                       TRAJECTORY_LOADERS)

from .topology import Topology
from .residue_names import _SOLVENT_TYPES
from ..utils import (ensure_type, in_units_of,
                     lengths_and_angles_to_box_vectors,
                     box_vectors_to_lengths_and_angles, cast_indices,
                     deprecated)
from ..utils.six import PY3, string_types

from mdtraj import _rmsd
from mdtraj import find_closest_contact
from mdtraj.geometry import distance
from mdtraj.geometry import _geometry


def _assert_files_exist(filenames):
    """Throw an IO error if files don't exist

    Parameters
    ----------
    filenames : {str, [str]}
        String or list of strings to check
    """
    if isinstance(filenames, string_types):
        filenames = [filenames]
    for fn in filenames:
        if not (os.path.exists(fn) and os.path.isfile(fn)):
            raise IOError('No such file: %s' % fn)


def _assert_files_or_dirs_exist(names):
    """Throw an IO error if files don't exist

    Parameters
    ----------
    filenames : {str, [str]}
        String or list of strings to check
    """
    if isinstance(names, string_types):
        names = [names]
    for fn in names:
        if not (os.path.exists(fn) and \
                        (os.path.isfile(fn) or os.path.isdir(fn))):
            raise IOError('No such file: %s' % fn)


def _get_extension(filename):
    (base, extension) = os.path.splitext(filename)
    if extension == '.gz':
        extension2 = os.path.splitext(base)[1]
        return extension2 + extension
    return extension


if PY3:
    def _hash_numpy_array(x):
        hash_value = hash(x.shape)
        hash_value ^= hash(x.strides)
        hash_value ^= hash(x.data.tobytes())
        return hash_value
else:
    def _hash_numpy_array(x):
        writeable = x.flags.writeable
        try:
            x.flags.writeable = False
            hash_value = hash(x.shape)
            hash_value ^= hash(x.strides)
            hash_value ^= hash(x.data)
        finally:
            x.flags.writeable = writeable
        return hash_value


def open(filename, mode='r', force_overwrite=True, **kwargs):
    """Open an mdtraj file-like object

    This function returns an instance of an open file-like
    object capable of reading/writing the trajectory (depending on
    'mode'). It does not actually load the trajectory from disk or
    write anything.

    Parameters
    ----------
    filename : str
        Path to the trajectory file on disk
    mode : {'r', 'w'}
        The mode in which to open the file, either 'r' for read or 'w' for
        write.
    force_overwrite : bool
        If opened in write mode, and a file by the name of `filename` already
        exists on disk, should we overwrite it?

    Other Parameters
    ----------------
    kwargs : dict
        Other keyword parameters are passed directly to the file object

    Returns
    -------
    fileobject : object
        Open trajectory file, whose type is determined by the filename
        extension

    See Also
    --------
    load, ArcTrajectoryFile, BINPOSTrajectoryFile, DCDTrajectoryFile,
    HDF5TrajectoryFile, LH5TrajectoryFile, MDCRDTrajectoryFile,
    NetCDFTrajectoryFile, PDBTrajectoryFile, TRRTrajectoryFile,
    XTCTrajectoryFile, TNGTrajectoryFile

    """
    ext = _get_extension(filename)
    try:
        loader = TRAJECTORY_FILEOBJECTS[ext]
    except KeyError:
        raise IOError('Sorry, no loader for extension "{}" was found. I can only open files with extensions: {}'
                      .format(ext, list(TRAJECTORY_FILEOBJECTS.keys())))
    return loader(filename, mode=mode, force_overwrite=force_overwrite, **kwargs)


class Coordinates:
    """A container for molecular dynamics coordinates.

    Stores a number of fields describing the system through time,
    including the cartesian coordinates of each atoms (`xyz`), and information about the
    unitcell if appropriate (`unitcell_vectors`, `unitcell_length`,
    `unitcell_angles`).

    `Coordinates` uses the native unit system of the file

    See Also
    --------
    md.Trajectory : class
        A complete trajectory in a consistent unit system and with a topology.
    """

    def __init__(self, xyz, time, unitcell_lengths=None, unitcell_angles=None, unitcell_vectors=None):
        self.xyz = xyz
        self.time = time
        self.unitcell_lengths = unitcell_lengths
        self.unitcell_angles = unitcell_angles

        # _rmsd_traces are the inner product of each centered conformation,
        # which are required for computing RMSD. Normally these values are
        # calculated on the fly in the cython code (rmsd/_rmsd.pyx), but
        # optionally, we enable the use precomputed values which can speed
        # up the calculation (useful for clustering), but potentially be unsafe
        # if self._xyz is modified without a corresponding change to
        # self._rmsd_traces. This array is populated computed by
        # center_conformations, and no other methods should really touch it.
        self._rmsd_traces = None


    @property
    def n_frames(self):
        """Number of frames in the trajectory

        Returns
        -------
        n_frames : int
            The number of frames in the trajectory
        """
        return self.xyz.shape[0]

    @property
    def n_atoms(self):
        """Number of atoms in the trajectory

        Returns
        -------
        n_atoms : int
            The number of atoms in the trajectory
        """
        return self.xyz.shape[1]

    @property
    def timesteps(self):
        if self.n_frames <= 1:
            raise ValueError("Cannot calculate timesteps if trajectory has one frame.")
        return self.time[1:] - self.time[:-1]

    @property
    def timestep(self):
        """Timestep between frames, in native units

        Returns
        -------
        timestep : float
            The timestep between frames, in native units.
        """
        timestep = np.unique(self.timesteps)
        if len(timestep) != 1:
            raise ValueError("Found {} unique timestep values.".format(len(timestep)))
        return timestep[0]

    @property
    def unitcell_vectors(self):
        """The vectors that define the shape of the unit cell in each frame

        Returns
        -------
        vectors : np.ndarray, shape(n_frames, 3, 3)
            Vectors defining the shape of the unit cell in each frame.
            The semantics of this array are that the shape of the unit cell
            in frame ``i`` are given by the three vectors, ``value[i, 0, :]``,
            ``value[i, 1, :]``, and ``value[i, 2, :]``.
        """
        if self._unitcell_lengths is None or self._unitcell_angles is None:
            return None

        v1, v2, v3 = lengths_and_angles_to_box_vectors(
            self._unitcell_lengths[:, 0],  # a
            self._unitcell_lengths[:, 1],  # b
            self._unitcell_lengths[:, 2],  # c
            self._unitcell_angles[:, 0],  # alpha
            self._unitcell_angles[:, 1],  # beta
            self._unitcell_angles[:, 2],  # gamma
        )
        return np.swapaxes(np.dstack((v1, v2, v3)), 1, 2)

    @unitcell_vectors.setter
    def unitcell_vectors(self, vectors):
        """Set the three vectors that define the shape of the unit cell

        Parameters
        ----------
        vectors : tuple of three arrays, each of shape=(n_frames, 3)
            The semantics of this array are that the shape of the unit cell
            in frame ``i`` are given by the three vectors, ``value[i, 0, :]``,
            ``value[i, 1, :]``, and ``value[i, 2, :]``.
        """
        if vectors is None or np.all(np.abs(vectors) < 1e-15):
            self._unitcell_lengths = None
            self._unitcell_angles = None
            return

        if not len(vectors) == len(self):
            raise TypeError('unitcell_vectors must be the same length as '
                            'the trajectory. you provided %s' % str(vectors))

        v1 = vectors[:, 0, :]
        v2 = vectors[:, 1, :]
        v3 = vectors[:, 2, :]
        func = box_vectors_to_lengths_and_angles
        a, b, c, alpha, beta, gamma = func(v1, v2, v3)

        self._unitcell_lengths = np.vstack((a, b, c)).T
        self._unitcell_angles = np.vstack((alpha, beta, gamma)).T

    @property
    def unitcell_volumes(self):
        """Volumes of unit cell for each frame.

        Returns
        -------
        volumes : {np.ndarray, shape=(n_frames), None}
            Volumes of the unit cell in each frame, in nanometers^3, or None
            if the Trajectory contains no unitcell information.
        """
        if self.unitcell_lengths is not None:
            return np.array(list(map(np.linalg.det, self.unitcell_vectors)))
        else:
            return None

    @property
    def unitcell_lengths(self):
        """Lengths that define the shape of the unit cell in each frame.

        Returns
        -------
        lengths : {np.ndarray, shape=(n_frames, 3), None}
            Lengths of the unit cell in each frame, in nanometers, or None
            if the Trajectory contains no unitcell information.
        """
        return self._unitcell_lengths

    @property
    def unitcell_angles(self):
        """Angles that define the shape of the unit cell in each frame.

        Returns
        -------
        lengths : np.ndarray, shape=(n_frames, 3)
            The angles between the three unitcell vectors in each frame,
            ``alpha``, ``beta``, and ``gamma``. ``alpha' gives the angle
            between vectors ``b`` and ``c``, ``beta`` gives the angle between
            vectors ``c`` and ``a``, and ``gamma`` gives the angle between
            vectors ``a`` and ``b``. The angles are in degrees.
        """
        return self._unitcell_angles

    @unitcell_lengths.setter
    def unitcell_lengths(self, value):
        """Set the lengths that define the shape of the unit cell in each frame

        Parameters
        ----------
        value : np.ndarray, shape=(n_frames, 3)
            The distances ``a``, ``b``, and ``c`` that define the shape of the
            unit cell in each frame, or None
        """
        self._unitcell_lengths = ensure_type(value, np.float32, 2,
                                             'unitcell_lengths',
                                             can_be_none=True,
                                             shape=(len(self), 3),
                                             warn_on_cast=False,
                                             add_newaxis_on_deficient_ndim=True)

    @unitcell_angles.setter
    def unitcell_angles(self, value):
        """Set the lengths that define the shape of the unit cell in each frame

        Parameters
        ----------
        value : np.ndarray, shape=(n_frames, 3)
            The angles ``alpha``, ``beta`` and ``gamma`` that define the
            shape of the unit cell in each frame. The angles should be in
            degrees.
        """
        self._unitcell_angles = ensure_type(value, np.float32, 2,
                                            'unitcell_angles', can_be_none=True,
                                            shape=(len(self), 3),
                                            warn_on_cast=False,
                                            add_newaxis_on_deficient_ndim=True)

    def _string_summary_basic(self):
        """Basic summary of traj in string form."""
        value = ("mdtraj.Coordinates with {} frames, {} atoms, {}"
                 .format(self.n_frames, self.n_atoms,
                         'and unitcells' if self._have_unitcell else 'without unitcells'))
        return value

    def __len__(self):
        return self.n_frames

    def __add__(self, other):
        return self.join(other)

    def __str__(self):
        return "<{}>".format(self._string_summary_basic())

    def __repr__(self):
        return "<{} at 0x{:02x}>".format(self._string_summary_basic(), id(self))

    def __hash__(self):
        hash_value = hash(self.top)
        # combine with hashes of arrays
        hash_value ^= _hash_numpy_array(self.xyz)
        hash_value ^= _hash_numpy_array(self.time)
        hash_value ^= _hash_numpy_array(self.unitcell_lengths)
        hash_value ^= _hash_numpy_array(self.unitcell_angles)
        return hash_value

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def superpose(self, reference, frame=0, atom_indices=None,
                  ref_atom_indices=None, parallel=True):
        """Superpose each conformation in this trajectory upon a reference

        Parameters
        ----------
        reference : md.Trajectory
            Align self to a particular frame in `reference`
        frame : int
            The index of the conformation in `reference` to align to.
        atom_indices : array_like, or None
            The indices of the atoms to superpose. If not
            supplied, all atoms will be used.
        ref_atom_indices : array_like, or None
            Use these atoms on the reference structure. If not supplied,
            the same atom indices will be used for this trajectory and the
            reference one.
        parallel : bool
            Use OpenMP to run the superposition in parallel over multiple cores

        Returns
        -------
        self
        """

        if atom_indices is None:
            atom_indices = slice(None)

        if ref_atom_indices is None:
            ref_atom_indices = atom_indices

        if (not isinstance(ref_atom_indices, slice)
            and (len(ref_atom_indices) != len(atom_indices))):
            raise ValueError("Number of atoms must be consistent!")

        n_frames = self.xyz.shape[0]
        self_align_xyz = np.asarray(self.xyz[:, atom_indices, :], order='c')
        self_displace_xyz = np.asarray(self.xyz, order='c')
        ref_align_xyz = np.array(reference.xyz[frame, ref_atom_indices, :],
                                 copy=True, order='c').reshape(1, -1, 3)

        offset = np.mean(self_align_xyz, axis=1, dtype=np.float64).reshape(
            n_frames, 1, 3)
        self_align_xyz -= offset
        if self_align_xyz.ctypes.data != self_displace_xyz.ctypes.data:
            # when atom_indices is None, these two arrays alias the same memory
            # so we only need to do the centering once
            self_displace_xyz -= offset

        ref_offset = ref_align_xyz[0].astype('float64').mean(0)
        ref_align_xyz[0] -= ref_offset

        self_g = np.einsum('ijk,ijk->i', self_align_xyz, self_align_xyz)
        ref_g = np.einsum('ijk,ijk->i', ref_align_xyz, ref_align_xyz)

        _rmsd.superpose_atom_major(
            ref_align_xyz, self_align_xyz, ref_g, self_g, self_displace_xyz,
            0, parallel=parallel)

        self_displace_xyz += ref_offset
        self.xyz = self_displace_xyz
        return self


    def __getitem__(self, key):
        "Get a slice of this trajectory"
        return self.slice(key)

    def slice(self, key, copy=True):
        """Slice trajectory, by extracting one or more frames into a separate object

        This method can also be called using index bracket notation, i.e
        `traj[1] == traj.slice(1)`

        Parameters
        ----------
        key : {int, np.ndarray, slice}
            The slice to take. Can be either an int, a list of ints, or a slice
            object.
        copy : bool, default=True
            Copy the arrays after slicing. If you set this to false, then if
            you modify a slice, you'll modify the original array since they
            point to the same data.
        """
        xyz = self.xyz[key]
        time = self.time[key]
        unitcell_lengths, unitcell_angles = None, None
        if self.unitcell_angles is not None:
            unitcell_angles = self.unitcell_angles[key]
        if self.unitcell_lengths is not None:
            unitcell_lengths = self.unitcell_lengths[key]

        if copy:
            xyz = xyz.copy()
            time = time.copy()
            topology = deepcopy(self._topology)

            if self.unitcell_angles is not None:
                unitcell_angles = unitcell_angles.copy()
            if self.unitcell_lengths is not None:
                unitcell_lengths = unitcell_lengths.copy()
        else:
            topology = self._topology

        newtraj = self.__class__(
            xyz, topology, time, unitcell_lengths=unitcell_lengths,
            unitcell_angles=unitcell_angles)

        if self._rmsd_traces is not None:
            newtraj._rmsd_traces = np.array(self._rmsd_traces[key],
                                            ndmin=1, copy=True)
        return newtraj


    def openmm_positions(self, frame):
        """OpenMM-compatable positions of a single frame.

        Examples
        --------
        >>> t = md.load('trajectory.h5')
        >>> context.setPositions(t.openmm_positions(0))

        Parameters
        ----------
        frame : int
            The index of frame of the trajectory that you wish to extract

        Returns
        -------
        positions : list
            The cartesian coordinates of specific trajectory frame, formatted
            for input to OpenMM

        """
        from simtk.openmm import Vec3
        from simtk.unit import nanometer

        Pos = []
        for xyzi in self.xyz[frame]:
            Pos.append(Vec3(xyzi[0], xyzi[1], xyzi[2]))

        return Pos * nanometer

    def openmm_boxes(self, frame):
        """OpenMM-compatable box vectors of a single frame.

        Examples
        --------
        >>> t = md.load('trajectory.h5')
        >>> context.setPeriodicBoxVectors(t.openmm_positions(0))

        Parameters
        ----------
        frame : int
            Return box for this single frame.

        Returns
        -------
        box : tuple
            The periodic box vectors for this frame, formatted for input to
            OpenMM.
        """
        from simtk.openmm import Vec3
        from simtk.unit import nanometer

        vectors = self.unitcell_vectors[frame]
        if vectors is None:
            raise ValueError(
                "this trajectory does not contain box size information")

        v1, v2, v3 = vectors
        return (Vec3(*v1), Vec3(*v2), Vec3(*v3)) * nanometer

    @staticmethod
    # im not really sure if the load function should be just a function or a method on the class
    # so effectively, lets make it both?
    def load(filenames, **kwargs):
        """Load a trajectory from disk

        Parameters
        ----------
        filenames : {str, [str]}
            Either a string or list of strings

        Other Parameters
        ----------------
        As requested by the various load functions -- it depends on the extension
        """
        return load(filenames, **kwargs)

    def _savers(self):
        """Return a dictionary mapping extensions to the appropriate format-specific save
        function"""
        return {'.xtc': self.save_xtc,
                '.trr': self.save_trr,
                '.pdb': self.save_pdb,
                '.pdb.gz': self.save_pdb,
                '.dcd': self.save_dcd,
                '.h5': self.save_hdf5,
                '.binpos': self.save_binpos,
                '.nc': self.save_netcdf,
                '.netcdf': self.save_netcdf,
                '.ncrst': self.save_netcdfrst,
                '.crd': self.save_mdcrd,
                '.mdcrd': self.save_mdcrd,
                '.ncdf': self.save_netcdf,
                '.lh5': self.save_lh5,
                '.lammpstrj': self.save_lammpstrj,
                '.xyz': self.save_xyz,
                '.xyz.gz': self.save_xyz,
                '.gro': self.save_gro,
                '.rst7': self.save_amberrst7,
                '.tng': self.save_tng,
                }


    def center_coordinates(self, mass_weighted=False):
        """Center each trajectory frame at the origin (0,0,0).

        This method acts inplace on the trajectory.  The centering can
        be either uniformly weighted (mass_weighted=False) or weighted by
        the mass of each atom (mass_weighted=True).

        Parameters
        ----------
        mass_weighted : bool, optional (default = False)
            If True, weight atoms by mass when removing COM.

        Returns
        -------
        self
        """
        if mass_weighted and self.top is not None:
            self.xyz -= distance.compute_center_of_mass(self)[:, np.newaxis, :]
        else:
            self._rmsd_traces = _rmsd._center_inplace_atom_major(self._xyz)

        return self

    @deprecated(
        'restrict_atoms was replaced by atom_slice and will be removed in 2.0')
    def restrict_atoms(self, atom_indices, inplace=True):
        """Retain only a subset of the atoms in a trajectory

        Deletes atoms not in `atom_indices`, and re-indexes those that remain

        Parameters
        ----------
        atom_indices : array-like, dtype=int, shape=(n_atoms)
            List of atom indices to keep.
        inplace : bool, default=True
            If ``True``, the operation is done inplace, modifying ``self``.
            Otherwise, a copy is returned with the restricted atoms, and
            ``self`` is not modified.

        Returns
        -------
        traj : md.Trajectory
            The return value is either ``self``, or the new trajectory,
            depending on the value of ``inplace``.
        """
        return self.atom_slice(atom_indices, inplace=inplace)

    def atom_slice(self, atom_indices, inplace=False):
        """Create a new trajectory from a subset of atoms

        Parameters
        ----------
        atom_indices : array-like, dtype=int, shape=(n_atoms)
            List of indices of atoms to retain in the new trajectory.
        inplace : bool, default=False
            If ``True``, the operation is done inplace, modifying ``self``.
            Otherwise, a copy is returned with the sliced atoms, and
            ``self`` is not modified.

        Returns
        -------
        traj : md.Trajectory
            The return value is either ``self``, or the new trajectory,
            depending on the value of ``inplace``.

        See Also
        --------
        stack : `Trajectory` method
            stack multiple trajectories along the atom axis
        """
        xyz = np.array(self.xyz[:, atom_indices], order='C')
        topology = None
        if self._topology is not None:
            topology = self._topology.subset(atom_indices)

        if inplace:
            if self._topology is not None:
                self._topology = topology
            self._xyz = xyz

            return self

        unitcell_lengths = unitcell_angles = None
        if self._have_unitcell:
            unitcell_lengths = self._unitcell_lengths.copy()
            unitcell_angles = self._unitcell_angles.copy()
        time = self._time.copy()

        return Trajectory(xyz=xyz, topology=topology, time=time,
                          unitcell_lengths=unitcell_lengths,
                          unitcell_angles=unitcell_angles)

    def remove_solvent(self, exclude=None, inplace=False):
        """
        Create a new trajectory without solvent atoms

        Parameters
        ----------
        exclude : array-like, dtype=str, shape=(n_solvent_types)
            List of solvent residue names to retain in the new trajectory.
        inplace : bool, default=False
            The return value is either ``self``, or the new trajectory,
            depending on the value of ``inplace``.

        Returns
        -------
        traj : md.Trajectory
            The return value is either ``self``, or the new trajectory,
            depending on the value of ``inplace``.
        """
        solvent_types = list(_SOLVENT_TYPES)

        if exclude is not None:

            if isinstance(exclude, str):
                raise TypeError('exclude must be array-like')
            if not isinstance(exclude, Iterable):
                raise TypeError('exclude is not iterable')

            for type in exclude:
                if type not in solvent_types:
                    raise ValueError(type + 'is not a valid solvent type')
                solvent_types.remove(type)

        atom_indices = [atom.index for atom in self.topology.atoms if
                        atom.residue.name not in solvent_types]

        return self.atom_slice(atom_indices, inplace=inplace)

    def smooth(self, width, order=3, atom_indices=None, inplace=False):
        """Smoothen a trajectory using a zero-delay Buttersworth filter. Please
        note that for optimal results the trajectory should be properly aligned
        prior to smoothing (see `md.Trajectory.superpose`).

        Parameters
        ----------
        width : int
            This acts very similar to the window size in a moving average
            smoother. In this implementation, the frequency of the low-pass
            filter is taken to be two over this width, so it's like
            "half the period" of the sinusiod where the filter starts
            to kick in. Must be an integer greater than one.
        order : int, optional, default=3
            The order of the filter. A small odd number is recommended. Higher
            order filters cutoff more quickly, but have worse numerical
            properties.
        atom_indices : array-like, dtype=int, shape=(n_atoms), default=None
            List of indices of atoms to retain in the new trajectory.
            Default is set to `None`, which applies smoothing to all atoms.
        inplace : bool, default=False
            The return value is either ``self``, or the new trajectory,
            depending on the value of ``inplace``.

        Returns
        -------
        traj : md.Trajectory
            The return value is either ``self``, or the new smoothed trajectory,
            depending on the value of ``inplace``.

        References
        ----------
        .. [1] "FiltFilt". Scipy Cookbook. SciPy. <http://www.scipy.org/Cookbook/FiltFilt>.
        """
        from scipy.signal import lfilter, lfilter_zi, filtfilt, butter

        if width < 2.0 or not isinstance(width, int):
            raise ValueError('width must be an integer greater than 1.')
        if not atom_indices:
            atom_indices = range(self.n_atoms)

        # find nearest odd integer
        pad = int(np.ceil((width + 1) / 2) * 2 - 1)

        # Use lfilter_zi to choose the initial condition of the filter.
        b, a = butter(order, 2.0 / width)
        zi = lfilter_zi(b, a)

        xyz = self.xyz.copy()

        for i in atom_indices:
            for j in range(3):
                signal = xyz[:, i, j]
                padded = np.r_[
                    signal[pad - 1: 0: -1], signal, signal[-1: -pad: -1]]

                # Apply the filter to the width.
                z, _ = lfilter(b, a, padded, zi=zi * padded[0])

                # Apply the filter again, to have a result filtered at an order
                # the same as filtfilt.
                z2, _ = lfilter(b, a, z, zi=zi * z[0])

                # Use filtfilt to apply the filter.
                output = filtfilt(b, a, padded)

                xyz[:, i, j] = output[(pad - 1): -(pad - 1)]

        if not inplace:
            return Trajectory(xyz=xyz, topology=self.topology,
                              time=self.time,
                              unitcell_lengths=self.unitcell_lengths,
                              unitcell_angles=self.unitcell_angles)

        self.xyz = xyz

    def _check_valid_unitcell(self):
        """Do some sanity checking on self.unitcell_lengths and self.unitcell_angles
        """
        if self.unitcell_lengths is not None and self.unitcell_angles is None:
            raise AttributeError('unitcell length data exists, but no angles')
        if self.unitcell_lengths is None and self.unitcell_angles is not None:
            raise AttributeError('unitcell angles data exists, but no lengths')

        if self.unitcell_lengths is not None and np.any(
                        self.unitcell_lengths < 0):
            raise ValueError('unitcell length < 0')

        if self.unitcell_angles is not None and np.any(
                        self.unitcell_angles < 0):
            raise ValueError('unitcell angle < 0')

    @property
    def _have_unitcell(self):
        return self._unitcell_lengths is not None and self._unitcell_angles is not None

    def make_molecules_whole(self, inplace=False, sorted_bonds=None):
        """Only make molecules whole

        Parameters
        ----------
        inplace : bool
            If False, a new Trajectory is created and returned.
            If True, this Trajectory is modified directly.
        sorted_bonds : array of shape (n_bonds, 2)
            Pairs of atom indices that define bonds, in sorted order.
            If not specified, these will be determined from the trajectory's
            topology.

        See Also
        --------
        image_molecules()
        """
        unitcell_vectors = self.unitcell_vectors
        if unitcell_vectors is None:
            raise ValueError('This Trajectory does not define a periodic unit cell')

        if inplace:
            result = self
        else:
            result = Trajectory(xyz=self.xyz, topology=self.topology,
                                time=self.time,
                                unitcell_lengths=self.unitcell_lengths,
                                unitcell_angles=self.unitcell_angles)

        if sorted_bonds is None:
            sorted_bonds = sorted(self._topology.bonds, key=lambda bond: bond[0].index)
            sorted_bonds = np.asarray([[b0.index, b1.index] for b0, b1 in sorted_bonds])

        box = np.asarray(result.unitcell_vectors, order='c')
        _geometry.whole_molecules(result.xyz, box, sorted_bonds)
        if not inplace:
            return result
        return self

    def image_molecules(self, inplace=False, anchor_molecules=None, other_molecules=None, sorted_bonds=None, make_whole=True):
        """Recenter and apply periodic boundary conditions to the molecules in each frame of the trajectory.

        This method is useful for visualizing a trajectory in which molecules were not wrapped
        to the periodic unit cell, or in which the macromolecules are not centered with respect
        to the solvent.  It tries to be intelligent in deciding what molecules to center, so you
        can simply call it and trust that it will "do the right thing".

        Parameters
        ----------
        inplace : bool, default=False
            If False, a new Trajectory is created and returned.  If True, this Trajectory
            is modified directly.
        anchor_molecules : list of atom sets, optional, default=None
            Molecule that should be treated as an "anchor".
            These molecules will be centered in the box and put near each other.
            If not specified, anchor molecules are guessed using a heuristic.
        other_molecules : list of atom sets, optional, default=None
            Molecules that are not anchors. If not specified,
            these will be molecules other than the anchor molecules
        sorted_bonds : array of shape (n_bonds, 2)
            Pairs of atom indices that define bonds, in sorted order.
            If not specified, these will be determined from the trajectory's
            topology. Only relevant if ``make_whole`` is True.
        make_whole : bool
            Whether to make molecules whole.

        Returns
        -------
        traj : md.Trajectory
            The return value is either ``self`` or the new trajectory,
            depending on the value of ``inplace``.

        See Also
        --------
        Topology.guess_anchor_molecules
        """
        unitcell_vectors = self.unitcell_vectors
        if unitcell_vectors is None:
            raise ValueError('This Trajectory does not define a periodic unit cell')

        if anchor_molecules is None:
            anchor_molecules = self.topology.guess_anchor_molecules()

        if other_molecules is None:
            # Determine other molecules by which molecules are not anchor molecules
            molecules = self._topology.find_molecules()
            other_molecules = [mol for mol in molecules if mol not in anchor_molecules]

        # Expand molecules into atom indices
        anchor_molecules_atom_indices = [np.fromiter((a.index for a in mol), dtype=np.int32) for mol in anchor_molecules]
        other_molecules_atom_indices  = [np.fromiter((a.index for a in mol), dtype=np.int32) for mol in other_molecules]

        if inplace:
            result = self
        else:
            result = Trajectory(xyz=self.xyz, topology=self.topology, time=self.time,
                unitcell_lengths=self.unitcell_lengths, unitcell_angles=self.unitcell_angles)

        if make_whole and sorted_bonds is None:
            sorted_bonds = sorted(self._topology.bonds, key=lambda bond: bond[0].index)
            sorted_bonds = np.asarray([[b0.index, b1.index] for b0, b1 in sorted_bonds])
        elif not make_whole:
            sorted_bonds = None

        box = np.asarray(result.unitcell_vectors, order='c')
        _geometry.image_molecules(result.xyz, box, anchor_molecules_atom_indices, other_molecules_atom_indices, sorted_bonds)
        if not inplace:
            return result
        return self
