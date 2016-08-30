from __future__ import absolute_import

from .amberrst import (AmberRestartFile, AmberNetCDFRestartFile,
                       load_restrt, load_ncrestrt)
from .arc import ArcTrajectoryFile, load_arc
from .binpos import BINPOSTrajectoryFile, load_binpos
from .dcd import DCDTrajectoryFile, load_dcd
from .dtr import DTRTrajectoryFile, load_stk, load_dtr
from .gro import GroTrajectoryFile, load_gro
from .hdf5 import HDF5TrajectoryFile, load_hdf5
from .hoomdxml import load_hoomdxml
from .lammpstrj import LAMMPSTrajectoryFile, load_lammpstrj
from .lh5 import LH5TrajectoryFile, load_lh5
from .mdcrd import MDCRDTrajectoryFile, load_mdcrd
from .mol2 import load_mol2
from .netcdf import NetCDFTrajectoryFile, load_netcdf
from .openmmxml import load_xml
from .pdb import PDBTrajectoryFile
from .pdb import load_pdb
from .prmtop import load_prmtop
from .psf import load_psf
from .tng import TNGTrajectoryFile, load_tng
from .trr import TRRTrajectoryFile, load_trr
from .xtc import XTCTrajectoryFile, load_xtc
from .xyzfile import XYZTrajectoryFile, load_xyz


def _from_traj(func):
    def _load_frame_as_top(fn, **kwargs):
        return func(fn, frame=0, **kwargs).topology

    return _load_frame_as_top


TOPOLOGY_LOADERS = {
    'pdb': _from_traj(load_pdb),
    'pdb.gz': _from_traj(load_pdb),
    'h5': _from_traj(load_hdf5),
    'lh5': _from_traj(load_lh5),
    'prmtop': load_prmtop,
    'parm7': load_prmtop,
    'psf': load_psf,
    'gro': _from_traj(load_gro),
    'arc': _from_traj(load_arc),
    'hoomdxml': _from_traj(load_hoomdxml),
    'mol2': _from_traj(load_mol2),
}

TRAJECTORY_LOADERS = {
    'rst7': load_restrt,
    'restrt': load_restrt,
    'inpcrd': load_restrt,
    'ncrst': load_ncrestrt,
    'arc': load_arc,
    'binpos': load_binpos,
    'dcd': load_dcd,
    'dtr': load_dtr,
    'stk': load_stk, # TODO: Does this even work?
    'gro': load_gro,
    'h5': load_hdf5,
    'hdf5': load_hdf5,
    'lammpstrj': load_lammpstrj,
    'lh5': load_lh5,
    'mdcrd': load_mdcrd,
    'crd': load_mdcrd,
    'mol2': load_mol2,
    'nc': load_netcdf,
    'netcdf': load_netcdf,
    'ncdf': load_netcdf,
    'xml': load_xml,
    'tng': load_tng,
    'trr': load_trr,
    'xtc': load_xtc,
    'xyz': XYZTrajectoryFile,
    'pdb': load_pdb,
}

TRAJECTORY_FILEOBJECTS = {
    'rst7': AmberRestartFile,
    'restrt': AmberRestartFile,
    'inpcrd': AmberRestartFile,
    'ncrst': AmberNetCDFRestartFile,
    'arc': ArcTrajectoryFile,
    'binpos': BINPOSTrajectoryFile,
    'dcd': DCDTrajectoryFile,
    'dtr': DTRTrajectoryFile,
    'gro': GroTrajectoryFile,
    'h5': HDF5TrajectoryFile,
    'hdf5': HDF5TrajectoryFile,
    'lammpstrj': LAMMPSTrajectoryFile,
    'lh5': LH5TrajectoryFile,
    'mdcrd': MDCRDTrajectoryFile,
    'crd': MDCRDTrajectoryFile,
    'nc': NetCDFTrajectoryFile,
    'netcdf': NetCDFTrajectoryFile,
    'ncdf': NetCDFTrajectoryFile,
    'tng': TNGTrajectoryFile,
    'trr': TRRTrajectoryFile,
    'xtc': XTCTrajectoryFile,
    'xtz': XYZTrajectoryFile,
    'pdb': PDBTrajectoryFile,
}