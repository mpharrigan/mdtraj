##############################################################################
# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2013 Stanford University and the Authors
#
# Authors: Robert McGibbon
# Contributors:
#
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
##############################################################################


"""The mdtraj package contains tools for loading and saving molecular dynamics
trajectories in a variety of formats, including Gromacs XTC & TRR, CHARMM/NAMD
DCD, AMBER BINPOS, PDB, and HDF5.
"""

from .formats import (load_xtc, load_trr, load_hdf5, load_lh5, load_netcdf,
                      load_mdcrd, load_dcd, load_binpos,
                      load_pdb, load_arc, load_xml, load_prmtop, load_psf,
                      load_mol2, load_restrt, load_ncrestrt, load_lammpstrj,
                      load_dtr, load_xyz, load_hoomdxml, load_tng, load_stk)

# from mdtraj.formats.registry import FormatRegistry


from mdtraj.core import element
from mdtraj._rmsd import rmsd
from mdtraj._lprmsd import lprmsd
from mdtraj.core.topology import Topology
from mdtraj.geometry import *
from mdtraj.core.trajectory import *
from mdtraj.nmr import *
import mdtraj.reporters


def test(label='full', verbose=2, extra_argv=None, doctests=False):
    """Run tests for mdtraj using nose.

    Parameters
    ----------
    label : {'fast', 'full'}
        Identifies the tests to run. The fast tests take about 10 seconds,
        and the full test suite takes about two minutes (as of this writing).
    verbose : int, optional
        Verbosity value for test outputs, in the range 1-10. Default is 2.
    """
    import mdtraj
    from mdtraj.testing.nosetester import MDTrajTester
    tester = MDTrajTester(mdtraj)
    return tester.test(label=label, verbose=verbose, extra_argv=extra_argv)


# prevent nose from discovering this function, or otherwise when its run
# the test suite in an infinite loop
test.__test__ = False


def capi():
    import os
    import sys
    module_path = sys.modules['mdtraj'].__path__[0]
    return {
        'lib_dir': os.path.join(module_path, 'core', 'lib'),
        'include_dir': os.path.join(module_path, 'core', 'lib'),
    }
