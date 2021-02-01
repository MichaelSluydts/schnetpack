import numpy as np
from yaff import *
import h5py as h5
from molmod.periodic import periodic
from ase import Atoms
from schnetpack2.md import AtomsConverter

#from schnetpack2.md import AtomsConverter
#
#conv = AtomsConverter(environment_provider=env)

def atoms2yaff(at):
    numbers = at.get_atomic_numbers()
    pos = at.get_positions()*angstrom
    rvecs = at.get_cell()*angstrom
    system = System( numbers, pos, rvecs=rvecs)
    return system
    
def yaff2atoms(sys):
    atom = Atoms(numbers=sys.numbers,positions=sys.pos/angstrom,cell=sys.cell.rvecs/angstrom,pbc=True)
    return atom

class ML_FF(ForcePart):
    def __init__(self, atom, model, conv, env):
        self.system = atoms2yaff(atom)
        self.model  = model
        self.conv   = conv
        self.env    = env 
        ForcePart.__init__(self, 'ml_ff', self.system)

#    def _internal_compute(self, gpos, vtens):
#        self.model(gpos)
#        atin = test.convert_atoms(atoms=atoms)
#
#        energy, new_gpos, new_vtens = self.model.compute_md(self.system.numbers, self.system.pos / angstrom, self.system.cell.rvecs / angstrom)
#
#        if not gpos is None:
#            gpos[:, :] = - new_gpos * electronvolt / angstrom
#
#        if not vtens is None: # vtens = F x R
#            vtens[:, :] = new_vtens * electronvolt
#
#        return energy * electronvolt

    def _internal_compute(self, gpos, vtens):

        results = self.model(self.yaff2schnet(self.system))
        if 'stress' in results.keys():
            energy, new_gpos, new_vtens = results['y'], results['dydx'], results['stress']
        else:
            energy, new_gpos, new_vtens = results['y'], results['dydx'], 0
        
        if not gpos is None:
            gpos[:, :] = - new_gpos * electronvolt / angstrom

        if not vtens is None: # vtens = F x R
            #vtens[:, :] = new_vtens * electronvolt
            vtens[:, :] = new_vtens / 1602176.6208 * np.abs(np.linalg.det(self.system.cell.rvecs))/angstrom**3 /2000 * electronvolt

        return energy * electronvolt

    def NVE(self, steps, nprint = 10, dt = 1, temp = 600, start = 0, name = 'run',
            restart = None):
        ff = ForceField(self.system, [self])

#        f = h5.File(name + '.h5', mode = 'w')
#        hdf5_writer = HDF5Writer(f, start = start, step = nprint)
        sl = VerletScreenLog(step = nprint)
        xyz = XYZWriter(name + '.xyz', start = start, step = nprint)
    
        f2 = h5.File(name + '_restart.h5', mode = 'w')
        restart_writer = RestartWriter(f2, start = start, step = 5000)
    
        verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [restart_writer, sl, xyz, hdf5_writer], temp0 = temp, restart_h5 = restart)
        verlet.run(steps)
    
        #f.close()
    
    def NPT(self, steps, nprint = 10, dt = 1, temp = 600, start = 0, name = 'run'):
        ff = ForceField(self.system, [self])
    
        thermo = NHCThermostat(temp = temp)
    
        baro = MTKBarostat(ff, temp = temp, press = 1 * 1e+05 * pascal,timecon=3e6)
        tbc = TBCombination(thermo, baro)
    
        #f = h5.File(name + '.h5', mode = 'w')
        #hdf5_writer = HDF5Writer(f, start = start, step = nprint)
        sl = VerletScreenLog(step = nprint)
        xyz = XYZWriter(name + '.xyz', start = start, step = nprint)
        f2 = h5.File(name + '_restart.h5', mode = 'w')
        restart_writer = RestartWriter(f2, start = start, step = 5000)
    
        verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [restart_writer, sl, tbc, xyz], temp0 = temp)
        #verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, tbc, xyz], temp0 = temp)
        verlet.run(steps)
    
        #f.close()
        
    def NVT(self, steps, nprint = 10, dt = 1, temp = 600, start = 0, name = 'run'):
        ff = ForceField(self.system, [self])
    
        thermo = NHCThermostat(temp = temp)
    
        #baro = MTKBarostat(ff, temp = temp, press = 1 * 1e+05 * pascal)
        #tbc = TBCombination(thermo, baro)
    
        #f = h5.File(name + '.h5', mode = 'w')
        #hdf5_writer = HDF5Writer(f, start = start, step = nprint)
        sl = VerletScreenLog(step = 10)
        xyz = XYZWriter(name + '.xyz', start = start, step = nprint)
        f2 = h5.File(name + '_restart.h5', mode = 'w')
        restart_writer = RestartWriter(f2, start = start, step = 5000)
    
        verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [restart_writer, sl, thermo, xyz], temp0 = temp)
        #verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, thermo, xyz], temp0 = temp)
        verlet.run(steps)
    
        #f2.close()
    
    def atoms2schnet(self,at):
        return self.conv.convert_atoms(atoms=at)
    
    def yaff2schnet(self,at):
        return self.atoms2schnet(yaff2atoms(at))
