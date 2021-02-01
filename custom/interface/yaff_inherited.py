from yaff.pes.ff import ForcePart
import numpy as np
from yaff import *
import h5py as h5
from molmod.periodic import periodic
from ase import Atoms
import pdb
import time
from schnetpack2.custom.md import AtomsConverter
import torch
import yaff

def atoms2yaff(at):
    numbers = at.get_atomic_numbers()
    pos = at.get_positions()*angstrom
    rvecs = at.get_cell()*angstrom
    system = System( numbers, pos, rvecs=rvecs)
    return system
    
def yaff2atoms(sys):
    atom = Atoms(numbers=sys.numbers,positions=sys.pos/angstrom,cell=sys.cell.rvecs/angstrom,pbc=True)
    return atom

class AtomsAppender(Hook):
    '''Aggregates atoms'''
    def __init__(self, start=0, step=1):
        Hook.__init__(self, start, step)

    def __call__(self,iterative):
        iterative.ff.atoms.append(yaff2atoms(iterative.ff.system))

class VerletScreenLog(Hook):#TODO let current code be inspiration, for storage use hdf5 file, iterative should contain the right properties eg sigma etc
    '''A screen logger for the Verlet algorithm'''
    def __init__(self, start=0, step=1):
        Hook.__init__(self, start, step)
        self.time0 = None

    def __call__(self, iterative):
        if log.do_medium:
            if self.time0 is None:
                self.time0 = time.time()
                if log.do_medium:
                    log.hline()
                    log('Cons.Err. =&the root of the ratio of the variance on the conserved quantity and the variance on the kinetic energy.')
                    log('d-rmsd    =&the root-mean-square displacement of the atoms.')
                    log('g-rmsd    =&the root-mean-square gradient of the energy.')
                    log('counter  Cons.Err.       Temp     d-RMSD     g-RMSD    Sigma   Walltime')
                    log.hline()
            log('%7i %10.5f %s %s %s %10.1f %10.3f' % (
                iterative.counter,
                iterative.cons_err,
                log.temperature(iterative.temp),
                log.length(iterative.rmsd_delta),
                log.force(iterative.rmsd_gpos),
                iterative.ff.sigma,
                time.time() - self.time0,
            ))

yaff.sampling.verlet.VerletScreenLog = VerletScreenLog

class SchnetForceField(ForcePart):
    '''Base class for anything that can compute energies (and optionally gradient
       and virial) for a ``System`` object.
    '''
    def __init__(self, name, atom, model, conv, env, uncertainty=True):
        """
           **Arguments:**
           name
                A name for this part of the force field. This name must adhere
                to the following conventions: all lower case, no white space,
                and short. It is used to construct part_* attributes in the
                ForceField class, where * is the name.
           system
                The system to which this part of the FF applies.
        """
        self.name = name
        self.system = atoms2yaff(atom)
        self.model  = model
        self.conv   = conv
        self.env    = env
        self.parts  = [self] 
        # backup copies of last call to compute:
        self.energy = 0.0
        self.sigma  = 0.0
        self.gpos = np.zeros((self.system.natom, 3), float)
        self.sigma_forces = torch.tensor(np.zeros((self.system.natom, 3), float))
        self.vtens = np.zeros((3, 3), float)
        self.atoms = []
        self.clear()
        

    def clear(self):
        """Fill in nan values in the cached results to indicate that they have
           become invalid.
        """
        self.energy = np.nan
        self.sigma  = np.nan
        self.gpos[:] = np.nan
        self.sigma_forces[:] = np.nan
        self.vtens[:] = np.nan
        
    def update_rvecs(self, rvecs):
        '''Let the ``ForcePart`` object know that the cell vectors have changed.
           **Arguments:**
           rvecs
                The new cell vectors.
        '''
        self.clear()

    def update_pos(self, pos):
        '''Let the ``ForcePart`` object know that the atomic positions have changed.
           **Arguments:**
           pos
                The new atomic coordinates.
        '''
        self.clear()

    def compute(self, gpos=None, vtens=None):
        """Compute the energy and optionally some derivatives for this FF (part)
           The only variable inputs for the compute routine are the atomic
           positions and the cell vectors, which can be changed through the
           ``update_rvecs`` and ``update_pos`` methods. All other aspects of
           a force field are considered to be fixed between subsequent compute
           calls. If changes other than positions or cell vectors are needed,
           one must construct new ``ForceField`` and/or ``ForcePart`` objects.
           **Optional arguments:**
           gpos
                The derivatives of the energy towards the Cartesian coordinates
                of the atoms. ('g' stands for gradient and 'pos' for positions.)
                This must be a writeable numpy array with shape (N, 3) where N
                is the number of atoms.
           vtens
                The force contribution to the pressure tensor. This is also
                known as the virial tensor. It represents the derivative of the
                energy towards uniform deformations, including changes in the
                shape of the unit cell. (v stands for virial and 'tens' stands
                for tensor.) This must be a writeable numpy array with shape (3,
                3).
           The energy is returned. The optional arguments are Fortran-style
           output arguments. When they are present, the corresponding results
           are computed and **added** to the current contents of the array.
        """
        if gpos is None:
            my_gpos = None
        else:
            my_gpos = self.gpos
            my_gpos[:] = 0.0
        if vtens is None:
            my_vtens = None
        else:
            my_vtens = self.vtens
            my_vtens[:] = 0.0

        self.energy, self.sigma, self.sigma_ensemble, self.sigma_forces = self._internal_compute(my_gpos, my_vtens)

        if np.isnan(self.energy):
            raise ValueError('The energy is not-a-number (nan).')
        if gpos is not None:
            if np.isnan(my_gpos).any():
                raise ValueError('Some gpos element(s) is/are not-a-number (nan).')
            gpos += my_gpos
        if vtens is not None:
            if np.isnan(my_vtens).any():
                raise ValueError('Some vtens element(s) is/are not-a-number (nan).')
            vtens += my_vtens
        return self.energy

    def _internal_compute(self, gpos, vtens):
        results = self.model(self.yaff2schnet(self.system))
        
        energy, new_gpos = results['y'].cpu().squeeze().item()*electronvolt, results['dydx'].cpu().detach().cpu().squeeze().numpy()*electronvolt / angstrom
        if 'stress' in results.keys(): new_vtens = results['stress'] 
        else: new_vtens = 0

        if 'sigma' in results.keys(): sigma = results['sigma'] * electronvolt
        else: sigma = torch.tensor([0])

        if 'sigma_ensemble' in results.keys(): sigma_ensemble = results['sigma_ensemble'] * electronvolt
        else: sigma_ensemble = torch.tensor([0])

        if 'sigma_forces' in results.keys(): sigma_forces = results['sigma_forces'] * electronvolt / angstrom
        else: sigma_forces = torch.tensor(np.zeros_like(new_gpos))
        
        if not gpos is None:
            gpos[:, :] = - new_gpos

        if not vtens is None: # vtens = F x R
            #vtens[:, :] = new_vtens * electronvolt   hier stond /1602176.6208
            vtens[:, :] = new_vtens*electronvolt#* np.abs(np.linalg.det(self.system.cell.rvecs))*#/angstrom**3 /2000 * electronvolt

        return energy, sigma.detach().cpu().squeeze().item(), sigma_ensemble.detach().cpu().numpy(), sigma_forces.detach().cpu().squeeze().numpy() #gpos not included as passed through gpos reference

    def NVE(self, steps, nprint = 250, nwrite=1, dt = 1, temp = 600, start = 0, name = 'run',
            restart = None):
#        ff = ForceField(self.system, [self])

#        f = h5.File(name + '.h5', mode = 'w')
#        hdf5_writer = HDF5Writer(f, start = start, step = nprint)
        sl = VerletScreenLog(step = nprint)
        xyz = XYZWriter(name + '.xyz', start = start, step = nwrite)
    
        f2 = h5.File(name + '_restart.h5', mode = 'w')
        restart_writer = RestartWriter(f2, start = start, step = nprint*20)
    
        verlet = VerletIntegrator(self, dt * femtosecond, state=[SigmaStateItem(), SigmaEnsembleStateItem(), SigmaForcesStateItem()], hooks = [restart_writer, hdf5_writer, sl, xyz, hdf5_writer], temp0 = temp, restart_h5 = restart)
        verlet.run(steps)
    
        #f.close()
    
    def NPT(self, steps, nprint = 1, nwrite=1, dt = 1, temp = 600, start = 0, name = 'run',thermotime=100,barotime=1000,annealing=0):
#        ff = ForceField(self.system, [self])
    
        thermo = NHCThermostat(temp = temp, timecon = thermotime *femtosecond)
    
        baro = MTKBarostat(self, temp = temp, press = 1 * 1e+05 * pascal,timecon=barotime *femtosecond)
        tbc = TBCombination(thermo, baro)
    
        f = h5.File(name + '.h5', mode = 'w')
        hdf5_writer = HDF5Writer(f, start = start, step = nwrite)
        sl = VerletScreenLog(step = nprint)
        xyz = XYZWriter(name + '.xyz', start = start, step = nwrite)
        aa = AtomsAppender(start = start, step = nwrite)
        f2 = h5.File(name + '_restart.h5', mode = 'w')
        restart_writer = RestartWriter(f2, start = start, step =nprint*20)
    
        verlet = VerletIntegrator(self, dt * femtosecond, state=[SigmaStateItem(), SigmaEnsembleStateItem(), SigmaForcesStateItem()], hooks = [aa,restart_writer, hdf5_writer, sl, tbc, xyz], temp0 = temp)
        #verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, tbc, xyz], temp0 = temp)
        verlet.run(steps)
    
        #f.close()
        
    def NVT(self, steps, nprint = 250, nwrite=1, dt = 1, temp = 600, start = 0, name = 'run', timecon=100,annealing=0):
#        ff = ForceField(self.system, [self])
        if annealing != 0:
            thermo = AndersenThermostat(temp = temp,annealing=annealing)
        else:
            thermo = NHCThermostat(temp = temp,timecon=timecon*femtosecond)
            #NHC timecon=timecon 
        #baro = MTKBarostat(ff, temp = temp, press = 1 * 1e+05 * pascal)
        #tbc = TBCombination(thermo, baro)
    
        f = h5.File(name + '.h5', mode = 'w')
        hdf5_writer = HDF5Writer(f, start = start, step = nwrite)
        sl = VerletScreenLog(step = nprint)
        xyz = XYZWriter(name + '.xyz', start = start, step = nwrite)
        aa = AtomsAppender(start = start, step = nwrite)
        f2 = h5.File(name + '_restart.h5', mode = 'w')
        restart_writer = RestartWriter(f2, start = start, step = nprint*20)
        verlet = VerletIntegrator(self, dt * femtosecond, state=[SigmaStateItem(), SigmaEnsembleStateItem(), SigmaForcesStateItem()], hooks = [aa,restart_writer, hdf5_writer, sl, thermo, xyz], temp0 = temp)
        #verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, thermo, xyz], temp0 = temp)
        verlet.run(steps)
    
        #f2.close()
    def NVE(self, steps, nprint = 250, nwrite=1, dt = 1, temp = 600, start = 0, name = 'run',annealing=0):
        f = h5.File(name + '.h5', mode = 'w')
        hdf5_writer = HDF5Writer(f, start = start, step = nwrite)
        sl = VerletScreenLog(step = nprint)
        xyz = XYZWriter(name + '.xyz', start = start, step = nwrite)
        aa = AtomsAppender(start = start, step = nwrite)
        f2 = h5.File(name + '_restart.h5', mode = 'w')
        restart_writer = RestartWriter(f2, start = start, step = nprint*20)
        verlet = VerletIntegrator(self, dt * femtosecond, state=[SigmaStateItem(), SigmaEnsembleStateItem(), SigmaForcesStateItem()], hooks = [aa,restart_writer, hdf5_writer, sl, xyz], temp0 = temp)
        #verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, thermo, xyz], temp0 = temp)
        verlet.run(steps)
    

    def atoms2schnet(self,at):
        return self.conv.convert_atoms(atoms=at)
    
    def yaff2schnet(self,at):
        return self.atoms2schnet(yaff2atoms(at))

    def update_rvecs(self, rvecs):
        '''See :meth:`yaff.pes.ff.ForcePart.update_rvecs`'''
        ForcePart.update_rvecs(self, rvecs)
        self.system.cell.update_rvecs(rvecs)

    def update_pos(self, pos):
        '''See :meth:`yaff.pes.ff.ForcePart.update_pos`'''
        ForcePart.update_pos(self, pos)
        self.system.pos[:] = pos
        

    def update_rvecs(self, rvecs):
        '''See :meth:`yaff.pes.ff.ForcePart.update_rvecs`'''
        ForcePart.update_rvecs(self, rvecs)
        self.system.cell.update_rvecs(rvecs)

    def update_pos(self, pos):
        '''See :meth:`yaff.pes.ff.ForcePart.update_pos`'''
        ForcePart.update_pos(self, pos)
        self.system.pos[:] = pos
        
        
class VerletIntegratorUncertainties(VerletIntegrator):
    def propagate(self):
        # Allow specialized hooks to modify the state before the regular verlet
        # step.
        self.call_verlet_hooks('pre')

        # Regular verlet step
        self.acc = -self.gpos/self.masses.reshape(-1,1)
        self.vel += 0.5*self.acc*self.timestep
        self.pos += self.timestep*self.vel
        self.ff.update_pos(self.pos)
        self.gpos[:] = 0.0
        self.vtens[:] = 0.0
        self.epot, self.sigma, self.sigma_forces  = self.ff.compute(self.gpos, self.vtens)
        self.acc = -self.gpos/self.masses.reshape(-1,1)
        self.vel += 0.5*self.acc*self.timestep
        self.ekin = self._compute_ekin()

        # Allow specialized verlet hooks to modify the state after the step
        self.call_verlet_hooks('post')

        # Calculate the total position change
        self.posnieuw = self.pos.copy()
        self.delta[:] = self.posnieuw-self.posoud
        self.posoud[:] = self.posnieuw

        # Common post-processing of a single step
        self.time += self.timestep
        self.compute_properties()
        Iterative.propagate(self) # Includes call to conventional hooks
        
class SigmaStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, 'sigma')

    def get_value(self, iterative):
        return iterative.ff.sigma

class SigmaEnsembleStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, 'sigma')

    def get_value(self, iterative):
        return iterative.ff.sigma_ensemble

class SigmaForcesStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, 'sigma_forces')

    def get_value(self, iterative):
        return iterative.ff.sigma_forces

