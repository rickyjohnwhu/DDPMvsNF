from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import openmmplumed
import mdtraj as md
import numpy as np
from openmm.app import modeller
from openmm.app.pdbfile import PDBFile
import pandas as pd
import matplotlib.pyplot as plt
from parmed import load_file

def run_simulation():

    directory = './aib9_openmm/'

    gro = load_file(directory + 'aib9.gro')
    top = load_file(directory + 'topol.top')

    top.box = gro.box[:] # load periodic boundary definition from .gro

    plumed_file = 'plumed_script.dat'
    sim_temp = 500.00 # simulation temperature in kelvin
    prefix = 'production_run_'
    integration_timestep = 0.002 # 2 femtoseconds
    simulation_steps = 500000000 # 1 microsecond -> increase this to run longer simulations

    NVT_steps =  500000 # 1 ns NVT -> no need to change
    NPT_steps =  1000000 # 2 ns NPT equilibration -> no need to change
    traj_frame_freq = 5000 # saves all atom coordinates as .dcd trajectory every 10 picoseconds
    colvar_save_freq = 100 # computes and saves phi,psi dihedral angles every 200 femtoseconds
    stdout_freq = 10000 # prints key system indicators every 20 picoseconds in console  -> no need to change
    ################################################################
    system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*nanometers, constraints=app.HBonds)
    integrator = LangevinMiddleIntegrator(sim_temp*kelvin, 1/picosecond, integration_timestep*picoseconds)
    simulation = Simulation(top.topology, system, integrator)
    simulation.context.setPositions(gro.positions)
    
    print("Minimizing energy")
    simulation.minimizeEnergy()
    lastpositions = simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(top.topology, lastpositions, open(directory + prefix +  'ener_minim.pdb', 'w'))
    
    print("Running NVT")
    simulation.step(NVT_steps)
    lastpositions = simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(top.topology, lastpositions, open(directory + prefix +  'NVT.pdb', 'w'))
    
    system.addForce(MonteCarloBarostat(1*bar, sim_temp*kelvin))
    simulation.context.reinitialize(preserveState=True)
    print("Running NPT")
    simulation.step(NPT_steps)
    lastpositions = simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(top.topology, lastpositions, open(directory + prefix +  'NPT.pdb', 'w'))
    
    print("System setup (energy minimization + equilibration) complete!")
    
    print("Starting production simulation!")
    simulation.reporters.append(DCDReporter(directory + prefix + '0.dcd', traj_frame_freq))
    simulation.reporters.append(StateDataReporter(stdout, stdout_freq, time = True, step=True, speed = True, potentialEnergy=True, kineticEnergy = True, temperature=True, volume=True, density=True, elapsedTime=True))
    
    lastpositions = simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(top.topology, lastpositions, open(directory + prefix + '0.pdb', 'w'))
    
    with open(directory + plumed_file) as f:
            script = f.read()
            script = script.format(colvar_save_freq, colvar_save_freq)
        
    system.addForce(openmmplumed.PlumedForce(script))
    simulation.context.reinitialize(preserveState=True)
    
    simulation.step(simulation_steps)
    
    print("Production simulation complete!")

def main():
    
    run_simulation()

if __name__ == "__main__":
    main()