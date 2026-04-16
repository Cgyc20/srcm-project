import numpy as np
from srcm import Domain, HybridState, MassProjector



def test_add_continuous_particle_mass_adds_one_unit_mass():
    dom = Domain(length=2.0, n_ssa=2, pde_multiple=4)  # h=1.0, dx=0.25
    # n_species=1
    ssa = np.zeros((1, dom.K), dtype=int)
    pde = np.zeros((1, dom.n_pde), dtype=float)
    state = HybridState(ssa, pde)
    state.assert_consistent(dom)

    # Add +1 particle to compartment 0 in PDE
    state.add_continuous_particle_mass(dom, species_idx=0, comp_idx=0, delta_particles=1)

    # Integrated mass in compartment 0 should now be 1
    proj = MassProjector(dom.pde_multiple, dom.dx, dom.h)




    mass = proj.integrate_pde_mass(pde_conc=pde)
    assert np.allclose(mass[0, 0], 1.0)
    assert np.allclose(mass[0, 1], 0.0)


def test_add_continuous_particle_mass_removes_one_unit_mass():
    dom = Domain(length=2.0, n_ssa=2, pde_multiple=4)  # h=1.0, dx=0.25
    ssa = np.zeros((1, dom.K), dtype=int)
    pde = np.zeros((1, dom.n_pde), dtype=float)
    state = HybridState(ssa, pde)

    # Add then remove
    state.add_continuous_particle_mass(dom, 0, 1, +1)
    state.add_continuous_particle_mass(dom, 0, 1, -1)
    proj = MassProjector(dom.pde_multiple, dom.dx, dom.h)
    mass = proj.integrate_pde_mass(pde_conc=pde)
    assert np.allclose(mass[0, 0], 0.0)
    assert np.allclose(mass[0, 1], 0.0)
