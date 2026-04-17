import numpy as np
import pytest

from srcm import Domain, ConversionParams
from srcm import HybridReactionSystem
from srcm import SRCMEngine
from srcm import HybridState
from srcm import MassProjector


def make_engine(domain, cd_removal_mode="standard"):
    conversion = ConversionParams(
        DC_threshold=5.0,
        CD_threshold=3.0,
        rate=2.0,
    )

    reactions = HybridReactionSystem(species=["U", "V"])
    reactions.add_hybrid_reaction(
        reactants={"D_U": 1},
        products={"D_U": 1},
        propensity=lambda D, C, r, h: 0.0,
        state_change={"D_U": 0},
        label="dummy",
    )

    def pde_terms(C, rates):
        return np.zeros_like(C)

    engine = SRCMEngine(
        reactions=reactions,
        pde_reaction_terms=pde_terms,
        diffusion_rates={"U": 0.1, "V": 0.2},
        domain=domain,
        conversion=conversion,
        reaction_rates={},
        cd_removal_mode=cd_removal_mode,
    )
    return engine, conversion


def make_mass_projector(domain):
    return MassProjector(
        pde_multiple=domain.pde_multiple,
        dx=domain.dx,
        h=domain.h,
    )


def cd_index_for_species_comp(domain, species_idx, n_species):
    block = n_species + species_idx
    return block * domain.K


def dc_index_for_species_comp(domain, species_idx, n_species):
    block = 2 * n_species + species_idx
    return block * domain.K


# ---------------------------------------------------------------------
# Propensity tests
# ---------------------------------------------------------------------

def test_engine_build_propensity_vector_shape():
    domain = Domain(length=1.0, n_ssa=4, pde_multiple=2, boundary="periodic")
    engine, conversion = make_engine(domain)
    mass_proj = make_mass_projector(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)
    state = HybridState(ssa=ssa, pde=pde)
    state.assert_consistent(domain)

    comb, pde_mass = mass_proj.get_combined_total_mass(
        ssa_counts=state.ssa,
        pde_conc=state.pde,
    )
    DC_mask = conversion.DC_mask(comb)
    CD_mask = conversion.CD_mask(comb)
    sufficient = mass_proj.get_high_concentration_mask(pde_conc=state.pde)

    a = engine.build_propensity_vector(
        state=state,
        pde_mass=pde_mass,
        DC_mask=DC_mask,
        CD_mask=CD_mask,
        sufficient_mask=sufficient,
        out=None,
    )

    n_species = 2
    n_hybrid = 1
    expected_len = (3 * n_species + n_hybrid) * domain.K
    assert a.shape == (expected_len,)
    assert np.all(a >= 0.0)


def test_cd_propensity_standard_requires_sufficient_mask():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, conversion = make_engine(domain, cd_removal_mode="standard")
    mass_proj = make_mass_projector(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # U species
    # compartment 0: mass = (1+1)*0.25 = 0.5 < CD threshold
    # but concentrations [1,1], and h=0.5 so 1/h = 2 => NOT sufficient
    # compartment 1: mass = (10+10)*0.25 = 5.0 > CD threshold
    pde[0, 0:2] = np.array([1.0, 1.0])
    pde[0, 2:4] = np.array([10.0, 10.0])

    state = HybridState(ssa=ssa, pde=pde)

    comb, pde_mass = mass_proj.get_combined_total_mass(
        ssa_counts=state.ssa,
        pde_conc=state.pde,
    )
    DC_mask = conversion.DC_mask(comb)
    CD_mask = conversion.CD_mask(comb)
    sufficient = mass_proj.get_high_concentration_mask(pde_conc=state.pde)

    a = engine.build_propensity_vector(
        state=state,
        pde_mass=pde_mass,
        DC_mask=DC_mask,
        CD_mask=CD_mask,
        sufficient_mask=sufficient,
    )

    n_species = 2
    K = domain.K
    block = n_species + 0
    got = a[block * K:(block + 1) * K]

    assert np.allclose(got, np.array([0.0, 0.0]))


def test_cd_propensity_probabilistic_ignores_sufficient_mask():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, conversion = make_engine(domain, cd_removal_mode="probabilistic")
    mass_proj = make_mass_projector(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # same setup as previous test
    pde[0, 0:2] = np.array([1.0, 1.0])      # mass 0.5, CD allowed
    pde[0, 2:4] = np.array([10.0, 10.0])    # mass 5.0, CD not allowed

    state = HybridState(ssa=ssa, pde=pde)

    comb, pde_mass = mass_proj.get_combined_total_mass(
        ssa_counts=state.ssa,
        pde_conc=state.pde,
    )
    DC_mask = conversion.DC_mask(comb)
    CD_mask = conversion.CD_mask(comb)
    sufficient = mass_proj.get_high_concentration_mask(pde_conc=state.pde)

    a = engine.build_propensity_vector(
        state=state,
        pde_mass=pde_mass,
        DC_mask=DC_mask,
        CD_mask=CD_mask,
        sufficient_mask=sufficient,
    )

    n_species = 2
    K = domain.K
    block = n_species + 0
    got = a[block * K:(block + 1) * K]

    assert np.allclose(got, np.array([1.0, 0.0]))


def test_dc_propensity_uses_DC_mask_and_ssa_count():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, conversion = make_engine(domain)
    mass_proj = make_mass_projector(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    ssa[0, 0] = 4
    ssa[0, 1] = 7

    # comp 0 combined mass = 4 + 0.5 = 4.5
    # comp 1 combined mass = 7 + 0.0 = 7.0
    pde[0, 0:2] = np.array([1.0, 1.0])

    state = HybridState(ssa=ssa, pde=pde)

    comb, pde_mass = mass_proj.get_combined_total_mass(
        ssa_counts=state.ssa,
        pde_conc=state.pde,
    )
    DC_mask = conversion.DC_mask(comb)
    CD_mask = conversion.CD_mask(comb)
    sufficient = mass_proj.get_high_concentration_mask(pde_conc=state.pde)

    a = engine.build_propensity_vector(
        state=state,
        pde_mass=pde_mass,
        DC_mask=DC_mask,
        CD_mask=CD_mask,
        sufficient_mask=sufficient,
    )

    n_species = 2
    K = domain.K
    block = 2 * n_species + 0
    got = a[block * K:(block + 1) * K]

    assert np.allclose(got, np.array([0.0, 14.0]))


def test_hysteresis_deadband_gives_zero_cd_and_dc_propensities():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, conversion = make_engine(domain)
    mass_proj = make_mass_projector(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # combined mass = 4 in both compartments
    ssa[0, :] = np.array([4, 4])

    state = HybridState(ssa=ssa, pde=pde)

    comb, pde_mass = mass_proj.get_combined_total_mass(
        ssa_counts=state.ssa,
        pde_conc=state.pde,
    )
    DC_mask = conversion.DC_mask(comb)
    CD_mask = conversion.CD_mask(comb)
    sufficient = mass_proj.get_high_concentration_mask(pde_conc=state.pde)

    a = engine.build_propensity_vector(
        state=state,
        pde_mass=pde_mass,
        DC_mask=DC_mask,
        CD_mask=CD_mask,
        sufficient_mask=sufficient,
    )

    n_species = 2
    K = domain.K

    cd_block = n_species + 0
    dc_block = 2 * n_species + 0

    cd_vals = a[cd_block * K:(cd_block + 1) * K]
    dc_vals = a[dc_block * K:(dc_block + 1) * K]

    assert np.allclose(cd_vals, 0.0)
    assert np.allclose(dc_vals, 0.0)


def test_negative_pde_mass_does_not_create_negative_cd_propensity():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, conversion = make_engine(domain, cd_removal_mode="probabilistic")
    mass_proj = make_mass_projector(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)
    pde[0, 0:2] = np.array([-1.0, -1.0])

    state = HybridState(ssa=ssa, pde=pde)

    comb, pde_mass = mass_proj.get_combined_total_mass(
        ssa_counts=state.ssa,
        pde_conc=state.pde,
    )
    DC_mask = conversion.DC_mask(comb)
    CD_mask = conversion.CD_mask(comb)
    sufficient = mass_proj.get_high_concentration_mask(pde_conc=state.pde)

    a = engine.build_propensity_vector(
        state=state,
        pde_mass=pde_mass,
        DC_mask=DC_mask,
        CD_mask=CD_mask,
        sufficient_mask=sufficient,
    )

    n_species = 2
    K = domain.K
    cd_block = n_species + 0
    cd_vals = a[cd_block * K:(cd_block + 1) * K]

    assert np.all(cd_vals >= 0.0)


def test_cd_removal_mode_validation():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    conversion = ConversionParams(
        DC_threshold=5.0,
        CD_threshold=3.0,
        rate=2.0,
    )
    reactions = HybridReactionSystem(species=["U", "V"])

    def pde_terms(C, rates):
        return np.zeros_like(C)

    with pytest.raises(ValueError, match="cd_removal_mode"):
        SRCMEngine(
            reactions=reactions,
            pde_reaction_terms=pde_terms,
            diffusion_rates={"U": 0.1, "V": 0.2},
            domain=domain,
            conversion=conversion,
            reaction_rates={},
            cd_removal_mode="banana",
        )


# ---------------------------------------------------------------------
# Event application tests: STANDARD / BOOLEAN removal
# ---------------------------------------------------------------------

def test_apply_event_cd_standard_removes_one_particle_mass_and_adds_one_discrete():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, _ = make_engine(domain, cd_removal_mode="standard")

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # U, compartment 0 slice
    pde[0, 0:2] = np.array([4.0, 4.0])

    state = HybridState(ssa=ssa, pde=pde)
    idx = cd_index_for_species_comp(domain, species_idx=0, n_species=2)

    before_mass = np.sum(state.pde[0, 0:2]) * domain.dx
    before_count = state.ssa[0, 0]

    engine.apply_event(
        idx=idx,
        state=state,
        rng=np.random.default_rng(123),
        pde_mass=np.zeros((2, domain.K)),
    )

    after_mass = np.sum(state.pde[0, 0:2]) * domain.dx
    after_count = state.ssa[0, 0]

    assert after_count == before_count + 1
    assert before_mass - after_mass == pytest.approx(1.0)


def test_apply_event_cd_standard_does_not_zero_subparticle_mass():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, _ = make_engine(domain, cd_removal_mode="standard")

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # deliberately subparticle total mass: (0.4 + 0.4) * 0.25 = 0.2
    pde[0, 0:2] = np.array([0.4, 0.4])

    state = HybridState(ssa=ssa, pde=pde.copy())
    idx = cd_index_for_species_comp(domain, species_idx=0, n_species=2)

    engine.apply_event(
        idx=idx,
        state=state,
        rng=np.random.default_rng(123),
        pde_mass=np.zeros((2, domain.K)),
    )

    # standard mode uses the original boolean removal path, not the probabilistic clear-all path
    assert state.ssa[0, 0] == 1
    assert not np.allclose(state.pde[0, 0:2], 0.0)


# ---------------------------------------------------------------------
# Event application tests: PROBABILISTIC removal
# ---------------------------------------------------------------------

class FixedRNG:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def test_apply_event_cd_probabilistic_subparticle_adds_particle_when_rng_below_Yk():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, _ = make_engine(domain, cd_removal_mode="probabilistic")

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # Y_k = (0.4 + 0.4) * 0.25 = 0.2
    pde[0, 0:2] = np.array([0.4, 0.4])

    state = HybridState(ssa=ssa, pde=pde)
    idx = cd_index_for_species_comp(domain, species_idx=0, n_species=2)

    engine.apply_event(
        idx=idx,
        state=state,
        rng=FixedRNG(0.1),
        pde_mass=np.zeros((2, domain.K)),
    )

    assert state.ssa[0, 0] == 1
    assert np.allclose(state.pde[0, 0:2], 0.0)


def test_apply_event_cd_probabilistic_subparticle_does_not_add_particle_when_rng_above_Yk():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, _ = make_engine(domain, cd_removal_mode="probabilistic")

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # Y_k = 0.2
    pde[0, 0:2] = np.array([0.4, 0.4])

    state = HybridState(ssa=ssa, pde=pde)
    idx = cd_index_for_species_comp(domain, species_idx=0, n_species=2)

    engine.apply_event(
        idx=idx,
        state=state,
        rng=FixedRNG(0.9),
        pde_mass=np.zeros((2, domain.K)),
    )

    assert state.ssa[0, 0] == 0
    assert np.allclose(state.pde[0, 0:2], 0.0)


def test_apply_event_cd_probabilistic_uniform_branch_removes_exactly_one_particle_mass():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, _ = make_engine(domain, cd_removal_mode="probabilistic")

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # h = length / K = 0.5, so drop = 1/h = 2.0
    # both cells >= 2.0 so uniform branch should fire
    pde[0, 0:2] = np.array([3.0, 3.0])

    state = HybridState(ssa=ssa, pde=pde)
    idx = cd_index_for_species_comp(domain, species_idx=0, n_species=2)

    before_mass = np.sum(state.pde[0, 0:2]) * domain.dx

    engine.apply_event(
        idx=idx,
        state=state,
        rng=np.random.default_rng(123),
        pde_mass=np.zeros((2, domain.K)),
    )

    after_mass = np.sum(state.pde[0, 0:2]) * domain.dx

    assert state.ssa[0, 0] == 1
    assert before_mass - after_mass == pytest.approx(1.0)


def test_apply_event_cd_probabilistic_fallback_branch_removes_exactly_one_particle_mass():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, _ = make_engine(domain, cd_removal_mode="probabilistic")

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # h = 0.5 => drop = 2.0
    # choose slice so total mass >= 1 but not all cells >= 2
    # mass = (3.0 + 1.5) * 0.25 = 1.125
    pde[0, 0:2] = np.array([3.0, 1.5])

    state = HybridState(ssa=ssa, pde=pde)
    idx = cd_index_for_species_comp(domain, species_idx=0, n_species=2)

    before_mass = np.sum(state.pde[0, 0:2]) * domain.dx

    engine.apply_event(
        idx=idx,
        state=state,
        rng=np.random.default_rng(123),
        pde_mass=np.zeros((2, domain.K)),
    )

    after_mass = np.sum(state.pde[0, 0:2]) * domain.dx

    assert state.ssa[0, 0] == 1
    assert before_mass - after_mass == pytest.approx(1.0)


def test_standard_and_probabilistic_modes_behave_differently_for_subparticle_case():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")

    engine_std, _ = make_engine(domain, cd_removal_mode="standard")
    engine_prob, _ = make_engine(domain, cd_removal_mode="probabilistic")

    ssa_std = np.zeros((2, domain.K), dtype=int)
    pde_std = np.zeros((2, domain.n_pde), dtype=float)
    pde_std[0, 0:2] = np.array([0.4, 0.4])

    ssa_prob = np.zeros((2, domain.K), dtype=int)
    pde_prob = np.zeros((2, domain.n_pde), dtype=float)
    pde_prob[0, 0:2] = np.array([0.4, 0.4])

    state_std = HybridState(ssa=ssa_std, pde=pde_std)
    state_prob = HybridState(ssa=ssa_prob, pde=pde_prob)

    idx = cd_index_for_species_comp(domain, species_idx=0, n_species=2)

    engine_std.apply_event(
        idx=idx,
        state=state_std,
        rng=np.random.default_rng(123),
        pde_mass=np.zeros((2, domain.K)),
    )

    engine_prob.apply_event(
        idx=idx,
        state=state_prob,
        rng=FixedRNG(0.9),  # no particle created, but PDE should still be cleared
        pde_mass=np.zeros((2, domain.K)),
    )

    # Standard mode should not zero the PDE slice in this case
    assert not np.allclose(state_std.pde[0, 0:2], 0.0)

    # Probabilistic mode should clear the PDE slice
    assert np.allclose(state_prob.pde[0, 0:2], 0.0)


# ---------------------------------------------------------------------
# DC + bounds smoke tests
# ---------------------------------------------------------------------

def test_apply_event_dc_moves_one_discrete_particle_to_pde():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, _ = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    ssa[0, 0] = 2

    state = HybridState(ssa=ssa, pde=pde)
    idx = dc_index_for_species_comp(domain, species_idx=0, n_species=2)

    before_mass = np.sum(state.pde[0, 0:2]) * domain.dx

    engine.apply_event(
        idx=idx,
        state=state,
        rng=np.random.default_rng(123),
        pde_mass=np.zeros((2, domain.K)),
    )

    after_mass = np.sum(state.pde[0, 0:2]) * domain.dx

    assert state.ssa[0, 0] == 1
    assert after_mass - before_mass == pytest.approx(1.0)


def test_apply_event_negative_idx_is_noop():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, _ = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)
    ssa[0, 0] = 1

    state = HybridState(ssa=ssa.copy(), pde=pde.copy())

    engine.apply_event(
        idx=-1,
        state=state,
        rng=np.random.default_rng(123),
        pde_mass=np.zeros((2, domain.K)),
    )

    assert np.array_equal(state.ssa, ssa)
    assert np.array_equal(state.pde, pde)