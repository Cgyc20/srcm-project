import numpy as np
import pytest
from srcm import MassProjector

# ---------------------------------------------------------
# 1. Existing Tests (Updated for Class API)
# ---------------------------------------------------------

def test_pde_mass_projection():
    # Setup: 3 PDE cells per SSA compartment, dx=0.1
    proj = MassProjector(pde_multiple=3, dx=0.1, h=0.3)
    
    # 1 species, 6 PDE cells (2 SSA compartments)
    pde = np.array([[1.0, 1.0, 1.0,
                     2.0, 2.0, 2.0]])

    mass = proj.integrate_pde_mass(pde)

    # Compartment 0: (1+1+1) * 0.1 = 0.3
    # Compartment 1: (2+2+2) * 0.1 = 0.6
    expected = np.array([[0.3, 0.6]])
    assert np.allclose(mass, expected)


def test_combined_mass():
    # Setup
    proj = MassProjector(pde_multiple=3, dx=0.1, h=0.3)
    ssa = np.array([[1, 2]])
    pde = np.array([[1.0, 1.0, 1.0,
                     2.0, 2.0, 2.0]])

    combined, pde_mass = proj.get_combined_total_mass(ssa, pde)

    assert np.allclose(pde_mass, [[0.3, 0.6]])
    assert np.allclose(combined, [[1.3, 2.6]])

# ---------------------------------------------------------
# 2. New Edge Case & Logic Tests
# ---------------------------------------------------------

def test_high_concentration_mask_logic():
    """
    Test that the mask only returns 1 if ALL PDE cells in a block 
    are above 1/h.
    """
    h_val = 0.5
    threshold = 1.0 / h_val  # 2.0
    proj = MassProjector(pde_multiple=2, dx=0.1, h=h_val)
    
    pde = np.array([[
        2.1, 2.5,  # Block 0: Both > 2.0 (Pass)
        1.9, 5.0,  # Block 1: One < 2.0 (Fail)
        2.0, 2.0   # Block 2: Exactly 2.0 (Pass)
    ]])
    
    mask = proj.get_high_concentration_mask(pde)
    expected = np.array([[1, 0, 1]], dtype=np.int8)
    
    assert np.array_equal(mask, expected)


def test_multi_species_broadcasting():
    """Ensure the projector handles multiple species simultaneously."""
    proj = MassProjector(pde_multiple=2, dx=1.0, h=2.0)
    
    # 2 species, 4 PDE cells each (2 compartments)
    pde = np.array([
        [1.0, 1.0, 2.0, 2.0], # Sp 0
        [0.5, 0.5, 3.0, 3.0]  # Sp 1
    ])
    
    mass = proj.integrate_pde_mass(pde)
    
    expected = np.array([
        [2.0, 4.0], # Sp 0
        [1.0, 6.0]  # Sp 1
    ])
    assert np.allclose(mass, expected)


def test_invalid_pde_shapes():
    """Ensure we raise errors if Npde is not a multiple of pde_multiple."""
    proj = MassProjector(pde_multiple=3, dx=0.1, h=0.3)
    
    # 7 cells is not divisible by 3
    pde_bad = np.ones((1, 7))
    
    with pytest.raises(ValueError, match="must be divisible by pde_multiple"):
        proj.integrate_pde_mass(pde_bad)


def test_mass_projector_validation():
    """Test that the projector itself prevents invalid geometry."""
    with pytest.raises(ValueError):
        MassProjector(pde_multiple=0, dx=0.1, h=0.1)
    with pytest.raises(ValueError):
        MassProjector(pde_multiple=2, dx=-0.1, h=0.1)


def test_ssa_pde_mismatch():
    """Test combined mass error when SSA and PDE grid widths don't match."""
    proj = MassProjector(pde_multiple=2, dx=0.1, h=0.2)
    
    ssa = np.ones((1, 5))    # 5 compartments
    pde = np.ones((1, 12))   # 12 cells / 2 = 6 compartments
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        proj.get_combined_total_mass(ssa, pde)