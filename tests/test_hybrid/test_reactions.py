import numpy as np
from srcm import HybridReactionSystem


def test_add_hybrid_reaction_and_labeling():
    rs = HybridReactionSystem(species=["U", "V"])

    rs.add_hybrid_reaction(
        reactants={"D_U": 2},
        products={"D_U": 3},
        propensity=lambda D, C, r, h: r["k"] * D["U"] * (D["U"] - 1) / h,
        state_change={"D_U": +1},
        label="R1_1"
    )

    assert len(rs.hybrid_reactions) == 1
    assert rs.hybrid_reactions[0].label == "R1_1"
    assert rs.hybrid_reactions[0].state_change["D_U"] == 1


def test_add_hybrid_reaction_validates_species():
    import pytest
    rs = HybridReactionSystem(species=["U", "V"])

    with pytest.raises(ValueError):
        rs.add_hybrid_reaction(
            reactants={"D_W": 1},  # W is not a known species
            products={},
            propensity=lambda D, C, r, h: 0.0,
            state_change={"D_W": -1},
        )
