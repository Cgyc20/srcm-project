from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

# Type alias for propensity functions: (D, C, rates, h) -> float
# D: {"U": discrete_count, ...}
# C: {"U": pde_mass_per_compartment, ...}
# rates: dict of rate constants
# h: SSA compartment size
PropensityFn = Callable[[Dict[str, float], Dict[str, float], Dict[str, float], float], float]

@dataclass(frozen=True)
class HybridReaction:
    """
    A single hybrid reaction channel operating within one SSA compartment.

    state_change convention:
      - 'D_<species>' means SSA discrete particle change in that compartment
      - 'C_<species>' means PDE particle-mass change (in units of particles)
        Engine implements this as +/- (1/h) concentration over that compartment's PDE slice.

    IMPORTANT:
      If state_change contains any 'C_<sp>' with delta < 0, then the reaction consumes
      continuous mass and must be gated by sufficient PDE concentration/mass.
    """
    label: str
    reactants: Dict[str, int]
    products: Dict[str, int]
    propensity: PropensityFn
    state_change: Dict[str, int]
    description: Optional[str] = None

    # Derived metadata (auto-filled)
    consumes_continuous: bool = field(init=False)
    produces_continuous: bool = field(init=False)
    consumed_species: Tuple[str, ...] = field(init=False)
    produced_species: Tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        consumed = []
        produced = []

        for key, delta in self.state_change.items():
            if not isinstance(key, str) or "_" not in key:
                raise ValueError(f"{self.label}: invalid state_change key '{key}' (expected 'D_<sp>' or 'C_<sp>')")

            prefix, sp = key.split("_", 1)
            if prefix not in ("D", "C"):
                raise ValueError(f"{self.label}: unknown prefix '{prefix}' in state_change key '{key}'")

            if not isinstance(delta, int):
                raise TypeError(f"{self.label}: state_change['{key}'] must be int (got {type(delta).__name__})")

            if prefix == "C":
                if delta < 0:
                    consumed.append(sp)
                elif delta > 0:
                    produced.append(sp)

        object.__setattr__(self, "consumed_species", tuple(sorted(set(consumed))))
        object.__setattr__(self, "produced_species", tuple(sorted(set(produced))))
        object.__setattr__(self, "consumes_continuous", len(consumed) > 0)
        object.__setattr__(self, "produces_continuous", len(produced) > 0)


@dataclass
class HybridReactionSystem:
    """
    Container for species and hybrid reactions.

    This is intentionally lightweight:
    - It stores the reactions
    - It helps build local D/C dict-views for propensity evaluation
    """
    species: List[str]

    def __post_init__(self):
        if len(self.species) == 0:
            raise ValueError("species must be a non-empty list")
        if len(set(self.species)) != len(self.species):
            raise ValueError("species must be unique")
        self.species_index = {s: i for i, s in enumerate(self.species)}

        self.pure_reactions: list[dict] = []
        self.hybrid_reactions: list[HybridReaction] = []

    @property
    def n_species(self) -> int:
        return len(self.species)

    def add_reaction(self, reactants: Dict[str, int], products: Dict[str, int], rate: float):
        """
        Optional: store macroscopic reactions for reference/documentation.
        (Engine doesn't need these if you provide PDE reaction terms separately.)
        """
        self.pure_reactions.append({"reactants": reactants, "products": products, "rate": float(rate)})

    def add_reaction_original(
        self,
        reactants: Dict[str, int],
        products: Dict[str, int],
        rate: float,
        rate_name: Optional[str] = None
    ):
        """
        Add a macroscopic reaction AND automatically decompose into hybrid reactions.
        Also records which hybrid reactions came from this macroscopic reaction.
        """
        if rate_name is None:
            rate_name = f"r_{len(self.pure_reactions) + 1}"

        # record where the new hybrid reactions will start
        start_idx = len(self.hybrid_reactions)

        # store macroscopic reaction
        self.pure_reactions.append({
            "reactants": dict(reactants),
            "products": dict(products),
            "rate": float(rate),
            "rate_name": str(rate_name),
            "hybrid_slice": None,  # fill after decomposition
        })

        order = sum(reactants.values())

        if order == 0:
            self._decompose_zero_order(reactants, products, rate_name)
        elif order == 1:
            self._decompose_first_order(reactants, products, rate_name)
        elif order == 2:
            self._decompose_second_order(reactants, products, rate_name)
        else:
            raise NotImplementedError(f"Reactions of order {order} not yet supported")

        # record where they ended
        end_idx = len(self.hybrid_reactions)
        self.pure_reactions[-1]["hybrid_slice"] = (start_idx, end_idx)

        
    def _fmt_macro(self, reactants: Dict[str, int], products: Dict[str, int]) -> str:
        def side(d: Dict[str, int]) -> str:
            if not d:
                return "∅"
            parts = []
            for sp, n in d.items():
                parts.append(f"{n}{sp}" if n != 1 else f"{sp}")
            return " + ".join(parts)

        return f"{side(reactants)} → {side(products)}"


    def _decompose_zero_order(self, reactants: Dict[str, int], products: Dict[str, int], rate_name: str):
        """
        Zero-order reaction: ∅ → products
        Single hybrid reaction creating discrete products.
        Propensity: r * h (rate times compartment volume)
        """
        # Compute state change: just add products as discrete
        state_change = {}
        for species, count in products.items():
            state_change[f"D_{species}"] = count
        
        # Build hybrid reactants/products (discrete)
        hybrid_reactants = {}
        hybrid_products = {f"D_{species}": count for species, count in products.items()}
        
        # Build propensity: zero-order is r * h
        def propensity(D, C, r, h):
            return r[rate_name] * h
        
        # Build label and description
        reactant_str = "∅"
        product_str = " + ".join([f"{count}D_{sp}" if count > 1 else f"D_{sp}" 
                                for sp, count in products.items()])
        label = f"{rate_name}_zero"
        description = f"{reactant_str} → {product_str}"
        
        self.add_hybrid_reaction(
            reactants=hybrid_reactants,
            products=hybrid_products,
            propensity=propensity,
            state_change=state_change,
            label=label,
            description=description
        )


    def _decompose_first_order(self, reactants: Dict[str, int], products: Dict[str, int], rate_name: str):
        """
        First-order reaction: A → products
        Single hybrid reaction: D_A → products
        Propensity: r * D_A
        """
        # Get the single reactant species
        assert len(reactants) == 1, "First-order must have exactly one reactant species"
        species = list(reactants.keys())[0]
        reactant_count = reactants[species]
        assert reactant_count == 1, "First-order must have stoichiometry of 1"
        
        # Compute state change
        state_change = {f"D_{species}": -1}  # Consume one discrete
        for prod_sp, prod_count in products.items():
            key = f"D_{prod_sp}"
            state_change[key] = state_change.get(key, 0) + prod_count
        
        # Build hybrid reactants/products
        hybrid_reactants = {f"D_{species}": 1}
        hybrid_products = {f"D_{sp}": count for sp, count in products.items()}
        
        # Build propensity: first-order is r * D_A
        def propensity(D, C, r, h):
            return r[rate_name] * D[species]
        
        # Build label and description
        product_str = " + ".join([f"{count}D_{sp}" if count > 1 else f"D_{sp}" 
                                for sp, count in products.items()]) if products else "∅"
        label = f"{rate_name}_D"
        description = f"D_{species} → {product_str}"
        
        self.add_hybrid_reaction(
            reactants=hybrid_reactants,
            products=hybrid_products,
            propensity=propensity,
            state_change=state_change,
            label=label,
            description=description
        )


    def _decompose_second_order(self, reactants: Dict[str, int], products: Dict[str, int], rate_name: str):
        """
        Second-order reaction: A + B → products or 2A → products
        
        Generates 2-3 hybrid reactions depending on whether it's homodimerization:
        
        Homodimerization (A + A):
            1. D_A + D_A: r * D_A * (D_A - 1) / h
            2. D_A + C_A: 2 * r * D_A * C_A / h  (factor 2!)
        
        Heterodimerization (A + B):
            1. D_A + D_B: r * D_A * D_B / h
            2. D_A + C_B: r * D_A * C_B / h
            3. C_A + D_B: r * C_A * D_B / h
        """
        # Check if homodimerization
        if len(reactants) == 1:
            # Homodimerization: 2A → products
            species = list(reactants.keys())[0]
            assert reactants[species] == 2, "Homodimerization must have stoichiometry of 2"
            self._add_homodimer_reactions(species, products, rate_name)
        else:
            # Heterodimerization: A + B → products
            assert len(reactants) == 2, "Second-order must have 1 or 2 species"
            species_list = list(reactants.keys())
            assert all(reactants[sp] == 1 for sp in species_list), "Hetero must have stoichiometry 1 each"
            self._add_heterodimer_reactions(species_list[0], species_list[1], products, rate_name)


    def _add_homodimer_reactions(self, species: str, products: Dict[str, int], rate_name: str):
        """
        Add hybrid reactions for homodimerization: 2A → products

        1. D + D: r * D_A * (D_A - 1) / h,     state_change includes -2 of reactant
        2. D + C: 2 * r * D_A * C_A / h,       only D is changed by the net stoichiometric delta,
                                            C is preserved.
        """
        # --- Reaction 1: D + D (unchanged) ---
        state_change_DD = {f"D_{species}": -2}
        for prod_sp, prod_count in products.items():
            key = f"D_{prod_sp}"
            state_change_DD[key] = state_change_DD.get(key, 0) + prod_count

        hybrid_reactants_DD = {f"D_{species}": 2}
        hybrid_products_DD = {f"D_{sp}": count for sp, count in products.items()}

        def propensity_DD(D, C, r, h):
            D_val = D[species]
            return r[rate_name] * D_val * (D_val - 1) / h

        product_str = " + ".join(
            [f"{count}D_{sp}" if count > 1 else f"D_{sp}" for sp, count in products.items()]
        ) if products else "∅"

        self.add_hybrid_reaction(
            reactants=hybrid_reactants_DD,
            products=hybrid_products_DD,
            propensity=propensity_DD,
            state_change=state_change_DD,
            label=f"{rate_name}_DD",
            description=f"D_{species} + D_{species} → {product_str}"
        )

        # --- Reaction 2: D + C, preserve C, adjust only D by net stoichiometric delta ---

        # total reactant stoichiometry of this species (2A on LHS)
        reactant_stoich = 2

        # total product stoichiometry for this species (e.g. 3A on RHS)
        product_stoich = products.get(species, 0)

        # net change in number of A molecules
        delta = product_stoich - reactant_stoich  # e.g. 3 - 2 = +1 for 2A -> 3A

        # state change: only D_A is modified by delta, C_A is preserved
        state_change_DC = {f"D_{species}": delta}

        for prod_sp, prod_count in products.items():
            if prod_sp == species:
                # A on RHS: cancel against the one A that was left as continuous
                # so ignore or adjust D_A if you want *only* the net D change
                continue
            key = f"D_{prod_sp}"
            state_change_DC[key] = state_change_DC.get(key, 0) + prod_count
        # C_{species} does not change, so no entry for C_{species} in state_change_DC

        # hybrid reactants: D_A + C_A
        hybrid_reactants_DC = {f"D_{species}": 1, f"C_{species}": 1}

        # hybrid products: start from reactants, then add delta to D_A
        # D_A: 1 + delta, C_A: 1
        hybrid_products_DC = {f"D_{species}": 1 + delta, f"C_{species}": 1}

        # propensity unchanged: 2 * r * D_A * C_A / h
        def propensity_DC(D, C, r, h):
            return 2.0 * r[rate_name] * D[species] * C[species] / h

        self.add_hybrid_reaction(
            reactants=hybrid_reactants_DC,
            products=hybrid_products_DC,
            propensity=propensity_DC,
            state_change=state_change_DC,
            label=f"{rate_name}_DC",
            description = (
                    f"D_{species} + C_{species} → "
                    f"{hybrid_products_DC['D_' + species]}D_{species} + C_{species} (factor 2, preserve C)"
                )

        )


    def _add_heterodimer_reactions(self, species_A: str, species_B: str, products: Dict[str, int], rate_name: str):
        """
        Add hybrid reactions for heterodimerization: A + B → products
        
        1. D_A + D_B: r * D_A * D_B / h
        2. D_A + C_B: r * D_A * C_B / h
        3. C_A + D_B: r * C_A * D_B / h
        """
        product_str = " + ".join([f"{count}D_{sp}" if count > 1 else f"D_{sp}" 
                                for sp, count in products.items()]) if products else "∅"
        
        # --- Reaction 1: D_A + D_B ---
        state_change_DD = {f"D_{species_A}": -1, f"D_{species_B}": -1}
        for prod_sp, prod_count in products.items():
            key = f"D_{prod_sp}"
            state_change_DD[key] = state_change_DD.get(key, 0) + prod_count
        
        hybrid_reactants_DD = {f"D_{species_A}": 1, f"D_{species_B}": 1}
        hybrid_products_DD = {f"D_{sp}": count for sp, count in products.items()}
        
        def propensity_DD(D, C, r, h):
            return r[rate_name] * D[species_A] * D[species_B] / h
        
        self.add_hybrid_reaction(
            reactants=hybrid_reactants_DD,
            products=hybrid_products_DD,
            propensity=propensity_DD,
            state_change=state_change_DD,
            label=f"{rate_name}_DD",
            description=f"D_{species_A} + D_{species_B} → {product_str}"
        )

        def _state_change_for_channel(channel_reactants: Dict[str, int], products: Dict[str, int]) -> Dict[str, int]:
            # start by consuming the channel reactants
            sc: Dict[str, int] = {k: -v for k, v in channel_reactants.items()}

            # map species -> prefix used in reactants for this channel ("C" or "D")
            reactant_prefix: Dict[str, str] = {}
            for token in channel_reactants:
                prefix, sp = token.split("_", 1)
                reactant_prefix[sp] = prefix

            # add products:
            # - if species is continuous in reactants, credit back to C_species
            # - else, credit to D_species
            for sp, count in products.items():
                prefix = reactant_prefix.get(sp, "D")
                key = f"{prefix}_{sp}"
                sc[key] = sc.get(key, 0) + count

            # IMPORTANT: drop only *continuous* zeros, keep discrete zeros (tests expect D_A: 0)
            for k in list(sc.keys()):
                if k.startswith("C_") and sc[k] == 0:
                    del sc[k]

            return sc

        
        # --- Reaction 2: D_A + C_B ---
        hybrid_reactants_DC = {f"D_{species_A}": 1, f"C_{species_B}": 1}
        state_change_DC = _state_change_for_channel(hybrid_reactants_DC, products)

        # Products: preserve continuous species that are continuous reactants in this channel
        hybrid_products_DC: Dict[str, int] = {}
        for token, v in hybrid_reactants_DC.items():
            hybrid_products_DC[token] = v
        for sp, count in products.items():
            # if B is continuous reactant, keep it continuous; otherwise discrete
            if sp == species_B:
                hybrid_products_DC[f"C_{sp}"] = hybrid_products_DC.get(f"C_{sp}", 0) + count
            else:
                hybrid_products_DC[f"D_{sp}"] = hybrid_products_DC.get(f"D_{sp}", 0) + count

        def propensity_DC(D, C, r, h):
            return r[rate_name] * D[species_A] * C[species_B] / h

        self.add_hybrid_reaction(
            reactants=hybrid_reactants_DC,
            products=hybrid_products_DC,
            propensity=propensity_DC,
            state_change=state_change_DC,
            label=f"{rate_name}_DC",
            description=f"D_{species_A} + C_{species_B} → {product_str}"
        )

        # --- Reaction 3: C_A + D_B ---
        hybrid_reactants_CD = {f"C_{species_A}": 1, f"D_{species_B}": 1}
        state_change_CD = _state_change_for_channel(hybrid_reactants_CD, products)

        hybrid_products_CD: Dict[str, int] = {}
        for token, v in hybrid_reactants_CD.items():
            hybrid_products_CD[token] = v
        for sp, count in products.items():
            if sp == species_A:
                hybrid_products_CD[f"C_{sp}"] = hybrid_products_CD.get(f"C_{sp}", 0) + count
            else:
                hybrid_products_CD[f"D_{sp}"] = hybrid_products_CD.get(f"D_{sp}", 0) + count

        def propensity_CD(D, C, r, h):
            return r[rate_name] * C[species_A] * D[species_B] / h

        self.add_hybrid_reaction(
            reactants=hybrid_reactants_CD,
            products=hybrid_products_CD,
            propensity=propensity_CD,
            state_change=state_change_CD,
            label=f"{rate_name}_CD",
            description=f"C_{species_A} + D_{species_B} → {product_str}"
        )



    def add_hybrid_reaction(
        self,
        reactants: Dict[str, int],
        products: Dict[str, int],
        propensity: PropensityFn,
        state_change: Dict[str, int],
        label: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        if label is None:
            label = f"HR{len(self.hybrid_reactions) + 1}"

        # light validation: all species tokens must look like D_U or C_V etc.
        for token_dict in (reactants, products, state_change):
            for key in token_dict.keys():
                if "_" not in key:
                    raise ValueError(f"Key '{key}' must contain '_' (e.g. 'D_U', 'C_V')")
                prefix, sp = key.split("_", 1)
                if prefix not in ("D", "C"):
                    raise ValueError(f"Key '{key}' must start with 'D_' or 'C_'")
                if sp not in self.species:
                    raise ValueError(f"Unknown species '{sp}' in key '{key}'. Known: {self.species}")

        self.hybrid_reactions.append(
            HybridReaction(
                label=label,
                reactants=reactants,
                products=products,
                propensity=propensity,
                state_change=state_change,
                description=description,
            )
        )

    def local_DC_views(self, ssa_counts_compartment: Dict[str, int], pde_mass_compartment: Dict[str, float]):
        """
        Return D and C dicts in the exact form your lambdas expect.

        Example
        -------
        D = {"U": 12, "V": 3}
        C = {"U": 8.4, "V": 0.1}
        """
        return ssa_counts_compartment, pde_mass_compartment

    # def describe(self) -> None:
    #     """
    #     Pretty-print all hybrid reactions in the system.
    #     """
    #     print("\n=== Hybrid Reactions ===")
    #     for idx, hr in enumerate(self.hybrid_reactions, start=1):
    #         print(f"[{idx}] {hr.label}")
    #         print(f"     Reactants: {hr.reactants}")
    #         print(f"     Products : {hr.products}")
    #         print(f"     State Δ  : {hr.state_change}")
    #         if hr.description:
    #             print(f"     Info     : {hr.description}")
    #         print()

    def describe(self, *, include_manual_hybrids: bool = True) -> None:
        """
        Pretty-print macroscopic reactions and their decomposed hybrid reactions.

        include_manual_hybrids:
            If True, also prints any hybrid reactions that were added manually
            (i.e. not generated by add_reaction_original).
        """
        print("\n==============================")
        print(" Reaction system description")
        print("==============================")

        # Track which hybrid reactions have been "claimed" by a macroscopic reaction
        claimed = set()

        if self.pure_reactions:
            print("\n=== Macroscopic reactions ===")
            for j, pr in enumerate(self.pure_reactions, start=1):
                rxn_str = self._fmt_macro(pr.get("reactants", {}), pr.get("products", {}))
                rate = pr.get("rate", None)
                rate_name = pr.get("rate_name", None)

                header = f"[{j}] {rxn_str}"
                if rate_name is not None:
                    header += f"    (rate_name='{rate_name}')"
                if rate is not None:
                    header += f"    (rate={rate})"
                print(header)

                sl = pr.get("hybrid_slice", None)
                if sl is None:
                    print("    (no decomposition recorded)")
                    continue

                s, e = sl
                if s == e:
                    print("    (no hybrid reactions generated)")
                    continue

                print("    Decomposed hybrid reactions:")
                for idx in range(s, e):
                    claimed.add(idx)
                    hr = self.hybrid_reactions[idx]
                    print(f"      - [{idx+1}] {hr.label}")
                    print(f"           Reactants: {hr.reactants}")
                    print(f"           Products : {hr.products}")
                    print(f"           State Δ  : {hr.state_change}")
                    if hr.description:
                        print(f"           Info     : {hr.description}")

        else:
            print("\n(no macroscopic reactions stored)")

        # Optional: show hybrid reactions not linked to any macroscopic reaction
        if include_manual_hybrids:
            leftovers = [i for i in range(len(self.hybrid_reactions)) if i not in claimed]
            if leftovers:
                print("\n=== Hybrid reactions (manual / ungrouped) ===")
                for i in leftovers:
                    hr = self.hybrid_reactions[i]
                    print(f"[{i+1}] {hr.label}")
                    print(f"     Reactants: {hr.reactants}")
                    print(f"     Products : {hr.products}")
                    print(f"     State Δ  : {hr.state_change}")
                    if hr.description:
                        print(f"     Info     : {hr.description}")
                    print()
    def get_reactions_metadata(self) -> List[Dict[str, Any]]:
        """Returns the macroscopic reaction list, removing non-serializable objects like slices."""
        meta_list = []
        for rxn in self.pure_reactions:
            # We copy the dict but skip 'hybrid_slice' which isn't useful for JSON
            meta_list.append({
                "reactants": rxn.get("reactants"),
                "products": rxn.get("products"),
                "rate": rxn.get("rate"),
                "rate_name": rxn.get("rate_name")
            })
        return meta_list