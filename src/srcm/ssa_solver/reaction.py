import numpy as np
import pandas as pd

class Reaction:
    """
    Stores reactions, species, stoichiometry, and reaction types.
    Automatically calculates stoichiometric matrix and labels when reactions are added.
    Can be passed directly into an SSA simulator.
    """

    def __init__(self):
        """Initialize an empty reaction system."""
        self.reaction_set = []
        self.species_list = []
        self.stoichiometric_matrix = None
        self.stoichiometric_df = None
        self.reaction_labels = []

    def add_reaction(self, reactants: dict, products: dict, reaction_rate: float):
        """
        Add a reaction and automatically update stoichiometry.

        Parameters
        ----------
        reactants : dict
            Dictionary of reactants and their stoichiometric coefficients.
        products : dict
            Dictionary of products and their stoichiometric coefficients.
        reaction_rate : float
            Reaction rate constant.
        """
        # --- Basic validation ---
        assert isinstance(reactants, dict), "Reactants must be a dictionary"
        assert isinstance(products, dict), "Products must be a dictionary"
        assert isinstance(reaction_rate, float), "Reaction rate must be a float"

        unique_species = set(reactants.keys()).union(products.keys())
        if len(unique_species) > 2:
            raise ValueError("Only supports up to two species currently")

        # Determine reaction type
        reactants_order = sum(reactants.values())
        if reactants_order == 0:
            reaction_type = "zero_order"
        elif reactants_order == 1:
            reaction_type = "first_order"
        elif reactants_order == 2:
            reaction_type = "second_order"
        else:
            raise ValueError("Only supports up to two reactant molecules currently")

        # Store reaction
        self.reaction_set.append({
            "reactants": reactants,
            "products": products,
            "reaction_rate": reaction_rate,
            "reaction_type": reaction_type
        })

        # Automatically update stoichiometry
        self._update_stoichiometry()

    def _update_stoichiometry(self):
        """Automatically recalculate species list and stoichiometric matrix."""
        # Species
        species_set = set()
        for r in self.reaction_set:
            species_set.update(r["reactants"].keys())
            species_set.update(r["products"].keys())
        self.species_list = sorted(species_set) #THis is the alphabetically ordered list.
        self.species_index = {s:i for i,s in enumerate(self.species_list)} #This gives the species indexing dictionary
        
        self.number_of_species = len(self.species_list)
        self.number_of_reactions = len(self.reaction_set)

        # Stoichiometric matrix
        self.stoichiometric_matrix = np.zeros(
            (self.number_of_species, self.number_of_reactions), dtype=int
        )

        for j, r in enumerate(self.reaction_set):
            for s, coeff in r["reactants"].items():
                i = self.species_list.index(s)
                self.stoichiometric_matrix[i, j] -= coeff
            for s, coeff in r["products"].items():
                i = self.species_list.index(s)
                self.stoichiometric_matrix[i, j] += coeff

        # Convert all reactant/product species names in reaction_set to indices
        
         # --- Precompute species indices in each reaction for fast lookup ---
        for r in self.reaction_set:
            r["reactant_indices"] = [self.species_index[s] for s in r["reactants"].keys()]
            r["product_indices"] = [self.species_index[s] for s in r["products"].keys()]


        # Labels
        self.reaction_labels = [f"R{j+1}" for j in range(self.number_of_reactions)]

        # Convert to DataFrame for easy viewing
        self.stoichiometric_df = pd.DataFrame(
            self.stoichiometric_matrix,
            index=self.species_list,
            columns=self.reaction_labels
        )

    def print_reactions(self):
        """Print all reactions in stochastic form, showing 0 if empty."""
        for idx, r in enumerate(self.reaction_set):
            react_str = " + ".join([f"{v}{k}" if v != 1 else f"{k}" 
                                    for k, v in r["reactants"].items()]) or "0"
            prod_str = " + ".join([f"{v}{k}" if v != 1 else f"{k}" 
                                   for k, v in r["products"].items()]) or "0"
            print(f"Reaction {idx+1}: {react_str} -> {prod_str} (Rate: {r['reaction_rate']})")

    def show_stoichiometry(self):
        """Return stoichiometric matrix as a labeled DataFrame."""
        return self.stoichiometric_df
