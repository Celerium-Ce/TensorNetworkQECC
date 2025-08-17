import quimb as qu
import quimb.tensor as qtn
import numpy as np

class TensorNetworkCode:
    def fuse_and_contract_indices(self, ind1, ind2):
        """
        Fuse two indices (legs) by renaming all occurrences of ind2 to ind1, then contract over the new shared index.

        Args:
            ind1 (str): The name of the first index (leg) to keep.
            ind2 (str): The name of the second index (leg) to rename to ind1.

        Returns:
            qtn.TensorNetwork: The resulting tensor network after fusing and contracting the indices.

        Example:
            contracted_tn = code.fuse_and_contract_indices('a', 'd')
        """
        tn = self.network.copy()
        # Rename all occurrences of ind2 to ind1
        for t in tn.tensors:
            t.reindex_({ind2: ind1})
        # Now contract over the fused index
        if ind1 not in tn.ind_map:
            raise ValueError(f"Index '{ind1}' not found in the tensor network after fusing.")
        tn.contract_ind(ind1)
        return tn
    def contract_two_indices(self, ind1, ind2):
        """
        Contract the network by merging the two specified indices (legs) into a single contraction step.

        Args:
            ind1 (str): The name of the first index (leg) to contract.
            ind2 (str): The name of the second index (leg) to contract.

        Returns:
            qtn.TensorNetwork: The resulting tensor network after contracting the two specified indices.

        Note:
            This contracts ind1 and ind2 sequentially. If both are present in a single tensor, they will be contracted together.

        Example:
            contracted_tn = code.contract_two_indices('a', 'b')
        """
        tn = self.network.copy()
        if ind1 not in tn.ind_map:
            raise ValueError(f"Index '{ind1}' not found in the tensor network.")
        if ind2 not in tn.ind_map:
            raise ValueError(f"Index '{ind2}' not found in the tensor network.")
        tn.contract_ind(ind1)
        tn.contract_ind(ind2)
        return tn
    """
    A basic framework for a quantum error-correcting code using tensor networks.

    Indices (legs):
        In this context, an 'index' (or 'leg') is a string label assigned to a dimension of a tensor in the network.
        When two or more tensors share the same index name, those tensors are connected along that leg.
        Contracting an index means summing over that shared dimension, effectively merging the connected tensors along that leg.
        Indices are specified as strings, e.g., 'a', 'b', etc.
    """
    def __init__(self, tensors, network_structure=None):
        """
        tensors: list of qtn.Tensor or dict of named tensors
        network_structure: optional, e.g. adjacency info or graph
        """
        self.tensors = tensors
        self.network = qtn.TensorNetwork(list(tensors.values()) if isinstance(tensors, dict) else tensors)
        self.structure = network_structure

    def contract(self, **kwargs):
        """Contract the full tensor network."""
        return self.network.contract(**kwargs)

    def contract_indices(self, indices):
        """
        Contract the network over the specified indices (legs).

        Args:
            indices (list or tuple of str): The names of the indices (legs) to contract over. Each index must be a string label shared by two or more tensors in the network.

        Returns:
            qtn.TensorNetwork: The resulting tensor network after contracting the specified indices.

        Example:
            # Contract only over index 'b'
            contracted_tn = code.contract_indices(['b'])
        """
        tn = self.network.copy()
        for ind in indices:
            if ind not in tn.ind_map:
                raise ValueError(f"Index '{ind}' not found in the tensor network.")
            tn.contract_ind(ind)
        return tn

    def add_tensor(self, tensor):
        """Add a tensor to the network."""
        self.network.add_tensor(tensor)

    def get_tensors(self):
        return self.network.tensors

    def draw(self, **kwargs):
        """Visualize the tensor network."""
        return self.network.draw(**kwargs)

# Example usage (to be removed or moved to tests/notebooks):
t1 = qtn.Tensor(np.random.rand(2,2), inds=('a','b'))
t3 = qtn.Tensor(np.random.rand(2,2), inds=('e','f'))

t2 = qtn.Tensor(np.random.rand(2,2), inds=('d','c'))
code = TensorNetworkCode([t1, t2,t3])
# Draw the initial network
code.network.draw(
    show_tags=True,         # Show tensor tags
    color=['T1', 'T2'],     # Color by tags
    legend=True,            # Show legend
    layout='kamada_kawai',  # Use a nice layout
    show_inds=True,         # Show index names on legs
    figsize=(6, 4)          # Set figure size
)
# Fuse legs 'a' and 'd', then contract over the new shared leg
contracted_fused = code.fuse_and_contract_indices('a', 'd')
contracted_fused.draw(
    show_tags=True,         # Show tensor tags
    color=['T1', 'T2'],     # Color by tags
    legend=True,            # Show legend
    layout='kamada_kawai',  # Use a nice layout
    show_inds=True,         # Show index names on legs
    figsize=(6, 4)          # Set figure size
)
# Contract the entire network (all remaining indices)
result = code.contract()

# Assign tags to tensors for labeling
t1 = qtn.Tensor(np.random.rand(2,2), inds=('a','b'), tags={'T1'})
t2 = qtn.Tensor(np.random.rand(2,2), inds=('d','c'), tags={'T2'})
code = TensorNetworkCode([t1, t2])

# Draw with tags and custom options
code.network.draw(
    show_tags=True,         # Show tensor tags
    color=['T1', 'T2'],     # Color by tags
    legend=True,            # Show legend
    layout='kamada_kawai',  # Use a nice layout
    show_inds=True,         # Show index names on legs
    figsize=(6, 4)          # Set figure size
)
