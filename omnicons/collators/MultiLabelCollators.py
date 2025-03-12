from dataclasses import dataclass
from typing import Dict, Literal, Tuple, Union

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData

from omnicons.collators.BaseCollators import BaseCollator


@dataclass
class MultiLabelCollator(BaseCollator):
    """Fixes one-dimensional tensors in multilabel data (e.g. in multilabel graph classification
    or binary node classification).In each of these instances, a 2-d tensor of shape (1, input)
    is required to be stacked properly by the omnicons.data.DataClass.batch_from_data_list() function

    Parameters
    -----------
    apply_batch: whether to apply batching in this collator. (E.g. turned off if used in MixedCollators)

    multilabel_properties: labels for the graph mapped to whether they map to the graph or underlying nodes.
    This fixes one-dimensional tensors to allow them to be stacked by the omnicons.data.DataClass.batch_from_data_list()
    function. Tensors that are not one dimensional will be silently ignored unless they are also in labels_to_sparsify.

    labels_to_sparsify: Label names that are generated as 'dense' in the datagenerator. This function will convert a tensor
    of arbitrary length n to a tensor of shape (1, number_of_labels) with binary labels corresponding to the original 'dense'
    tensor labels. The format of the input tensor with length of n has position 0 specifying the number of classes, and
    [n+1:] enumerating the label indices found in that data point.

    labels_to_consider: Label names to process in the collator. any labels in the 'multilabel_properties' keys not in
    'labels_to_consider' will be removed from the Data (or HeteroData) object and not processed further.

    allow_data_without_all_labels: not all of 'labels_to_consider' need to be attributes of the input Data/HeteroData object.
    This is likely only applicable when multiple datasets are merged and do not share all labels.

    node_types_to_consider: node types to consider for heterogeneous graphs.

    labels_to_ignore: any extraneous labels in your data objects to be ignored in all cases
    """

    apply_batch: bool = False
    multilabel_properties: Dict[str, Literal["graph", "node"]] = None
    labels_to_sparsify: Tuple[str] = ()
    node_types_to_consider: Tuple = ()
    labels_to_consider: Tuple = ()
    labels_to_ignore: Tuple = ()
    allow_data_without_all_labels: bool = False

    def prepare_individual_data(
        self, data: Union[Data, HeteroData]
    ) -> Union[Data, HeteroData]:
        data = data.clone()
        if isinstance(data, Data):
            for prop, prop_type in self.multilabel_properties.items():
                if prop not in self.labels_to_consider:
                    delattr(data, prop)
                    continue
                for lab in self.labels_to_ignore:
                    delattr(data, lab)
                prop_tensor = getattr(data, prop, "not_here")
                if prop_tensor == "not_here":
                    if self.allow_data_without_all_labels:
                        continue
                    else:
                        raise KeyError(
                            f"Properties in 'multilabel_properties' must be in the Data object. The property '{prop}' is not found in {data.to_dict().keys()}"
                        )
                if prop_tensor.ndim == 1:
                    if prop_type == "graph":
                        setattr(data, prop, prop_tensor.view(1, -1))
                    # correct single-label (multi-class or single class) node tensors of ndim==1
                    # note this is not a fix for multilabel multiclass nodes - as these should not have a dimensionality of 1.
                    elif prop_type == "node":
                        setattr(data, prop, prop_tensor.view(-1, 1))
                if prop in self.labels_to_sparsify:
                    prop_tensor = getattr(
                        data, prop, "not_here"
                    )  # regenerate in case changed.
                    sparse_labels = multilabel_dense_to_sparse(prop_tensor)
                    setattr(data, prop, sparse_labels)
        elif isinstance(data, HeteroData):
            for node_type in data.node_types:
                if node_type not in self.node_types_to_consider:
                    continue
                for prop, prop_type in self.multilabel_properties.items():
                    if prop not in self.labels_to_consider():
                        delattr(data, prop)
                        continue
                    prop_tensor = getattr(data[node_type], prop, "not_here")
                    if prop_tensor == "not_here":
                        if self.allow_data_without_all_labels:
                            continue
                        else:
                            raise KeyError(
                                f"Properties in 'multilabel_properties' must be in the Data object. The property '{prop}' is not found in {data.to_dict().keys()}"
                            )
                    if prop_tensor.ndim == 1:
                        if prop_type == "graph":
                            setattr(
                                data[node_type], prop, prop_tensor.view(1, -1)
                            )
                        # correct single-label (multi-class or single class) node tensors of ndim==1
                        # note this is not a fix for multilabel multiclass nodes - as these should not have a dimensionality of 1.
                        elif prop_type == "node":
                            setattr(
                                data[node_type], prop, prop_tensor.view(-1, 1)
                            )
                    if prop in self.labels_to_sparsify:
                        prop_tensor = getattr(
                            data, prop, "not_here"
                        )  # regenerate in case changed.
                        sparse_labels = multilabel_dense_to_sparse(prop_tensor)
                        setattr(data[node_type], prop, sparse_labels)
        return data


# correct multilabel multiclass dense tensors to the sparse tensors required for training
def multilabel_dense_to_sparse(t=torch.tensor):
    num_classes = t[:, 0].max()
    # check tensor is built correctly
    assert torch.all(
        t[:, 0] == num_classes
    ), "All zero positions should have the same value - equal to the number of classes for that label type"
    # replace ignore index with an unused value for one-hot encoding
    # since num_labels is a count of 0-indexed values, the value num_labels is unused (assuming there are no gaps in label numbers)
    t[t == -100] = (
        num_classes  # Shape (n_nodes, maximum number of labels per node in input)
    )
    # one-hot encode t_slice to get label positions from label_ids
    oh = F.one_hot(
        t
    )  # shape (n_nodes, maximum number of labels per node in input, num_classes+1)
    # max the labels represented by each node --> gets all of the classes represented per node
    oh_max = torch.amax(oh, dim=1)  # shape (n_nodes, num_classes+1)
    # remove replaced ignore index column
    final = oh_max[:, :-1]  # shape (n_nodes, num_classes)
    assert final.size(1) == num_classes
    return final
