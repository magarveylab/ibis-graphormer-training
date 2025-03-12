import os
import random
from collections import Counter
from typing import List

import torch
from sklearn.utils.class_weight import compute_class_weight


def create_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def batchify(l: list, bs=500):  # group calls for database
    return [l[x : x + bs] for x in range(0, len(l), bs)]


def split(a: list, n: int):
    k, m = divmod(len(a), n)
    return [
        a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
    ]


def shuffle(l):
    l = sorted(l)
    random.shuffle(l)
    return l


def most_common(l):
    return max(set(l), key=l.count)


def get_single_label_class_weight(
    datapoints: List[int], ignore_id: int = -100
) -> List[float]:
    # not datapoints are already tokenized
    datapoints = [c for c in datapoints if c != ignore_id]
    classes = sorted(set(datapoints))
    w_calc = compute_class_weight("balanced", classes=classes, y=datapoints)
    # create a template class size vector based on label tokenizer length
    weights = [0] * (max(classes) + 1)
    for idx, w in zip(classes, w_calc):
        weights[idx] = w
    return list(weights)


def get_multi_label_class_weight(
    datapoints: List[List[int]], ignore_id: int = -100
) -> List[float]:
    class_len = len(datapoints[0])
    weights = [0] * class_len
    for idx in range(class_len):
        sample = [r[idx] for r in datapoints if r[idx] != ignore_id]
        try:
            w_calc = compute_class_weight("balanced", classes=[0, 1], y=sample)
            weights[idx] = w_calc[1]
        # if not in dataset assign weight of zero.
        except ValueError:
            weights[idx] == 0
    return weights


class NodeMultiLabelDict:
    def __init__(
        self,
        ignore_index: int = -100,
        is_dense: bool = False,
        num_classes: int = None,
    ):
        self.ignore_index = ignore_index
        self.label_list = None
        self.is_dense = is_dense
        self.num_classes = num_classes

    def _update_sparse(self, input_tens: torch.tensor):
        # transpose so you are iterating over the label dimension - this will mean that each
        # iteration will have all of the node labels for the given label position
        t_input = input_tens.T.tolist()
        if self.label_list is None:
            self.num_classes = (
                len(t_input) if self.num_classes is None else self.num_classes
            )
            self.label_list = [
                {0: 0, 1: 0, -100: 0} for i in range(self.num_classes + 1)
            ]
        for label_idx, node_labels in enumerate(t_input):
            inp_dict = dict(Counter(node_labels))
            for node_label, count in inp_dict.items():
                self.label_list[label_idx][node_label] += count

    def _update_dense(self, input_tens: torch.tensor):
        clean_input = input_tens[:, 1:]
        class_count = input_tens[:, 0]
        if self.num_classes is None:
            self.num_classes = int(input_tens[0][0])
        assert torch.all(class_count == self.num_classes)
        if self.label_list is None:
            self.label_list = [
                {0: 0, 1: 0, -100: 0} for i in range(self.num_classes)
            ]
        for node_labels in clean_input:
            # input is fully padded with -100, therefore node has no labels for this category.
            if torch.all(node_labels == self.ignore_index):
                continue
            else:
                valid_labels = set(
                    [
                        int(x)
                        for x in list(
                            node_labels[
                                torch.where(node_labels != self.ignore_index)
                            ]
                        )
                    ]
                )
                all_labels = set([x for x in range(self.num_classes)])
                not_labelled = all_labels - valid_labels
                # if valid labels exist, increase the positive count of these labels.
                for label_idx in valid_labels:
                    self.label_list[label_idx][1] += 1
                # for all other labels, increase the number of negative counts
                for label_idx in not_labelled:
                    self.label_list[label_idx][0] += 1

    def update(self, input_tens: torch.tensor):
        """By default, takes 2d tensor of shape (num_nodes, num_labels) with 0 & 1 for the
        absence or presence of the label, respectively. If 'is_dense' is true, the input is
        assumed to be (num_nodes, x) where x is a dense array of the label indices that are
        present, padded with -100 if necessary. note that the first position of this array
        indicates the number of labels in that class"""
        if self.is_dense:
            self._update_dense(input_tens=input_tens)
        else:
            self._update_sparse(input_tens=input_tens)

    def get_class_weight(self):
        weights = [0] * self.num_classes
        for idx, label_dict in enumerate(self.label_list):
            num_zeros = label_dict.get(0)
            num_ones = label_dict.get(1)
            sample = []
            if num_zeros is not None:
                sample += [0] * num_zeros
            if num_ones is not None:
                sample += [1] * num_ones
            if sample == [] or set(sample) == set([0]):
                weights[idx] = 0
            else:
                w_calc = compute_class_weight(
                    "balanced", classes=[0, 1], y=sample
                )
                weights[idx] = w_calc[1]
        self.weights = weights
        return weights


class NodeSingleLabelDict:
    def __init__(self, ignore_index: int = -100):
        self.ignore_index = ignore_index
        self.label_dict = {}

    def update(self, input_tens: torch.tensor):
        """takes 1D tensor of shape (num_nodes)"""
        # number of classes cannot necessarily be inferred directly from input
        node_classes = input_tens.tolist()
        inp_dict = dict(Counter(node_classes))
        for node_class, class_count in inp_dict.items():
            if self.label_dict.get(node_class) is None:
                self.label_dict[node_class] = 0
            self.label_dict[node_class] += class_count

    def get_class_weight(
        self, fill_weights: bool = False, num_classes: int = None
    ):
        # classes are all labels except the ignore index.
        classes = sorted(
            set(self.label_dict.keys()) - set([self.ignore_index])
        )
        # num_classes should always be able to be inferred - but can be externally overriden if not all classes may be in the dataset
        num_classes = (
            (max(classes) + 1) if num_classes is None else num_classes
        )
        # add any missing classes at the minimum weight of 1. (For testing with a subset, should not be used in production)
        self.added_classes = set()
        if fill_weights:
            for i in range(num_classes):
                if (
                    self.label_dict.get(i) is None
                    or self.label_dict.get(i) == 0
                ):
                    self.label_dict[i] = 1
                    self.added_classes.add(i)
            classes = sorted(
                set(self.label_dict.keys()) - set([self.ignore_index])
            )
        all_labels = []
        for node_class, class_count in self.label_dict.items():
            if node_class == self.ignore_index:
                continue
            all_labels.extend([node_class] * class_count)
        w_calc = compute_class_weight(
            "balanced", classes=classes, y=all_labels
        )
        weights = [1] * num_classes
        for idx, w in zip(classes, w_calc):
            weights[idx] = w
        self.weights = list(weights)
        return list(weights)


class GraphMultiLabelDict:
    def __init__(self, ignore_index: int = -100, is_dense: bool = False):
        self.ignore_index = ignore_index
        self.label_list = None
        self.is_dense = is_dense

    def _update_dense(self, input_tens: torch.tensor):
        clean_input = input_tens[1:]
        if self.label_list is None:
            self.num_classes = int(input_tens[0])
            self.label_list = [
                {0: 0, 1: 0, -100: 0} for i in range(self.num_classes)
            ]
        if torch.all(clean_input == -100):
            pass
        else:
            valid_labels = set(
                [
                    int(x)
                    for x in list(
                        clean_input[
                            torch.where(clean_input != self.ignore_index)
                        ]
                    )
                ]
            )
            all_labels = set([x for x in range(self.num_classes)])
            not_labelled = all_labels - valid_labels
            # if valid labels exist, increase the positive count of these labels.
            for label_idx in valid_labels:
                self.label_list[label_idx][1] += 1
            # for all other labels, increase the number of negative counts
            for label_idx in not_labelled:
                self.label_list[label_idx][0] += 1

    def _update_sparse(self, input_tens: torch.tensor):
        tens_list = input_tens.tolist()
        if self.label_list is None:
            self.num_classes = len(tens_list)
            self.label_list = [
                {0: 0, 1: 0, -100: 0} for i in range(self.num_classes)
            ]
        for label_idx, graph_label in enumerate(tens_list):
            self.label_list[label_idx][graph_label] += 1

    def update(self, input_tens: torch.tensor):
        if self.is_dense:
            self._update_dense(input_tens=input_tens)
        else:
            self._update_sparse(input_tens=input_tens)

    def get_class_weight(self):
        weights = [0] * self.num_classes
        for idx, label_dict in enumerate(self.label_list):
            num_zeros = label_dict.get(0)
            num_ones = label_dict.get(1)
            sample = []
            if num_zeros is not None:
                sample += [0] * num_zeros
            if num_ones is not None:
                sample += [1] * num_ones
            if sample == [] or set(sample) == set([0]):
                weights[idx] = 0
            else:
                w_calc = compute_class_weight(
                    "balanced", classes=[0, 1], y=sample
                )
                weights[idx] = w_calc[1]
        self.weights = weights
        return weights
