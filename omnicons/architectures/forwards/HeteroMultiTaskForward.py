from typing import Union

import torch
from torch import nn
from torch.nn import ParameterDict

from omnicons.models import DataStructs


class HeteroMultiTaskForward(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self, predict: bool = False, loss_can_be_zero: bool = False, **kwargs
    ) -> DataStructs.MultiTaskOutput:
        # variables to track
        final_loss = 0
        lookup_dict = {}
        output_dict = {}
        logits_dict = {}
        labels_dict = {}
        for inp in self.inputs:
            lookup_dict[inp], output_dict[inp] = self.get_model_outputs(
                kwargs[inp]
            )
        # multi task processing
        for head_name, head in self.heads.items():
            # single input classification
            if head.training_task in [
                "graph_classification",
                "node_classification",
                "node_regression",
                "edge_classification",
            ]:
                # classification tasks
                for inp in head.analyze_inputs:
                    batch = kwargs[inp]
                    if batch == None:
                        continue
                    if head.training_task == "graph_classification":
                        # name to access predictions
                        cache_name = f"{head_name}___{inp}"
                        # ground truth
                        labels = getattr(batch, head_name)
                        # input features
                        feats = output_dict[inp].pooled_output
                        # prediction
                        if predict == False:
                            out = head.classification(feats, labels)
                            final_loss += out["loss"]
                            logits_dict[cache_name] = out["logits"]
                            labels_dict[cache_name] = out["labels"]
                        else:
                            logits_dict[cache_name] = head(feats)
                    elif head.training_task == "node_classification":
                        head_node_type = lookup_dict[inp][head.node_type]
                        # name to access predictions
                        cache_name = f"{head_name}___{inp}"
                        # ground truth
                        labels = output_dict[inp][head_node_type][head_name]
                        # input features
                        feats = output_dict[inp][head_node_type]["x"]
                        # prediction
                        if predict == False:
                            out = head.classification(feats, labels)
                            final_loss += out["loss"]
                            logits_dict[cache_name] = out["logits"]
                            labels_dict[cache_name] = out["labels"]
                        else:
                            logits_dict[cache_name] = head(feats)
                    elif head.training_task == "node_regression":
                        head_node_type = lookup_dict[inp][head.node_type]
                        # name to access predictions
                        cache_name = f"{head_name}___{inp}"
                        # ground truth
                        labels = output_dict[inp][head_node_type][head_name]
                        target = output_dict[inp][head_node_type][
                            f"{head_name}_target"
                        ]
                        # input features
                        feats = output_dict[inp][head_node_type]["x"]
                        # prediction
                        if predict == False:
                            out = head.regression(feats, labels, target)
                            final_loss += out["loss"]
                            logits_dict[cache_name] = out["logits"]
                            labels_dict[cache_name] = out["labels"]
                        else:
                            logits_dict[cache_name] = head(feats)
                    elif head.training_task == "edge_classification":
                        head_edge_type = lookup_dict[inp][head.edge_type]
                        # name to access predictions
                        cache_name = f"{head_name}___{inp}"
                        # ground truth
                        labels = output_dict[inp][head_name]
                        # input features
                        links = output_dict[inp][f"{head_name}_links"]
                        n1, n2 = head_edge_type[0], head_edge_type[2]
                        n1_feats = output_dict[inp][n1].x
                        n2_feats = output_dict[inp][n2].x
                        # prediction
                        if predict == False:
                            logits = self.edge_pooler(
                                n1_feats, n2_feats, links
                            )
                            out = head.classification(logits, labels)
                            final_loss += out["loss"]
                            logits_dict[cache_name] = out["logits"]
                            labels_dict[cache_name] = out["labels"]
                        else:
                            logits = self.edge_pooler(
                                n1_feats, n2_feats, links
                            )
                            logits_dict[cache_name] = head(logits)
            # pairwise (dual inputs) classification
            if head.training_task in [
                "siamese_graph_classification",
                "siamese_graph_regression",
            ]:
                for inp_1, inp_2 in head.analyze_inputs:
                    # name to access predictions
                    cache_name = f"{head_name}___{inp_1}___{inp_2}"
                    # ground truth
                    labels = kwargs.get(f"common___{cache_name}")
                    # input features
                    feats_1 = output_dict[inp_1].pooled_output
                    feats_2 = output_dict[inp_2].pooled_output
                    # graph classification
                    if head.training_task == "siamese_graph_classification":
                        # prediction
                        if predict == False:
                            out = head.classification(feats_1, feats_2, labels)
                            final_loss += out["loss"]
                            logits_dict[cache_name] = out["logits"]
                            labels_dict[cache_name] = out["labels"]
                        else:
                            logits_dict[cache_name] = head(feats_1, feats_2)
                    # graph regression
                    if head.training_task == "siamese_graph_regression":
                        # prediction
                        if predict == False:
                            out = head.regression(feats_1, feats_2, labels)
                            final_loss += out["loss"]
                            logits_dict[cache_name] = out["logits"]
                            labels_dict[cache_name] = out["labels"]
                        else:
                            logits_dict[cache_name] = head(feats_1, feats_2)
            # contrastive loss
            if head.training_task in ["siamese_graph_convergence"]:
                for inp_1, inp_2, inp_3 in head.analyze_inputs:
                    # name to access predictions
                    cache_name = f"{head_name}___{inp_1}___{inp_2}___{inp_3}"
                    # ground truth
                    labels = kwargs.get(f"common___{cache_name}")
                    if head.training_task == "siamese_graph_convergence":
                        feats_1 = output_dict[inp_1].pooled_output
                        feats_2 = output_dict[inp_2].pooled_output
                        feats_3 = output_dict[inp_3].pooled_output
                        # prediction
                        if predict == False:
                            out = head.convergence(
                                feats_1, feats_2, feats_3, labels
                            )
                            final_loss += out["loss"]
                            logits_dict[cache_name] = out["logits"]
                            labels_dict[cache_name] = out["labels"]
        # average loss
        if len(self.heads) > 0 and loss_can_be_zero == False:
            final_loss = final_loss / len(self.heads)
        # prepare output
        output = DataStructs.MultiTaskOutput(
            loss=final_loss, logits=logits_dict, labels=labels_dict
        )
        return output.split(tensor_names=self.tensor_names)
