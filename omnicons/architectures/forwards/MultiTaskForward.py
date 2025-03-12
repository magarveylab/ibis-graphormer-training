from torch import nn

from omnicons.data.DataClass import MultiInputData, recover_homogeneous_data
from omnicons.models import DataStructs


class MultiTaskForward(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self, predict: bool = False, loss_can_be_zero: bool = False, **kwargs
    ) -> DataStructs.MultiTaskOutput:
        # variables to track
        final_loss = 0
        output_dict = {}
        logits_dict = {}
        labels_dict = {}
        # run encoder and base model on batch
        if kwargs.get("batch") != None and isinstance(
            kwargs["batch"], MultiInputData
        ):
            data_dict = kwargs["batch"].graphs
        data_dict = recover_homogeneous_data(self.inputs, **kwargs)
        for inp in self.inputs:
            output_dict[inp] = self.get_model_outputs(data_dict[inp])
        # multi task processing
        for head_name, head in self.heads.items():
            # single input classification
            if head.training_task in [
                "node_classification",
                "graph_classification",
                "edge_classification",
            ]:
                # classification tasks
                for inp in head.analyze_inputs:
                    batch = data_dict[inp]
                    if batch == None:
                        continue
                    if head.training_task in [
                        "node_classification",
                        "graph_classification",
                    ]:
                        # name to access predictions
                        cache_name = f"{head_name}___{inp}"
                        # ground truth
                        labels = getattr(batch, head_name)
                        # input features
                        if head.training_task == "node_classification":
                            feats = output_dict[inp].x
                        elif head.training_task == "graph_classification":
                            feats = output_dict[inp].pooled_output
                        # prediction
                        if predict == False:
                            out = head.classification(feats, labels)
                            final_loss += out["loss"]
                            logits_dict[cache_name] = out["logits"]
                            labels_dict[cache_name] = out["labels"]
                        else:
                            logits_dict[cache_name] = head(feats)
                    elif head.training_task in ["edge_classification"]:
                        # name to access predictions
                        cache_name = f"{head_name}___{inp}"
                        # ground truth
                        labels = getattr(batch, head_name)
                        # input features
                        links = getattr(batch, f"{head_name}_links")
                        feats = output_dict[inp].x
                        # prediction
                        if predict == False:
                            logits = self.edge_pooler(feats, links)
                            out = head.classification(logits, labels)
                            final_loss += out["loss"]
                            logits_dict[cache_name] = out["logits"]
                            labels_dict[cache_name] = out["labels"]
                        else:
                            logits = self.edge_pooler(feats, links)
                            logits_dict[cache_name] = head(logits)
            # pairwise (dual inputs) classification
            if head.training_task in [
                "siamese_graph_classification",
                "siamese_node_classification",
            ]:
                for inp_1, inp_2 in head.analyze_inputs:
                    # name to access predictions
                    cache_name = f"{head_name}___{inp_1}___{inp_2}"
                    # ground truth
                    labels = kwargs.get(f"common___{cache_name}")
                    # graph classification
                    if head.training_task == "siamese_graph_classification":
                        # input features
                        feats_1 = output_dict[inp_1].pooled_output
                        feats_2 = output_dict[inp_2].pooled_output
                        # prediction
                        if predict == False:
                            out = head.classification(feats_1, feats_2, labels)
                            final_loss += out["loss"]
                            logits_dict[cache_name] = out["logits"]
                            labels_dict[cache_name] = out["labels"]
                        else:
                            logits_dict[cache_name] = head(feats_1, feats_2)
                    # node classification
                    elif head.training_task == "siamese_node_classification":
                        # input features
                        feats_1 = output_dict[inp_1].x
                        feats_2 = output_dict[inp_2].x
                        feats_indexes_1 = getattr(
                            output_dict[inp_1], f"{head_name}___{inp_1}"
                        )
                        feats_indexes_2 = getattr(
                            output_dict[inp_2], f"{head_name}___{inp_2}"
                        )
                        if predict == False:
                            out = head.classification(
                                feats_1,
                                feats_indexes_1,
                                feats_2,
                                feats_indexes_2,
                                labels,
                            )
                            final_loss += out["loss"]
                            logits_dict[cache_name] = out["logits"]
                            labels_dict[cache_name] = out["labels"]
                        else:
                            logits_dict[cache_name] = head(
                                feats_1,
                                feats_indexes_1,
                                feats_2,
                                feats_indexes_2,
                            )
            # contrastive loss
            if head.training_task in [
                "siamese_graph_convergence",
                "siamese_node_convergence",
            ]:
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
                    elif head.training_task == "siamese_node_convergence":
                        feats_1 = output_dict[inp_1].x
                        feats_2 = output_dict[inp_2].x
                        feats_3 = output_dict[inp_3].x
                        feats_indexes_1 = getattr(
                            output_dict[inp_1], f"{head_name}___{inp_1}"
                        )
                        feats_indexes_2 = getattr(
                            output_dict[inp_2], f"{head_name}___{inp_2}"
                        )
                        feats_indexes_3 = getattr(
                            output_dict[inp_3], f"{head_name}___{inp_3}"
                        )
                        # prediction
                        if predict == False:
                            out = head.convergence(
                                feats_1,
                                feats_indexes_1,
                                feats_2,
                                feats_indexes_2,
                                feats_3,
                                feats_indexes_3,
                                labels,
                            )
                            final_loss += out["loss"]
                            logits_dict[cache_name] = out["logits"]
                            labels_dict[cache_name] = out["labels"]
            # multi model convergance
            if head.training_task == "multi_model_static_graph_convergence":
                for inp_1, inp_2, inp_3 in head.analyze_inputs:
                    # name to access predictions
                    cache_name = f"{head_name}___{inp_1}___{inp_2}___{inp_3}"
                    if head.single_input == True:
                        # current training model
                        feats_1 = output_dict[inp_1].pooled_output  # anchor
                        # other model embeddings
                        feats_2 = kwargs.get(
                            f"common___{head_name}___{inp_2}"
                        )  # positive
                        feats_3 = kwargs.get(
                            f"common___{head_name}___{inp_3}"
                        )  # negative
                    else:
                        # current training model
                        feats_1 = output_dict[inp_1].pooled_output  # anchor
                        feats_2 = output_dict[
                            inp_2
                        ].pooled_output  # positive / negative
                        # other model embeddings
                        feats_3 = kwargs.get(
                            f"common___{head_name}___{inp_3}"
                        )  # negative
                    # prediction
                    if predict == False:
                        out = head.convergence(feats_1, feats_2, feats_3)
                        final_loss += out["loss"]
                        logits_dict[cache_name] = out["logits"]
                        labels_dict[cache_name] = out["labels"]
            # regression tasks
            if head.training_task in ["node_regression"]:
                for inp in head.analyze_inputs:
                    batch = data_dict[inp]
                    if batch == None:
                        continue
                    # name to access predictions
                    cache_name = f"{head_name}___{inp}"
                    # ground truth
                    target = getattr(batch, head_name)
                    is_target = getattr(batch, f"{head_name}_target", None)
                    # features
                    feats = output_dict[inp].x
                    if predict == False:
                        out = head.regression(feats, target, is_target)
                        final_loss += out["loss"]
                        logits_dict[cache_name] = out["logits"]
                        labels_dict[cache_name] = out["labels"]
                    else:
                        logits_dict[cache_name] = head(feats)
        # average loss
        if len(self.heads) > 0 and loss_can_be_zero == False:
            final_loss = final_loss / len(self.heads)
        # prepare output
        output = DataStructs.MultiTaskOutput(
            loss=final_loss, logits=logits_dict, labels=labels_dict
        )
        del data_dict
        del output_dict
        return output.split(tensor_names=self.tensor_names)
