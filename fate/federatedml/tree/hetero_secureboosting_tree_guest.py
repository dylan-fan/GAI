#!/usr/bin/env python    
# -*- coding: utf-8 -*- 

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
################################################################################
#
#
################################################################################

# =============================================================================
# HeteroSecureBoostingGuest 
# =============================================================================

from federatedml.feature.quantile import Quantile
from federatedml.util import ClassifyLabelChecker
from federatedml.util import RegressionLabelChecker
from federatedml.tree import HeteroDecisionTreeGuest
from federatedml.optim import DiffConverge
from federatedml.tree import BoostingTree
from federatedml.util import HeteroSecureBoostingTreeTransferVariable
from federatedml.util import consts
from numpy import random
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.encrypt_mode import EncryptModeCalculator
from federatedml.loss import SigmoidBinaryCrossEntropyLoss
from federatedml.loss import SoftmaxCrossEntropyLoss
from federatedml.loss import LeastSquaredErrorLoss
from federatedml.loss import HuberLoss
from federatedml.loss import LeastAbsoluteErrorLoss
from federatedml.loss import TweedieLoss
from federatedml.loss import LogCoshLoss
from federatedml.loss import FairLoss
from federatedml.evaluation import Evaluation

from arch.api import federation
from arch.api.model_manager import manager
from arch.api.proto.boosting_tree_model_meta_pb2 import ObjectiveMeta
from arch.api.proto.boosting_tree_model_meta_pb2 import QuantileMeta
from arch.api.proto.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from arch.api.proto.boosting_tree_model_param_pb2 import FeatureImportanceInfo
from arch.api.proto.boosting_tree_model_param_pb2 import BoostingTreeModelParam
import numpy as np
import functools
from operator import itemgetter
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class HeteroSecureBoostingTreeGuest(BoostingTree):
    def __init__(self, secureboost_tree_param):
        super(HeteroSecureBoostingTreeGuest, self).__init__(secureboost_tree_param)

        self.convegence = None
        self.y = None
        self.F = None
        self.data_bin = None
        self.loss = None
        self.init_score = None
        self.classes_dict = {}
        self.classes_ = []
        self.num_classes = 0
        self.classify_target = "binary"
        self.feature_num = None
        self.encrypter = None
        self.grad_and_hess = None
        self.flowid = 0
        self.tree_dim = 1
        self.tree_meta = None
        self.trees_ = []
        self.history_loss = []
        self.bin_split_points = None
        self.bin_sparse_points = None
        self.encrypted_mode_calculator = None
        self.runtime_idx = 0
        self.feature_importances_ = {}

        self.transfer_inst = HeteroSecureBoostingTreeTransferVariable()

    def set_loss(self, objective_param):
        loss_type = objective_param.objective
        params = objective_param.params
        LOGGER.info("set objective, objective is {}".format(loss_type))
        if self.task_type == consts.CLASSIFICATION:
            if loss_type == "cross_entropy":
                if self.num_classes == 2:
                    self.loss = SigmoidBinaryCrossEntropyLoss()
                else:
                    self.loss = SoftmaxCrossEntropyLoss()
            else:
                raise NotImplementedError("objective %s not supported yet" % (loss_type))
        elif self.task_type == consts.REGRESSION:
            if loss_type == "lse":
                self.loss = LeastSquaredErrorLoss()
            elif loss_type == "lae":
                self.loss = LeastAbsoluteErrorLoss()
            elif loss_type == "huber":
                self.loss = HuberLoss(params[0])
            elif loss_type == "fair":
                self.loss = FairLoss(params[0])
            elif loss_type == "tweedie":
                self.loss = TweedieLoss(params[0])
            elif loss_type == "log_cosh":
                self.loss = LogCoshLoss()
            else:
                raise NotImplementedError("objective %s not supported yet" % (loss_type))
        else:
            raise NotImplementedError("objective %s not supported yet" % (loss_type))

    def convert_feature_to_bin(self, data_instance):
        LOGGER.info("convert feature to bins")
        self.data_bin, self.bin_split_points, self.bin_sparse_points = \
            Quantile.convert_feature_to_bin(
                data_instance, self.quantile_method, self.bin_num,
                self.bin_gap, self.bin_sample_num)
        LOGGER.info("convert feature to bins over")

    def set_y(self):
        LOGGER.info("set label from data and check label")
        self.y = self.data_bin.mapValues(lambda instance: instance.label)
        self.check_label()

    def set_flowid(self, flowid=0):
        LOGGER.info("set flowid, flowid is {}".format(flowid))
        self.flowid = flowid

    def set_runtime_idx(self, runtime_idx):
        self.runtime_idx = runtime_idx

    def generate_flowid(self, round_num, tree_num):
        LOGGER.info("generate flowid")
        return ".".join(map(str, [self.flowid, round_num, tree_num]))

    def check_label(self):
        LOGGER.info("check label")
        if self.task_type == consts.CLASSIFICATION:
            self.num_classes, self.classes_ = ClassifyLabelChecker.validate_y(self.y)
            if self.num_classes > 2:
                self.classify_target = "multinomial"
                self.tree_dim = self.num_classes

            range_from_zero = True
            for _class in self.classes_:
                try:
                    if _class >= 0 and _class < range_from_zero and isinstance(_class, int):
                        continue
                    else:
                        range_from_zero = False
                        break
                except:
                    range_from_zero = False

            self.classes_ = sorted(self.classes_)
            if not range_from_zero:
                class_mapping = dict(zip(self.classes_, range(self.num_classes)))
                self.y = self.y.mapValues(lambda _class: class_mapping[_class])

        else:
            RegressionLabelChecker.validate_y(self.y)

        self.set_loss(self.objective_param)

    def generate_encrypter(self):
        LOGGER.info("generate encrypter")
        if self.encrypt_param.method == consts.PAILLIER:
            self.encrypter = PaillierEncrypt()
            self.encrypter.generate_key(self.encrypt_param.key_length)
        else:
            raise NotImplementedError("encrypt method not supported yes!!!")

        self.encrypted_calculator = EncryptModeCalculator(self.encrypter, self.calculated_mode, self.re_encrypted_rate)

    @staticmethod
    def accumulate_f(f_val, new_f_val, lr=0.1, idx=0):
        f_val[idx] += lr * new_f_val
        return f_val

    def update_feature_importance(self, tree_feature_importance):
        for fid in tree_feature_importance:
            if fid not in self.feature_importances_:
                self.feature_importances_[fid] = 0

            self.feature_importances_[fid] += tree_feature_importance[fid]

    def update_f_value(self, new_f=None, tidx=-1):
        LOGGER.info("update tree f value, tree idx is {}".format(tidx))
        if self.F is None:
            if self.tree_dim > 1:
                self.F, self.init_score = self.loss.initialize(self.y, self.tree_dim)
            else:
                LOGGER.info("tree_dim is %d" % (self.tree_dim))
                self.F, self.init_score = self.loss.initialize(self.y)
        else:
            accumuldate_f = functools.partial(self.accumulate_f,
                                              lr=self.learning_rate,
                                              idx=tidx)

            self.F = self.F.join(new_f, accumuldate_f)

    def compute_grad_and_hess(self):
        LOGGER.info("compute grad and hess")
        loss_method = self.loss
        if self.task_type == consts.CLASSIFICATION:
            self.grad_and_hess = self.y.join(self.F, lambda y, f_val: \
                (loss_method.compute_grad(y, loss_method.predict(f_val)), \
                 loss_method.compute_hess(y, loss_method.predict(f_val))))
        else:
            self.grad_and_hess = self.y.join(self.F, lambda y, f_val:
            (loss_method.compute_grad(y, f_val),
             loss_method.compute_hess(y, f_val)))

    def compute_loss(self):
        LOGGER.info("compute loss")
        if self.task_type == consts.CLASSIFICATION:
            loss_method = self.loss
            y_predict = self.F.mapValues(lambda val: loss_method.predict(val))
            loss = loss_method.compute_loss(self.y, y_predict)
        elif self.task_type == consts.REGRESSION:
            if self.objective_param.objective in ["lse", "lae", "logcosh", "tweedie", "log_cosh", "huber"]:
                loss_method = self.loss
                loss = loss_method.compute_loss(self.y, self.F)
            else:
                loss_method = self.loss
                y_predict = self.F.mapValues(lambda val: loss_method.predict(val))
                loss = loss_method.compute_loss(self.y, y_predict)

        return loss

    def get_grad_and_hess(self, tree_idx):
        LOGGER.info("get grad and hess of tree {}".format(tree_idx))
        grad_and_hess_subtree = self.grad_and_hess.mapValues(
            lambda grad_and_hess: (grad_and_hess[0][tree_idx], grad_and_hess[1][tree_idx]))
        return grad_and_hess_subtree

    def check_convergence(self, loss):
        LOGGER.info("check convergence")
        if self.convegence is None:
            self.convegence = DiffConverge()

        return self.convegence.is_converge(loss)

    def sample_valid_features(self):
        LOGGER.info("sample valid features")
        if self.feature_num is None:
            self.feature_num = self.bin_split_points.shape[0]

        choose_feature = random.choice(range(0, self.feature_num), \
                                       max(1, int(self.subsample_feature_rate * self.feature_num)), replace=False)

        valid_features = [False for i in range(self.feature_num)]
        for fid in choose_feature:
            valid_features[fid] = True
        return valid_features

    def sync_tree_dim(self):
        LOGGER.info("sync tree dim to host")
        federation.remote(obj=self.tree_dim,
                          name=self.transfer_inst.tree_dim.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.tree_dim),
                          role=consts.HOST,
                          idx=-1)

    def sync_stop_flag(self, stop_flag, num_round):
        LOGGER.info("sync stop flag to host, boosting round is {}".format(num_round))
        federation.remote(obj=stop_flag,
                          name=self.transfer_inst.stop_flag.name,
                          tag=self.transfer_inst.generate_transferid(self.transfer_inst.stop_flag, num_round),
                          role=consts.HOST,
                          idx=-1)

    def fit(self, data_inst):
        LOGGER.info("begin to train secureboosting guest model")
        data_inst = self.data_alignment(data_inst)
        self.convert_feature_to_bin(data_inst)
        self.set_y()
        self.update_f_value()
        self.generate_encrypter()

        self.sync_tree_dim()

        for i in range(self.num_trees):
            # n_tree = []
            self.compute_grad_and_hess()
            for tidx in range(self.tree_dim):
                tree_inst = HeteroDecisionTreeGuest(self.tree_param)

                tree_inst.set_inputinfo(self.data_bin, self.get_grad_and_hess(tidx), self.bin_split_points,
                                        self.bin_sparse_points)

                valid_features = self.sample_valid_features()
                tree_inst.set_valid_features(valid_features)
                tree_inst.set_encrypter(self.encrypter)
                tree_inst.set_encrypted_mode_calculator(self.encrypted_calculator)
                tree_inst.set_flowid(self.generate_flowid(i, tidx))

                tree_inst.fit()

                tree_meta, tree_param = tree_inst.get_model()
                self.trees_.append(tree_param)
                if self.tree_meta is None:
                    self.tree_meta = tree_meta
                # n_tree.append(tree_inst.get_tree_model())
                self.update_f_value(new_f=tree_inst.predict_weights, tidx=tidx)
                self.update_feature_importance(tree_inst.get_feature_importance())

            # self.trees_.append(n_tree)
            loss = self.compute_loss()
            self.history_loss.append(loss)
            LOGGER.info("round {} loss is {}".format(i, loss))

            if self.n_iter_no_change is True:
                if self.check_convergence(loss):
                    self.sync_stop_flag(True, i)
                    break
                else:
                    self.sync_stop_flag(False, i)

        LOGGER.info("end to train secureboosting guest model")

    def predict_f_value(self, data_inst):
        LOGGER.info("predict tree f value, there are {} trees".format(len(self.trees_)))
        tree_dim = self.tree_dim
        init_score = self.init_score
        self.F = data_inst.mapValues(lambda v: init_score)
        rounds = len(self.trees_) // self.tree_dim
        for i in range(rounds):
            for tidx in range(self.tree_dim):
                tree_inst = HeteroDecisionTreeGuest(self.tree_param)
                tree_inst.load_model(self.tree_meta, self.trees_[i * self.tree_dim + tidx])
                # tree_inst.set_tree_model(self.trees_[i * self.tree_dim + tidx])
                tree_inst.set_flowid(self.generate_flowid(i, tidx))

                predict_data = tree_inst.predict(data_inst)
                self.update_f_value(new_f=predict_data, tidx=tidx)

    def predict(self, data_inst, predict_param):
        LOGGER.info("start predict")
        data_inst = self.data_alignment(data_inst)
        self.predict_f_value(data_inst)
        if self.task_type == consts.CLASSIFICATION:
            loss_method = self.loss
            predicts = self.F.mapValues(lambda f: loss_method.predict(f))
        elif self.task_type == consts.REGRESSION:
            if self.objective_param.objective in ["lse", "lae", "huber", "log_cosh", "fair", "tweedie"]:
                predicts = self.F
            else:
                raise NotImplementedError("objective {} not supprted yet".format(self.objective_param.objective))

        if self.task_type == consts.CLASSIFICATION:
            classes_ = self.classes_
            if self.num_classes == 2:
                predict_label = predicts.mapValues(
                    lambda pred: classes_[1] if pred > predict_param.threshold else classes_[0])
            else:
                predict_label = predicts.mapValues(lambda preds: classes_[np.argmax(preds)])

            if predict_param.with_proba:
                predict_result = data_inst.join(predicts, lambda inst, predict_prob: (inst.label, predict_prob))
            else:
                predict_result = data_inst.mapValues(lambda inst: (inst.label, None))

            predict_result = predict_result.join(predict_label,
                                                 lambda label_prob, predict_label: (
                                                     label_prob[0], label_prob[1], predict_label))
        elif self.task_type == consts.REGRESSION:
            predict_result = data_inst.join(predicts, lambda inst, pred: (inst.label, pred, None))

        else:
            raise NotImplementedError("task type {} not supported yet".format(self.task_type))

        LOGGER.info("end predict")

        return predict_result

    def get_feature_importance(self):
        return self.feature_importances_
        
    def get_model_meta(self):
        model_meta = BoostingTreeModelMeta()
        model_meta.tree_meta.CopyFrom(self.tree_meta)
        model_meta.learning_rate = self.learning_rate 
        model_meta.num_trees = self.num_trees 
        model_meta.quantile_meta.CopyFrom(QuantileMeta(quantile_method=self.quantile_method,
                                                       bin_num=self.bin_num,
                                                       bin_gap=self.bin_gap,
                                                       bin_sample_num=self.bin_sample_num))
        #modelmeta.objective.CopyFrom(ObjectiveParamMeta(objective=self.objective_param.objective, param=self.objective_param.params))
        model_meta.objective_meta.CopyFrom(ObjectiveMeta(objective=self.objective_param.objective,
                                                         param=self.objective_param.params))
        model_meta.task_type = self.task_type
        model_meta.tree_dim = self.tree_dim
        model_meta.n_iter_no_change = self.n_iter_no_change
        model_meta.tol = self.tol
        model_meta.num_classes = self.num_classes
        model_meta.classes_.extend(map(str, self.classes_))
        
        meta_name = "HeteroSecureBoostingTreeGuest.meta"
          
        return meta_name, model_meta

    def set_model_meta(self, model_meta):
        self.tree_meta = model_meta.tree_meta
        self.learning_rate = model_meta.learning_rate
        self.num_trees = model_meta.num_trees
        self.quantile_method = model_meta.quantile_meta.quantile_method
        self.bin_num = model_meta.quantile_meta.bin_num
        self.bin_gap = model_meta.quantile_meta.bin_gap
        self.bin_sample_num = model_meta.quantile_meta.bin_sample_num
        self.objective_param.objective = model_meta.objective_meta.objective
        self.objective_param.params = list(model_meta.objective_meta.param)
        self.task_type = model_meta.task_type
        self.tree_dim = model_meta.tree_dim
        self.num_classes = model_meta.num_classes
        self.n_iter_no_change = model_meta.n_iter_no_change
        self.tol = model_meta.tol
        self.classes_ = list(model_meta.classes_)

        self.set_loss(self.objective_param)

    def get_model_param(self):
        model_param = BoostingTreeModelParam()
        model_param.tree_num = len(list(self.trees_))
        model_param.trees_.extend(self.trees_)
        model_param.init_score.extend(self.init_score)
        model_param.losses.extend(self.history_loss)

        feature_importances = list(self.get_feature_importance().items())
        feature_importances = sorted(feature_importances, key=itemgetter(1), reverse=True)
        feature_importance_param = []
        for (sitename, fid), _importance in feature_importances:
            feature_importance_param.append(FeatureImportanceInfo(sitename=sitename,
                                                                  fid=fid,
                                                                  importance=_importance))
        model_param.feature_importances.extend(feature_importance_param)

        param_name = "HeteroSecureBoostingTreeGuest.param"

        return param_name, model_param

    def set_model_param(self, model_param):
        self.trees_ = list(model_param.trees_)
        self.init_score = np.array(list(model_param.init_score))
        self.history_loss = list(model_param.losses)

    def save_model(self, model_table, model_namespace):
        LOGGER.info("save model")
        meta_name, meta_protobuf = self.get_model_meta()
        param_name, param_protobuf = self.get_model_param()
        manager.save_model(buffer_type=meta_name,
                           proto_buffer=meta_protobuf,
                           name=model_table,
                           namespace=model_namespace)

        manager.save_model(buffer_type=param_name,
                           proto_buffer=param_protobuf,
                           name=model_table,
                           namespace=model_namespace)

        return [(meta_name, param_name)]

    def load_model(self, model_table, model_namespace):
        LOGGER.info("load model")
        model_meta = BoostingTreeModelMeta()
        manager.read_model(buffer_type="HeteroSecureBoostingTreeGuest.meta",
                           proto_buffer=model_meta,
                           name=model_table,
                           namespace=model_namespace)
        self.set_model_meta(model_meta)

        model_param = BoostingTreeModelParam()
        manager.read_model(buffer_type="HeteroSecureBoostingTreeGuest.param",
                           proto_buffer=model_param,
                           name=model_table,
                           namespace=model_namespace)
        self.set_model_param(model_param)

    def evaluate(self, labels, pred_prob, pred_labels, evaluate_param):
        LOGGER.info("evaluate data")
        predict_res = None

        if self.task_type == consts.CLASSIFICATION:
            if evaluate_param.classi_type == consts.BINARY:
                predict_res = pred_prob
            elif evaluate_param.classi_type == consts.MULTY:
                predict_res = pred_labels
            else:
                LOGGER.warning("unknown classification type, return None as evaluation results")
        elif self.task_type == consts.REGRESSION:
            predict_res = pred_prob
        else:
            LOGGER.warning("unknown task type, return None as evaluation results")

        eva = Evaluation(evaluate_param.classi_type)
        return eva.report(labels, predict_res, evaluate_param.metrics, evaluate_param.thresholds,
                          evaluate_param.pos_label)
