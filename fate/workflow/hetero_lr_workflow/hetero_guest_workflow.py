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


from federatedml.logistic_regression.hetero_logistic_regression import HeteroLRGuest
from federatedml.param import LogisticParam
from federatedml.util import consts
from federatedml.util import ParamExtract
from workflow.workflow import WorkFlow
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class LRGuestWorkFlow(WorkFlow):
    def _initialize_model(self, config):
        logistic_param = LogisticParam()
        self.logistic_param = ParamExtract.parse_param_from_config(logistic_param, config)
        self.model = HeteroLRGuest(self.logistic_param)
    
    def _initialize_role_and_mode(self):
        self.role = consts.GUEST
        self.mode = consts.HETERO

if __name__ == "__main__":
    guest_wf = LRGuestWorkFlow()
    guest_wf.run()
