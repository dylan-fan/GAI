/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.webank.ai.fate.core.api.grpc.client.crud;

import com.google.protobuf.ExperimentalApi;
import com.webank.ai.fate.api.eggroll.meta.service.StorageMetaServiceGrpc;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

@Component
@Scope("prototype")
@Deprecated
@ExperimentalApi
public class StorageMetaClientNotCompleted
        extends BaseCrudDelayedResultClient<StorageMetaServiceGrpc.StorageMetaServiceStub> {
/*    private static final Logger LOGGER = LogManager.getLogger(StorageMetaClient.class);

    public Dtable createTable(Dtable dtable) {
        return (Dtable) doCrudRequest(dtable, (stub, request, responseObserver) -> stub.createTable(request, responseObserver));
    }

    public Dtable updateTable(Dtable dtable) {
        return (Dtable) doCrudRequest(dtable, (stub, request, responseObserver) -> stub.updateTable(request, responseObserver));
    }

    public Dtable getById(Dtable dtable) {
        return (Dtable) doCrudRequest(dtable, ((stub, request, responseObserver) -> stub.getTableById(request, responseObserver)));
    }*/
}
