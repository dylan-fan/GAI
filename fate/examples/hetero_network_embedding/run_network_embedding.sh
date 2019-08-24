#!/usr/bin/env bash

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

work_mode=$1
jobid=$2
guest_partyid=$3
host_partyid=$4
arbiter_partyid=$5
if [[ $work_mode -eq 1 ]]; then
    role=$5
fi

cur_dir=$(pwd)
data_dir=$cur_dir/../data
conf_dir=$cur_dir/conf
log_dir=$cur_dir/../../logs
guest_runtime_conf=$conf_dir/guest_runtime_conf.json
host_runtime_conf=$conf_dir/host_runtime_conf.json
arbiter_runtime_conf=$conf_dir/arbiter_runtime_conf.json


echo "data dir is : "$data_dir
mode='network_embedding'

log_file=''

mkdir -p $log_dir


network_embedding() {
    role=$1
    runtime_conf=''
    if [ $role = 'guest' ]; then
        runtime_conf=$guest_runtime_conf
    elif [ $role = 'arbiter' ]; then
        runtime_conf=$arbiter_runtime_conf
    else
        runtime_conf=$host_runtime_conf
    fi

    cur_runtime_conf=${runtime_conf}_$jobid
    cp $runtime_conf $cur_runtime_conf

    echo "current runtime conf is "$cur_runtime_conf
  
    sed -ie "s/_workflow_method/network_embedding/g" $cur_runtime_conf
    sed -ie "s/_work_mode/$work_mode/g" $cur_runtime_conf
    sed -ie "s/_guest_party_id/$guest_partyid/g" $cur_runtime_conf
    sed -ie "s/_host_party_id/$host_partyid/g" $cur_runtime_conf
    sed -ie "s/_arbiter_party_id/$arbiter_partyid/g" $cur_runtime_conf
    sed -ie "s/_jobid/$jobid/g" $cur_runtime_conf

    log_file=${log_dir}/${jobid}
    echo "Please check log file in "${log_file}
    if [[ $role == 'guest' ]]; then
        echo "enter guest"
        nohup bash run_guest.sh $cur_runtime_conf $jobid > nohup.guest &        #设置workflow
    elif [ $role == 'arbiter' ]; then
        echo "enter arbiter"
        nohup bash run_arbiter.sh $cur_runtime_conf $jobid > nohup.arbiter &    #设置workflow
    else
        echo "enter host"
        nohup bash run_host.sh $cur_runtime_conf $jobid > nohup.host &          #设置workflow
    fi


}



get_log_result() {
    log_path=$1
    keyword=$2
    sleep 5s
    time_pass=0
    while true
    do
        if [ ! -f $log_path ];then
            echo "task is prepraring, please wait"
            sleep 5s
            time_pass=$(($time_pass+10))

            if [ $time_pass -gt 120 ]; then
                echo "task failed, check nohup in current path"
                break
            fi

            continue
        fi

        num=$(cat $log_path | grep $keyword | wc -l)
        if [ $num -ge 1 ]; then
            cat $log_path | grep $keyword
            break
        else
            echo "please wait or check more info in "$log_path
            sleep 10s
        fi
    done
}


if [ $mode = 'network_embedding' ]; then
    if [[ $work_mode -eq 0 ]]; then

        network_embedding guest
        network_embedding host
        network_embedding arbiter

        workflow_log=${log_file}/workflow.log
        get_log_result ${workflow_log} Network_embedding_finish
    fi
fi
