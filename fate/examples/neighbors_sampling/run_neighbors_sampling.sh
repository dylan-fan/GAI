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
if [[ $work_mode -eq 1 ]]; then
    role=$5
fi

cur_dir=$(pwd)
data_dir=$cur_dir/../data
load_file_program=$cur_dir/../load_file/load_file.py
conf_dir=$cur_dir/conf
log_dir=$cur_dir/../../logs
load_data_conf=$conf_dir/load_file.json
guest_runtime_conf=$conf_dir/guest_runtime_conf.json
host_runtime_conf=$conf_dir/host_runtime_conf.json
neighbors_sampling_output_name=neighbors_sampling_output_table_name_${jobid}
neighbors_sampling_output_namespace=neighbors_sampling_output_namespace_${jobid}
python_file_name=get_neighbors_sampling_output
python_file=$cur_dir/${python_file_name}.py

data_set=karate
#data_set=default_credit
#data_set=give_credit
neighbors_sampling_data_host=$data_dir/${data_set}_a.csv
neighbors_sampling_data_guest=$data_dir/${data_set}_b.csv

echo "data dir is : "$data_dir
mode='neighbors_sampling'
#mode='predict'
#mode='neighbors_sampling'
data_table=''
log_file=''

mkdir -p $log_dir

load_file() {
    input_path=$1
    role=$2   
    load_mode=$3
    conf_path=$conf_dir/load_file.json_${role}_${load_mode}_$jobid
    cp $load_data_conf $conf_path
    data_table=${data_set}_${role}_${load_mode}_$jobid
	sed -i "s|_input_path|${input_path}|g" ${conf_path}
	sed -i "s/_table_name/${data_table}/g" ${conf_path}
    sed -i "s/_work_mode/${work_mode}/g" ${conf_path}
    
    python $load_file_program -c ${conf_path}
}

neighbors_sampling() {
    role=$1
    neighbors_sampling_table=$2
    runtime_conf=''
    if [ $role = 'guest' ]; then
        runtime_conf=$guest_runtime_conf
    else
        runtime_conf=$host_runtime_conf
    fi

    cur_runtime_conf=${runtime_conf}_$jobid
    cp $runtime_conf $cur_runtime_conf

    echo "current runtime conf is "$cur_runtime_conf
    echo "neighbors_sampling talbe is "$neighbors_sampling_table
    sed -i "s/_workflow_method/neighbors_sampling/g" $cur_runtime_conf
    sed -i "s/_neighbors_sampling_table_name/$neighbors_sampling_table/g" $cur_runtime_conf
    sed -i "s/_work_mode/$work_mode/g" $cur_runtime_conf
    sed -i "s/_guest_party_id/$guest_partyid/g" $cur_runtime_conf
    sed -i "s/_host_party_id/$host_partyid/g" $cur_runtime_conf
    sed -i "s/_neighbors_sampling_output_table_name/$neighbors_sampling_output_name/g" $cur_runtime_conf
    sed -i "s/_neighbors_sampling_output_namespace/$neighbors_sampling_output_namespace/g" $cur_runtime_conf

    log_file=${log_dir}/${jobid}
    echo "Please check log file in "${log_file}
    if [[ $role == 'guest' ]]; then
        echo "enter guest"
        nohup bash run_guest.sh $cur_runtime_conf $jobid > nohup.guest &
    else
        echo "enter host"
        nohup bash run_host.sh $cur_runtime_conf $jobid > nohup.host &
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


if [ $mode = 'neighbors_sampling' ]; then
    if [[ $work_mode -eq 0 ]]; then
        load_file $neighbors_sampling_data_guest guest neighbors_sampling
        neighbors_sampling_table_guest=$data_table
        load_file $neighbors_sampling_data_host host neighbors_sampling
        neighbors_sampling_table_host=$data_table

        echo "neighbors_sampling table guest is:"$neighbors_sampling_table_guest
        echo "neighbors_sampling table host is:"$neighbors_sampling_table_host

        neighbors_sampling guest $neighbors_sampling_table_guest
        neighbors_sampling host $neighbors_sampling_table_host

        workflow_log=${log_file}/workflow.log
        get_log_result ${workflow_log} Neighbors_sampling_finish
    fi
fi
