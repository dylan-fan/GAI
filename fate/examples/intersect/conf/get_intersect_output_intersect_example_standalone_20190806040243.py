# This the base python file of get intersection output, should no modify it

from arch.api import eggroll

name = "host_intersect_output_table_name_intersect_example_standalone_20190806040243"
namespace = "host_intersect_output_namespace_intersect_example_standalone_20190806040243"
role = name.split('_')[0]
eggroll.init("get_intersect_output", mode = 0)
table = eggroll.table(name, namespace)

print(role +"_intersect output count:"+str(table.count()))
