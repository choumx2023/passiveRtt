from CuckooHashTable import CuckooHashTable
import ipaddress

class FlowTable(CuckooHashTable):
    def __init__(self, flow_table_params, test):
        super().__init__(flow_table_params['initial_size'])
        self._custom_print('round{}/{}: Initialized flow table'.format(test, flow_table_params['rounds']))
        # Set SYN action
        