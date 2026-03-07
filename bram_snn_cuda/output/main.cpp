#include <stdlib.h>
#include "objects.h"
#include <csignal>
#include <ctime>
#include <time.h>

#include "run.h"
#include "brianlib/common_math.h"

#include "code_objects/exc_inh_pre_codeobject.h"
#include "code_objects/exc_inh_pre_push_spikes.h"
#include "code_objects/before_run_exc_inh_pre_push_spikes.h"
#include "code_objects/exc_inh_synapses_create_generator_codeobject.h"
#include "code_objects/exc_spike_resetter_codeobject.h"
#include "code_objects/exc_spike_thresholder_codeobject.h"
#include "code_objects/after_run_exc_spike_thresholder_codeobject.h"
#include "code_objects/exc_stateupdater_codeobject.h"
#include "code_objects/inh_exc_pre_codeobject.h"
#include "code_objects/inh_exc_pre_push_spikes.h"
#include "code_objects/before_run_inh_exc_pre_push_spikes.h"
#include "code_objects/inh_exc_synapses_create_generator_codeobject.h"
#include "code_objects/inh_spike_resetter_codeobject.h"
#include "code_objects/inh_spike_thresholder_codeobject.h"
#include "code_objects/after_run_inh_spike_thresholder_codeobject.h"
#include "code_objects/inh_stateupdater_codeobject.h"
#include "code_objects/inp_exc_post_codeobject.h"
#include "code_objects/inp_exc_post_push_spikes.h"
#include "code_objects/before_run_inp_exc_post_push_spikes.h"
#include "code_objects/inp_exc_pre_codeobject.h"
#include "code_objects/inp_exc_pre_group_variable_set_conditional_codeobject.h"
#include "code_objects/inp_exc_pre_push_spikes.h"
#include "code_objects/before_run_inp_exc_pre_push_spikes.h"
#include "code_objects/inp_exc_synapses_create_generator_codeobject.h"
#include "code_objects/inp_spike_thresholder_codeobject.h"
#include "code_objects/after_run_inp_spike_thresholder_codeobject.h"
#include "code_objects/spikes_codeobject.h"


#include <iostream>
#include <fstream>
#include <string>




void set_from_command_line(const std::vector<std::string> args)
{
    for (const auto& arg : args) {
		// Split into two parts
		size_t equal_sign = arg.find("=");
		auto name = arg.substr(0, equal_sign);
		auto value = arg.substr(equal_sign + 1, arg.length());
		brian::set_variable_by_name(name, value);
	}
}

void _int_handler(int signal_num) {
	if (Network::_globally_running && !Network::_globally_stopped) {
		Network::_globally_stopped = true;
	} else {
		std::signal(signal_num, SIG_DFL);
		std::raise(signal_num);
	}
}

int main(int argc, char **argv)
{
	std::signal(SIGINT, _int_handler);
	std::random_device _rd;
	std::vector<std::string> args(argv + 1, argv + argc);
	if (args.size() >=2 && args[0] == "--results_dir")
	{
		brian::results_dir = args[1];
		#ifdef DEBUG
		std::cout << "Setting results dir to '" << brian::results_dir << "'" << std::endl;
		#endif
		args.erase(args.begin(), args.begin()+2);
	}
        

	brian_start();
        

	{
		using namespace brian;

		
                
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.001;
        for (int _i=0; _i<1; _i++)
            brian::_random_generators[_i].seed(0L + _i);
        
                        
                        for(int i=0; i<_num__array_exc_lastspike; i++)
                        {
                            _array_exc_lastspike[i] = - 10000.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_exc_not_refractory; i++)
                        {
                            _array_exc_not_refractory[i] = true;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_exc_v; i++)
                        {
                            _array_exc_v[i] = - 0.065;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_exc_theta; i++)
                        {
                            _array_exc_theta[i] = 0.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_inh_lastspike; i++)
                        {
                            _array_inh_lastspike[i] = - 10000.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_inh_not_refractory; i++)
                        {
                            _array_inh_not_refractory[i] = true;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_inh_v; i++)
                        {
                            _array_inh_v[i] = - 0.06;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_inp_rates; i++)
                        {
                            _array_inp_rates[i] = _static_array__array_inp_rates[i];
                        }
                        
        _run_inp_exc_synapses_create_generator_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_inp_exc_w.size(); i++)
                        {
                            _dynamic_array_inp_exc_w[i] = _static_array__dynamic_array_inp_exc_w[i];
                        }
                        
        _run_inp_exc_pre_group_variable_set_conditional_codeobject();
        _run_exc_inh_synapses_create_generator_codeobject();
        _run_inh_exc_synapses_create_generator_codeobject();
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_t[0] = 0.0;
        _before_run_exc_inh_pre_push_spikes();
        _before_run_inh_exc_pre_push_spikes();
        _before_run_inp_exc_pre_push_spikes();
        _before_run_inp_exc_post_push_spikes();
        network.clear();
        network.add(&defaultclock, _run_exc_stateupdater_codeobject);
        network.add(&defaultclock, _run_inh_stateupdater_codeobject);
        network.add(&defaultclock, _run_exc_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_inh_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_inp_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_spikes_codeobject);
        network.add(&defaultclock, _run_exc_inh_pre_push_spikes);
        network.add(&defaultclock, _run_exc_inh_pre_codeobject);
        network.add(&defaultclock, _run_inh_exc_pre_push_spikes);
        network.add(&defaultclock, _run_inh_exc_pre_codeobject);
        network.add(&defaultclock, _run_inp_exc_pre_push_spikes);
        network.add(&defaultclock, _run_inp_exc_pre_codeobject);
        network.add(&defaultclock, _run_inp_exc_post_push_spikes);
        network.add(&defaultclock, _run_inp_exc_post_codeobject);
        network.add(&defaultclock, _run_exc_spike_resetter_codeobject);
        network.add(&defaultclock, _run_inh_spike_resetter_codeobject);
        set_from_command_line(args);
        network.run(0.0, NULL, 10.0);
        _after_run_exc_spike_thresholder_codeobject();
        _after_run_inh_spike_thresholder_codeobject();
        _after_run_inp_spike_thresholder_codeobject();
        #ifdef DEBUG
        _debugmsg_spikes_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_exc_inh_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_inh_exc_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_inp_exc_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_inp_exc_post_codeobject();
        #endif

	}
        

	brian_end();
        

	return 0;
}