#include<stdlib.h>
#include "objects.h"
#include<ctime>
#include<random>

#include "code_objects/exc_inh_pre_codeobject.h"
#include "code_objects/exc_inh_pre_push_spikes.h"
#include "code_objects/exc_inh_synapses_create_generator_codeobject.h"
#include "code_objects/exc_spike_resetter_codeobject.h"
#include "code_objects/exc_spike_thresholder_codeobject.h"
#include "code_objects/exc_stateupdater_codeobject.h"
#include "code_objects/inh_exc_pre_codeobject.h"
#include "code_objects/inh_exc_pre_push_spikes.h"
#include "code_objects/inh_exc_synapses_create_generator_codeobject.h"
#include "code_objects/inh_spike_resetter_codeobject.h"
#include "code_objects/inh_spike_thresholder_codeobject.h"
#include "code_objects/inh_stateupdater_codeobject.h"
#include "code_objects/inp_exc_post_codeobject.h"
#include "code_objects/inp_exc_post_push_spikes.h"
#include "code_objects/inp_exc_pre_codeobject.h"
#include "code_objects/inp_exc_pre_group_variable_set_conditional_codeobject.h"
#include "code_objects/inp_exc_pre_push_spikes.h"
#include "code_objects/inp_exc_synapses_create_generator_codeobject.h"
#include "code_objects/inp_spike_thresholder_codeobject.h"
#include "code_objects/spikes_codeobject.h"


void brian_start()
{
	_init_arrays();
	_load_arrays();
	// Initialize clocks (link timestep and dt to the respective arrays)
    brian::defaultclock.timestep = brian::_array_defaultclock_timestep;
    brian::defaultclock.dt = brian::_array_defaultclock_dt;
    brian::defaultclock.t = brian::_array_defaultclock_t;
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}


