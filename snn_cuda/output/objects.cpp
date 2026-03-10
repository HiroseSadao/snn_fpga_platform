

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include<random>
#include<vector>
#include<iostream>
#include<fstream>
#include<map>
#include<tuple>
#include<cstdlib>
#include<string>

namespace brian {

std::string results_dir = "results/";  // can be overwritten by --results_dir command line arg

// For multhreading, we need one generator for each thread. We also create a distribution for
// each thread, even though this is not strictly necessary for the uniform distribution, as
// the distribution is stateless.
std::vector< RandomGenerator > _random_generators;

//////////////// networks /////////////////
Network network;

void set_variable_from_value(std::string varname, char* var_pointer, size_t size, char value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << (value == 1 ? "True" : "False") << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_value(std::string varname, T* var_pointer, size_t size, T value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << value << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_file(std::string varname, T* var_pointer, size_t data_size, std::string filename) {
    ifstream f;
    streampos size;
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' from file '" << filename << "'" << std::endl;
    #endif
    f.open(filename, ios::in | ios::binary | ios::ate);
    size = f.tellg();
    if (size != data_size) {
        std::cerr << "Error reading '" << filename << "': file size " << size << " does not match expected size " << data_size << std::endl;
        return;
    }
    f.seekg(0, ios::beg);
    if (f.is_open())
        f.read(reinterpret_cast<char *>(var_pointer), data_size);
    else
        std::cerr << "Could not read '" << filename << "'" << std::endl;
    if (f.fail())
        std::cerr << "Error reading '" << filename << "'" << std::endl;
}

//////////////// set arrays by name ///////
void set_variable_by_name(std::string name, std::string s_value) {
    size_t var_size;
    size_t data_size;
    // C-style or Python-style capitalization is allowed for boolean values
    if (s_value == "true" || s_value == "True")
        s_value = "1";
    else if (s_value == "false" || s_value == "False")
        s_value = "0";
    // non-dynamic arrays
    if (name == "exc._spikespace") {
        var_size = 51;
        data_size = 51*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_exc__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_exc__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "exc.ge") {
        var_size = 50;
        data_size = 50*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_exc_ge, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_exc_ge, data_size, s_value);
        }
        return;
    }
    if (name == "exc.gi") {
        var_size = 50;
        data_size = 50*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_exc_gi, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_exc_gi, data_size, s_value);
        }
        return;
    }
    if (name == "exc.lastspike") {
        var_size = 50;
        data_size = 50*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_exc_lastspike, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_exc_lastspike, data_size, s_value);
        }
        return;
    }
    if (name == "exc.not_refractory") {
        var_size = 50;
        data_size = 50*sizeof(char);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value(name, _array_exc_not_refractory, var_size, (char)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_exc_not_refractory, data_size, s_value);
        }
        return;
    }
    if (name == "exc.theta") {
        var_size = 50;
        data_size = 50*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_exc_theta, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_exc_theta, data_size, s_value);
        }
        return;
    }
    if (name == "exc.v") {
        var_size = 50;
        data_size = 50*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_exc_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_exc_v, data_size, s_value);
        }
        return;
    }
    if (name == "inh._spikespace") {
        var_size = 51;
        data_size = 51*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_inh__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_inh__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "inh.ge") {
        var_size = 50;
        data_size = 50*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_inh_ge, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_inh_ge, data_size, s_value);
        }
        return;
    }
    if (name == "inh.lastspike") {
        var_size = 50;
        data_size = 50*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_inh_lastspike, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_inh_lastspike, data_size, s_value);
        }
        return;
    }
    if (name == "inh.not_refractory") {
        var_size = 50;
        data_size = 50*sizeof(char);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value(name, _array_inh_not_refractory, var_size, (char)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_inh_not_refractory, data_size, s_value);
        }
        return;
    }
    if (name == "inh.v") {
        var_size = 50;
        data_size = 50*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_inh_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_inh_v, data_size, s_value);
        }
        return;
    }
    if (name == "inp._spikespace") {
        var_size = 785;
        data_size = 785*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_inp__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_inp__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "inp.rates") {
        var_size = 784;
        data_size = 784*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_inp_rates, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_inp_rates, data_size, s_value);
        }
        return;
    }
    // dynamic arrays (1d)
    if (name == "exc_inh.delay") {
        var_size = _dynamic_array_exc_inh_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_exc_inh_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_exc_inh_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "inh_exc.delay") {
        var_size = _dynamic_array_inh_exc_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_inh_exc_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_inh_exc_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "inp_exc.delay") {
        var_size = _dynamic_array_inp_exc_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_inp_exc_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_inp_exc_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "inp_exc.delay") {
        var_size = _dynamic_array_inp_exc_delay_1.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_inp_exc_delay_1[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_inp_exc_delay_1[0], data_size, s_value);
        }
        return;
    }
    if (name == "inp_exc.lastupdate") {
        var_size = _dynamic_array_inp_exc_lastupdate.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_inp_exc_lastupdate[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_inp_exc_lastupdate[0], data_size, s_value);
        }
        return;
    }
    if (name == "inp_exc.post1") {
        var_size = _dynamic_array_inp_exc_post1.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_inp_exc_post1[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_inp_exc_post1[0], data_size, s_value);
        }
        return;
    }
    if (name == "inp_exc.post2") {
        var_size = _dynamic_array_inp_exc_post2.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_inp_exc_post2[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_inp_exc_post2[0], data_size, s_value);
        }
        return;
    }
    if (name == "inp_exc.post2_before") {
        var_size = _dynamic_array_inp_exc_post2_before.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_inp_exc_post2_before[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_inp_exc_post2_before[0], data_size, s_value);
        }
        return;
    }
    if (name == "inp_exc.pre") {
        var_size = _dynamic_array_inp_exc_pre.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_inp_exc_pre[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_inp_exc_pre[0], data_size, s_value);
        }
        return;
    }
    if (name == "inp_exc.w") {
        var_size = _dynamic_array_inp_exc_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_inp_exc_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_inp_exc_w[0], data_size, s_value);
        }
        return;
    }
    std::cerr << "Cannot set unknown variable '" << name << "'." << std::endl;
    exit(1);
}
//////////////// arrays ///////////////////
double * _array_defaultclock_dt;
const int _num__array_defaultclock_dt = 1;
double * _array_defaultclock_t;
const int _num__array_defaultclock_t = 1;
int64_t * _array_defaultclock_timestep;
const int _num__array_defaultclock_timestep = 1;
int32_t * _array_exc__spikespace;
const int _num__array_exc__spikespace = 51;
double * _array_exc_ge;
const int _num__array_exc_ge = 50;
double * _array_exc_gi;
const int _num__array_exc_gi = 50;
int32_t * _array_exc_i;
const int _num__array_exc_i = 50;
int32_t * _array_exc_inh_N;
const int _num__array_exc_inh_N = 1;
double * _array_exc_lastspike;
const int _num__array_exc_lastspike = 50;
char * _array_exc_not_refractory;
const int _num__array_exc_not_refractory = 50;
double * _array_exc_theta;
const int _num__array_exc_theta = 50;
double * _array_exc_v;
const int _num__array_exc_v = 50;
int32_t * _array_inh__spikespace;
const int _num__array_inh__spikespace = 51;
int32_t * _array_inh_exc_N;
const int _num__array_inh_exc_N = 1;
double * _array_inh_ge;
const int _num__array_inh_ge = 50;
int32_t * _array_inh_i;
const int _num__array_inh_i = 50;
double * _array_inh_lastspike;
const int _num__array_inh_lastspike = 50;
char * _array_inh_not_refractory;
const int _num__array_inh_not_refractory = 50;
double * _array_inh_v;
const int _num__array_inh_v = 50;
int32_t * _array_inp__spikespace;
const int _num__array_inp__spikespace = 785;
int32_t * _array_inp_exc_N;
const int _num__array_inp_exc_N = 1;
int32_t * _array_inp_i;
const int _num__array_inp_i = 784;
double * _array_inp_rates;
const int _num__array_inp_rates = 784;
int32_t * _array_spikes__source_idx;
const int _num__array_spikes__source_idx = 50;
int32_t * _array_spikes_count;
const int _num__array_spikes_count = 50;
int32_t * _array_spikes_N;
const int _num__array_spikes_N = 1;

//////////////// dynamic arrays 1d /////////
std::vector<int32_t> _dynamic_array_exc_inh__synaptic_post;
std::vector<int32_t> _dynamic_array_exc_inh__synaptic_pre;
std::vector<double> _dynamic_array_exc_inh_delay;
std::vector<int32_t> _dynamic_array_exc_inh_N_incoming;
std::vector<int32_t> _dynamic_array_exc_inh_N_outgoing;
std::vector<int32_t> _dynamic_array_inh_exc__synaptic_post;
std::vector<int32_t> _dynamic_array_inh_exc__synaptic_pre;
std::vector<double> _dynamic_array_inh_exc_delay;
std::vector<int32_t> _dynamic_array_inh_exc_N_incoming;
std::vector<int32_t> _dynamic_array_inh_exc_N_outgoing;
std::vector<int32_t> _dynamic_array_inp_exc__synaptic_post;
std::vector<int32_t> _dynamic_array_inp_exc__synaptic_pre;
std::vector<double> _dynamic_array_inp_exc_delay;
std::vector<double> _dynamic_array_inp_exc_delay_1;
std::vector<double> _dynamic_array_inp_exc_lastupdate;
std::vector<int32_t> _dynamic_array_inp_exc_N_incoming;
std::vector<int32_t> _dynamic_array_inp_exc_N_outgoing;
std::vector<double> _dynamic_array_inp_exc_post1;
std::vector<double> _dynamic_array_inp_exc_post2;
std::vector<double> _dynamic_array_inp_exc_post2_before;
std::vector<double> _dynamic_array_inp_exc_pre;
std::vector<double> _dynamic_array_inp_exc_w;
std::vector<int32_t> _dynamic_array_spikes_i;
std::vector<double> _dynamic_array_spikes_t;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
double * _static_array__array_inp_rates;
const int _num__static_array__array_inp_rates = 784;
double * _static_array__dynamic_array_inp_exc_w;
const int _num__static_array__dynamic_array_inp_exc_w = 39200;

//////////////// synapses /////////////////
// exc_inh
SynapticPathway exc_inh_pre(
    _dynamic_array_exc_inh__synaptic_pre,
    0, 50);
// inh_exc
SynapticPathway inh_exc_pre(
    _dynamic_array_inh_exc__synaptic_pre,
    0, 50);
// inp_exc
SynapticPathway inp_exc_post(
    _dynamic_array_inp_exc__synaptic_post,
    0, 50);
SynapticPathway inp_exc_pre(
    _dynamic_array_inp_exc__synaptic_pre,
    0, 784);

//////////////// clocks ///////////////////
Clock defaultclock;  // attributes will be set in run.cpp

// Profiling information for each code object
}

void _init_arrays()
{
    using namespace brian;

    // Arrays initialized to 0
    _array_defaultclock_dt = new double[1];
    
    for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;

    _array_defaultclock_t = new double[1];
    
    for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;

    _array_defaultclock_timestep = new int64_t[1];
    
    for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;

    _array_exc__spikespace = new int32_t[51];
    
    for(int i=0; i<51; i++) _array_exc__spikespace[i] = 0;

    _array_exc_ge = new double[50];
    
    for(int i=0; i<50; i++) _array_exc_ge[i] = 0;

    _array_exc_gi = new double[50];
    
    for(int i=0; i<50; i++) _array_exc_gi[i] = 0;

    _array_exc_i = new int32_t[50];
    
    for(int i=0; i<50; i++) _array_exc_i[i] = 0;

    _array_exc_inh_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_exc_inh_N[i] = 0;

    _array_exc_lastspike = new double[50];
    
    for(int i=0; i<50; i++) _array_exc_lastspike[i] = 0;

    _array_exc_not_refractory = new char[50];
    
    for(int i=0; i<50; i++) _array_exc_not_refractory[i] = 0;

    _array_exc_theta = new double[50];
    
    for(int i=0; i<50; i++) _array_exc_theta[i] = 0;

    _array_exc_v = new double[50];
    
    for(int i=0; i<50; i++) _array_exc_v[i] = 0;

    _array_inh__spikespace = new int32_t[51];
    
    for(int i=0; i<51; i++) _array_inh__spikespace[i] = 0;

    _array_inh_exc_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_inh_exc_N[i] = 0;

    _array_inh_ge = new double[50];
    
    for(int i=0; i<50; i++) _array_inh_ge[i] = 0;

    _array_inh_i = new int32_t[50];
    
    for(int i=0; i<50; i++) _array_inh_i[i] = 0;

    _array_inh_lastspike = new double[50];
    
    for(int i=0; i<50; i++) _array_inh_lastspike[i] = 0;

    _array_inh_not_refractory = new char[50];
    
    for(int i=0; i<50; i++) _array_inh_not_refractory[i] = 0;

    _array_inh_v = new double[50];
    
    for(int i=0; i<50; i++) _array_inh_v[i] = 0;

    _array_inp__spikespace = new int32_t[785];
    
    for(int i=0; i<785; i++) _array_inp__spikespace[i] = 0;

    _array_inp_exc_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_inp_exc_N[i] = 0;

    _array_inp_i = new int32_t[784];
    
    for(int i=0; i<784; i++) _array_inp_i[i] = 0;

    _array_inp_rates = new double[784];
    
    for(int i=0; i<784; i++) _array_inp_rates[i] = 0;

    _array_spikes__source_idx = new int32_t[50];
    
    for(int i=0; i<50; i++) _array_spikes__source_idx[i] = 0;

    _array_spikes_count = new int32_t[50];
    
    for(int i=0; i<50; i++) _array_spikes_count[i] = 0;

    _array_spikes_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_spikes_N[i] = 0;


    // Arrays initialized to an "arange"
    _array_exc_i = new int32_t[50];
    
    for(int i=0; i<50; i++) _array_exc_i[i] = 0 + i;

    _array_inh_i = new int32_t[50];
    
    for(int i=0; i<50; i++) _array_inh_i[i] = 0 + i;

    _array_inp_i = new int32_t[784];
    
    for(int i=0; i<784; i++) _array_inp_i[i] = 0 + i;

    _array_spikes__source_idx = new int32_t[50];
    
    for(int i=0; i<50; i++) _array_spikes__source_idx[i] = 0 + i;


    // static arrays
    _static_array__array_inp_rates = new double[784];
    _static_array__dynamic_array_inp_exc_w = new double[39200];

    // Random number generator states
    std::random_device rd;
    for (int i=0; i<1; i++)
        _random_generators.push_back(RandomGenerator());
}

void _load_arrays()
{
    using namespace brian;

    ifstream f_static_array__array_inp_rates;
    f_static_array__array_inp_rates.open("static_arrays/_static_array__array_inp_rates", ios::in | ios::binary);
    if(f_static_array__array_inp_rates.is_open())
    {
        f_static_array__array_inp_rates.read(reinterpret_cast<char*>(_static_array__array_inp_rates), 784*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _static_array__array_inp_rates." << endl;
    }
    ifstream f_static_array__dynamic_array_inp_exc_w;
    f_static_array__dynamic_array_inp_exc_w.open("static_arrays/_static_array__dynamic_array_inp_exc_w", ios::in | ios::binary);
    if(f_static_array__dynamic_array_inp_exc_w.is_open())
    {
        f_static_array__dynamic_array_inp_exc_w.read(reinterpret_cast<char*>(_static_array__dynamic_array_inp_exc_w), 39200*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_inp_exc_w." << endl;
    }
}

void _write_arrays()
{
    using namespace brian;

    ofstream outfile__array_defaultclock_dt;
    outfile__array_defaultclock_dt.open(results_dir + "_array_defaultclock_dt_1978099143", ios::binary | ios::out);
    if(outfile__array_defaultclock_dt.is_open())
    {
        outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
        outfile__array_defaultclock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
    }
    ofstream outfile__array_defaultclock_t;
    outfile__array_defaultclock_t.open(results_dir + "_array_defaultclock_t_2669362164", ios::binary | ios::out);
    if(outfile__array_defaultclock_t.is_open())
    {
        outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
        outfile__array_defaultclock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_t." << endl;
    }
    ofstream outfile__array_defaultclock_timestep;
    outfile__array_defaultclock_timestep.open(results_dir + "_array_defaultclock_timestep_144223508", ios::binary | ios::out);
    if(outfile__array_defaultclock_timestep.is_open())
    {
        outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
        outfile__array_defaultclock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
    }
    ofstream outfile__array_exc__spikespace;
    outfile__array_exc__spikespace.open(results_dir + "_array_exc__spikespace_186237112", ios::binary | ios::out);
    if(outfile__array_exc__spikespace.is_open())
    {
        outfile__array_exc__spikespace.write(reinterpret_cast<char*>(_array_exc__spikespace), 51*sizeof(_array_exc__spikespace[0]));
        outfile__array_exc__spikespace.close();
    } else
    {
        std::cout << "Error writing output file for _array_exc__spikespace." << endl;
    }
    ofstream outfile__array_exc_ge;
    outfile__array_exc_ge.open(results_dir + "_array_exc_ge_597544964", ios::binary | ios::out);
    if(outfile__array_exc_ge.is_open())
    {
        outfile__array_exc_ge.write(reinterpret_cast<char*>(_array_exc_ge), 50*sizeof(_array_exc_ge[0]));
        outfile__array_exc_ge.close();
    } else
    {
        std::cout << "Error writing output file for _array_exc_ge." << endl;
    }
    ofstream outfile__array_exc_gi;
    outfile__array_exc_gi.open(results_dir + "_array_exc_gi_707501103", ios::binary | ios::out);
    if(outfile__array_exc_gi.is_open())
    {
        outfile__array_exc_gi.write(reinterpret_cast<char*>(_array_exc_gi), 50*sizeof(_array_exc_gi[0]));
        outfile__array_exc_gi.close();
    } else
    {
        std::cout << "Error writing output file for _array_exc_gi." << endl;
    }
    ofstream outfile__array_exc_i;
    outfile__array_exc_i.open(results_dir + "_array_exc_i_2892359347", ios::binary | ios::out);
    if(outfile__array_exc_i.is_open())
    {
        outfile__array_exc_i.write(reinterpret_cast<char*>(_array_exc_i), 50*sizeof(_array_exc_i[0]));
        outfile__array_exc_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_exc_i." << endl;
    }
    ofstream outfile__array_exc_inh_N;
    outfile__array_exc_inh_N.open(results_dir + "_array_exc_inh_N_1971115519", ios::binary | ios::out);
    if(outfile__array_exc_inh_N.is_open())
    {
        outfile__array_exc_inh_N.write(reinterpret_cast<char*>(_array_exc_inh_N), 1*sizeof(_array_exc_inh_N[0]));
        outfile__array_exc_inh_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_exc_inh_N." << endl;
    }
    ofstream outfile__array_exc_lastspike;
    outfile__array_exc_lastspike.open(results_dir + "_array_exc_lastspike_2193292514", ios::binary | ios::out);
    if(outfile__array_exc_lastspike.is_open())
    {
        outfile__array_exc_lastspike.write(reinterpret_cast<char*>(_array_exc_lastspike), 50*sizeof(_array_exc_lastspike[0]));
        outfile__array_exc_lastspike.close();
    } else
    {
        std::cout << "Error writing output file for _array_exc_lastspike." << endl;
    }
    ofstream outfile__array_exc_not_refractory;
    outfile__array_exc_not_refractory.open(results_dir + "_array_exc_not_refractory_2812404659", ios::binary | ios::out);
    if(outfile__array_exc_not_refractory.is_open())
    {
        outfile__array_exc_not_refractory.write(reinterpret_cast<char*>(_array_exc_not_refractory), 50*sizeof(_array_exc_not_refractory[0]));
        outfile__array_exc_not_refractory.close();
    } else
    {
        std::cout << "Error writing output file for _array_exc_not_refractory." << endl;
    }
    ofstream outfile__array_exc_theta;
    outfile__array_exc_theta.open(results_dir + "_array_exc_theta_488540787", ios::binary | ios::out);
    if(outfile__array_exc_theta.is_open())
    {
        outfile__array_exc_theta.write(reinterpret_cast<char*>(_array_exc_theta), 50*sizeof(_array_exc_theta[0]));
        outfile__array_exc_theta.close();
    } else
    {
        std::cout << "Error writing output file for _array_exc_theta." << endl;
    }
    ofstream outfile__array_exc_v;
    outfile__array_exc_v.open(results_dir + "_array_exc_v_560851782", ios::binary | ios::out);
    if(outfile__array_exc_v.is_open())
    {
        outfile__array_exc_v.write(reinterpret_cast<char*>(_array_exc_v), 50*sizeof(_array_exc_v[0]));
        outfile__array_exc_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_exc_v." << endl;
    }
    ofstream outfile__array_inh__spikespace;
    outfile__array_inh__spikespace.open(results_dir + "_array_inh__spikespace_928358693", ios::binary | ios::out);
    if(outfile__array_inh__spikespace.is_open())
    {
        outfile__array_inh__spikespace.write(reinterpret_cast<char*>(_array_inh__spikespace), 51*sizeof(_array_inh__spikespace[0]));
        outfile__array_inh__spikespace.close();
    } else
    {
        std::cout << "Error writing output file for _array_inh__spikespace." << endl;
    }
    ofstream outfile__array_inh_exc_N;
    outfile__array_inh_exc_N.open(results_dir + "_array_inh_exc_N_2538356766", ios::binary | ios::out);
    if(outfile__array_inh_exc_N.is_open())
    {
        outfile__array_inh_exc_N.write(reinterpret_cast<char*>(_array_inh_exc_N), 1*sizeof(_array_inh_exc_N[0]));
        outfile__array_inh_exc_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_inh_exc_N." << endl;
    }
    ofstream outfile__array_inh_ge;
    outfile__array_inh_ge.open(results_dir + "_array_inh_ge_1828648284", ios::binary | ios::out);
    if(outfile__array_inh_ge.is_open())
    {
        outfile__array_inh_ge.write(reinterpret_cast<char*>(_array_inh_ge), 50*sizeof(_array_inh_ge[0]));
        outfile__array_inh_ge.close();
    } else
    {
        std::cout << "Error writing output file for _array_inh_ge." << endl;
    }
    ofstream outfile__array_inh_i;
    outfile__array_inh_i.open(results_dir + "_array_inh_i_280161296", ios::binary | ios::out);
    if(outfile__array_inh_i.is_open())
    {
        outfile__array_inh_i.write(reinterpret_cast<char*>(_array_inh_i), 50*sizeof(_array_inh_i[0]));
        outfile__array_inh_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_inh_i." << endl;
    }
    ofstream outfile__array_inh_lastspike;
    outfile__array_inh_lastspike.open(results_dir + "_array_inh_lastspike_3624176193", ios::binary | ios::out);
    if(outfile__array_inh_lastspike.is_open())
    {
        outfile__array_inh_lastspike.write(reinterpret_cast<char*>(_array_inh_lastspike), 50*sizeof(_array_inh_lastspike[0]));
        outfile__array_inh_lastspike.close();
    } else
    {
        std::cout << "Error writing output file for _array_inh_lastspike." << endl;
    }
    ofstream outfile__array_inh_not_refractory;
    outfile__array_inh_not_refractory.open(results_dir + "_array_inh_not_refractory_1374068593", ios::binary | ios::out);
    if(outfile__array_inh_not_refractory.is_open())
    {
        outfile__array_inh_not_refractory.write(reinterpret_cast<char*>(_array_inh_not_refractory), 50*sizeof(_array_inh_not_refractory[0]));
        outfile__array_inh_not_refractory.close();
    } else
    {
        std::cout << "Error writing output file for _array_inh_not_refractory." << endl;
    }
    ofstream outfile__array_inh_v;
    outfile__array_inh_v.open(results_dir + "_array_inh_v_2646270437", ios::binary | ios::out);
    if(outfile__array_inh_v.is_open())
    {
        outfile__array_inh_v.write(reinterpret_cast<char*>(_array_inh_v), 50*sizeof(_array_inh_v[0]));
        outfile__array_inh_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_inh_v." << endl;
    }
    ofstream outfile__array_inp__spikespace;
    outfile__array_inp__spikespace.open(results_dir + "_array_inp__spikespace_3646018706", ios::binary | ios::out);
    if(outfile__array_inp__spikespace.is_open())
    {
        outfile__array_inp__spikespace.write(reinterpret_cast<char*>(_array_inp__spikespace), 785*sizeof(_array_inp__spikespace[0]));
        outfile__array_inp__spikespace.close();
    } else
    {
        std::cout << "Error writing output file for _array_inp__spikespace." << endl;
    }
    ofstream outfile__array_inp_exc_N;
    outfile__array_inp_exc_N.open(results_dir + "_array_inp_exc_N_3279786679", ios::binary | ios::out);
    if(outfile__array_inp_exc_N.is_open())
    {
        outfile__array_inp_exc_N.write(reinterpret_cast<char*>(_array_inp_exc_N), 1*sizeof(_array_inp_exc_N[0]));
        outfile__array_inp_exc_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_inp_exc_N." << endl;
    }
    ofstream outfile__array_inp_i;
    outfile__array_inp_i.open(results_dir + "_array_inp_i_42409688", ios::binary | ios::out);
    if(outfile__array_inp_i.is_open())
    {
        outfile__array_inp_i.write(reinterpret_cast<char*>(_array_inp_i), 784*sizeof(_array_inp_i[0]));
        outfile__array_inp_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_inp_i." << endl;
    }
    ofstream outfile__array_inp_rates;
    outfile__array_inp_rates.open(results_dir + "_array_inp_rates_1476716205", ios::binary | ios::out);
    if(outfile__array_inp_rates.is_open())
    {
        outfile__array_inp_rates.write(reinterpret_cast<char*>(_array_inp_rates), 784*sizeof(_array_inp_rates[0]));
        outfile__array_inp_rates.close();
    } else
    {
        std::cout << "Error writing output file for _array_inp_rates." << endl;
    }
    ofstream outfile__array_spikes__source_idx;
    outfile__array_spikes__source_idx.open(results_dir + "_array_spikes__source_idx_2303802139", ios::binary | ios::out);
    if(outfile__array_spikes__source_idx.is_open())
    {
        outfile__array_spikes__source_idx.write(reinterpret_cast<char*>(_array_spikes__source_idx), 50*sizeof(_array_spikes__source_idx[0]));
        outfile__array_spikes__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikes__source_idx." << endl;
    }
    ofstream outfile__array_spikes_count;
    outfile__array_spikes_count.open(results_dir + "_array_spikes_count_328618678", ios::binary | ios::out);
    if(outfile__array_spikes_count.is_open())
    {
        outfile__array_spikes_count.write(reinterpret_cast<char*>(_array_spikes_count), 50*sizeof(_array_spikes_count[0]));
        outfile__array_spikes_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikes_count." << endl;
    }
    ofstream outfile__array_spikes_N;
    outfile__array_spikes_N.open(results_dir + "_array_spikes_N_3246544668", ios::binary | ios::out);
    if(outfile__array_spikes_N.is_open())
    {
        outfile__array_spikes_N.write(reinterpret_cast<char*>(_array_spikes_N), 1*sizeof(_array_spikes_N[0]));
        outfile__array_spikes_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikes_N." << endl;
    }

    ofstream outfile__dynamic_array_exc_inh__synaptic_post;
    outfile__dynamic_array_exc_inh__synaptic_post.open(results_dir + "_dynamic_array_exc_inh__synaptic_post_2709685911", ios::binary | ios::out);
    if(outfile__dynamic_array_exc_inh__synaptic_post.is_open())
    {
        if (! _dynamic_array_exc_inh__synaptic_post.empty() )
        {
            outfile__dynamic_array_exc_inh__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_exc_inh__synaptic_post[0]), _dynamic_array_exc_inh__synaptic_post.size()*sizeof(_dynamic_array_exc_inh__synaptic_post[0]));
            outfile__dynamic_array_exc_inh__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_exc_inh__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_exc_inh__synaptic_pre;
    outfile__dynamic_array_exc_inh__synaptic_pre.open(results_dir + "_dynamic_array_exc_inh__synaptic_pre_1472676030", ios::binary | ios::out);
    if(outfile__dynamic_array_exc_inh__synaptic_pre.is_open())
    {
        if (! _dynamic_array_exc_inh__synaptic_pre.empty() )
        {
            outfile__dynamic_array_exc_inh__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_exc_inh__synaptic_pre[0]), _dynamic_array_exc_inh__synaptic_pre.size()*sizeof(_dynamic_array_exc_inh__synaptic_pre[0]));
            outfile__dynamic_array_exc_inh__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_exc_inh__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_exc_inh_delay;
    outfile__dynamic_array_exc_inh_delay.open(results_dir + "_dynamic_array_exc_inh_delay_3843339198", ios::binary | ios::out);
    if(outfile__dynamic_array_exc_inh_delay.is_open())
    {
        if (! _dynamic_array_exc_inh_delay.empty() )
        {
            outfile__dynamic_array_exc_inh_delay.write(reinterpret_cast<char*>(&_dynamic_array_exc_inh_delay[0]), _dynamic_array_exc_inh_delay.size()*sizeof(_dynamic_array_exc_inh_delay[0]));
            outfile__dynamic_array_exc_inh_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_exc_inh_delay." << endl;
    }
    ofstream outfile__dynamic_array_exc_inh_N_incoming;
    outfile__dynamic_array_exc_inh_N_incoming.open(results_dir + "_dynamic_array_exc_inh_N_incoming_473016067", ios::binary | ios::out);
    if(outfile__dynamic_array_exc_inh_N_incoming.is_open())
    {
        if (! _dynamic_array_exc_inh_N_incoming.empty() )
        {
            outfile__dynamic_array_exc_inh_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_exc_inh_N_incoming[0]), _dynamic_array_exc_inh_N_incoming.size()*sizeof(_dynamic_array_exc_inh_N_incoming[0]));
            outfile__dynamic_array_exc_inh_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_exc_inh_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_exc_inh_N_outgoing;
    outfile__dynamic_array_exc_inh_N_outgoing.open(results_dir + "_dynamic_array_exc_inh_N_outgoing_992860121", ios::binary | ios::out);
    if(outfile__dynamic_array_exc_inh_N_outgoing.is_open())
    {
        if (! _dynamic_array_exc_inh_N_outgoing.empty() )
        {
            outfile__dynamic_array_exc_inh_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_exc_inh_N_outgoing[0]), _dynamic_array_exc_inh_N_outgoing.size()*sizeof(_dynamic_array_exc_inh_N_outgoing[0]));
            outfile__dynamic_array_exc_inh_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_exc_inh_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_inh_exc__synaptic_post;
    outfile__dynamic_array_inh_exc__synaptic_post.open(results_dir + "_dynamic_array_inh_exc__synaptic_post_1032635143", ios::binary | ios::out);
    if(outfile__dynamic_array_inh_exc__synaptic_post.is_open())
    {
        if (! _dynamic_array_inh_exc__synaptic_post.empty() )
        {
            outfile__dynamic_array_inh_exc__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_inh_exc__synaptic_post[0]), _dynamic_array_inh_exc__synaptic_post.size()*sizeof(_dynamic_array_inh_exc__synaptic_post[0]));
            outfile__dynamic_array_inh_exc__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inh_exc__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_inh_exc__synaptic_pre;
    outfile__dynamic_array_inh_exc__synaptic_pre.open(results_dir + "_dynamic_array_inh_exc__synaptic_pre_1350859130", ios::binary | ios::out);
    if(outfile__dynamic_array_inh_exc__synaptic_pre.is_open())
    {
        if (! _dynamic_array_inh_exc__synaptic_pre.empty() )
        {
            outfile__dynamic_array_inh_exc__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_inh_exc__synaptic_pre[0]), _dynamic_array_inh_exc__synaptic_pre.size()*sizeof(_dynamic_array_inh_exc__synaptic_pre[0]));
            outfile__dynamic_array_inh_exc__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inh_exc__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_inh_exc_delay;
    outfile__dynamic_array_inh_exc_delay.open(results_dir + "_dynamic_array_inh_exc_delay_3779943007", ios::binary | ios::out);
    if(outfile__dynamic_array_inh_exc_delay.is_open())
    {
        if (! _dynamic_array_inh_exc_delay.empty() )
        {
            outfile__dynamic_array_inh_exc_delay.write(reinterpret_cast<char*>(&_dynamic_array_inh_exc_delay[0]), _dynamic_array_inh_exc_delay.size()*sizeof(_dynamic_array_inh_exc_delay[0]));
            outfile__dynamic_array_inh_exc_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inh_exc_delay." << endl;
    }
    ofstream outfile__dynamic_array_inh_exc_N_incoming;
    outfile__dynamic_array_inh_exc_N_incoming.open(results_dir + "_dynamic_array_inh_exc_N_incoming_2784204773", ios::binary | ios::out);
    if(outfile__dynamic_array_inh_exc_N_incoming.is_open())
    {
        if (! _dynamic_array_inh_exc_N_incoming.empty() )
        {
            outfile__dynamic_array_inh_exc_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_inh_exc_N_incoming[0]), _dynamic_array_inh_exc_N_incoming.size()*sizeof(_dynamic_array_inh_exc_N_incoming[0]));
            outfile__dynamic_array_inh_exc_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inh_exc_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_inh_exc_N_outgoing;
    outfile__dynamic_array_inh_exc_N_outgoing.open(results_dir + "_dynamic_array_inh_exc_N_outgoing_2196760383", ios::binary | ios::out);
    if(outfile__dynamic_array_inh_exc_N_outgoing.is_open())
    {
        if (! _dynamic_array_inh_exc_N_outgoing.empty() )
        {
            outfile__dynamic_array_inh_exc_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_inh_exc_N_outgoing[0]), _dynamic_array_inh_exc_N_outgoing.size()*sizeof(_dynamic_array_inh_exc_N_outgoing[0]));
            outfile__dynamic_array_inh_exc_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inh_exc_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_inp_exc__synaptic_post;
    outfile__dynamic_array_inp_exc__synaptic_post.open(results_dir + "_dynamic_array_inp_exc__synaptic_post_1083293328", ios::binary | ios::out);
    if(outfile__dynamic_array_inp_exc__synaptic_post.is_open())
    {
        if (! _dynamic_array_inp_exc__synaptic_post.empty() )
        {
            outfile__dynamic_array_inp_exc__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_inp_exc__synaptic_post[0]), _dynamic_array_inp_exc__synaptic_post.size()*sizeof(_dynamic_array_inp_exc__synaptic_post[0]));
            outfile__dynamic_array_inp_exc__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inp_exc__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_inp_exc__synaptic_pre;
    outfile__dynamic_array_inp_exc__synaptic_pre.open(results_dir + "_dynamic_array_inp_exc__synaptic_pre_1248288757", ios::binary | ios::out);
    if(outfile__dynamic_array_inp_exc__synaptic_pre.is_open())
    {
        if (! _dynamic_array_inp_exc__synaptic_pre.empty() )
        {
            outfile__dynamic_array_inp_exc__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_inp_exc__synaptic_pre[0]), _dynamic_array_inp_exc__synaptic_pre.size()*sizeof(_dynamic_array_inp_exc__synaptic_pre[0]));
            outfile__dynamic_array_inp_exc__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inp_exc__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_inp_exc_delay;
    outfile__dynamic_array_inp_exc_delay.open(results_dir + "_dynamic_array_inp_exc_delay_4118314961", ios::binary | ios::out);
    if(outfile__dynamic_array_inp_exc_delay.is_open())
    {
        if (! _dynamic_array_inp_exc_delay.empty() )
        {
            outfile__dynamic_array_inp_exc_delay.write(reinterpret_cast<char*>(&_dynamic_array_inp_exc_delay[0]), _dynamic_array_inp_exc_delay.size()*sizeof(_dynamic_array_inp_exc_delay[0]));
            outfile__dynamic_array_inp_exc_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inp_exc_delay." << endl;
    }
    ofstream outfile__dynamic_array_inp_exc_delay_1;
    outfile__dynamic_array_inp_exc_delay_1.open(results_dir + "_dynamic_array_inp_exc_delay_1_1975336661", ios::binary | ios::out);
    if(outfile__dynamic_array_inp_exc_delay_1.is_open())
    {
        if (! _dynamic_array_inp_exc_delay_1.empty() )
        {
            outfile__dynamic_array_inp_exc_delay_1.write(reinterpret_cast<char*>(&_dynamic_array_inp_exc_delay_1[0]), _dynamic_array_inp_exc_delay_1.size()*sizeof(_dynamic_array_inp_exc_delay_1[0]));
            outfile__dynamic_array_inp_exc_delay_1.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inp_exc_delay_1." << endl;
    }
    ofstream outfile__dynamic_array_inp_exc_lastupdate;
    outfile__dynamic_array_inp_exc_lastupdate.open(results_dir + "_dynamic_array_inp_exc_lastupdate_2840491242", ios::binary | ios::out);
    if(outfile__dynamic_array_inp_exc_lastupdate.is_open())
    {
        if (! _dynamic_array_inp_exc_lastupdate.empty() )
        {
            outfile__dynamic_array_inp_exc_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_inp_exc_lastupdate[0]), _dynamic_array_inp_exc_lastupdate.size()*sizeof(_dynamic_array_inp_exc_lastupdate[0]));
            outfile__dynamic_array_inp_exc_lastupdate.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inp_exc_lastupdate." << endl;
    }
    ofstream outfile__dynamic_array_inp_exc_N_incoming;
    outfile__dynamic_array_inp_exc_N_incoming.open(results_dir + "_dynamic_array_inp_exc_N_incoming_818361908", ios::binary | ios::out);
    if(outfile__dynamic_array_inp_exc_N_incoming.is_open())
    {
        if (! _dynamic_array_inp_exc_N_incoming.empty() )
        {
            outfile__dynamic_array_inp_exc_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_inp_exc_N_incoming[0]), _dynamic_array_inp_exc_N_incoming.size()*sizeof(_dynamic_array_inp_exc_N_incoming[0]));
            outfile__dynamic_array_inp_exc_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inp_exc_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_inp_exc_N_outgoing;
    outfile__dynamic_array_inp_exc_N_outgoing.open(results_dir + "_dynamic_array_inp_exc_N_outgoing_400246510", ios::binary | ios::out);
    if(outfile__dynamic_array_inp_exc_N_outgoing.is_open())
    {
        if (! _dynamic_array_inp_exc_N_outgoing.empty() )
        {
            outfile__dynamic_array_inp_exc_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_inp_exc_N_outgoing[0]), _dynamic_array_inp_exc_N_outgoing.size()*sizeof(_dynamic_array_inp_exc_N_outgoing[0]));
            outfile__dynamic_array_inp_exc_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inp_exc_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_inp_exc_post1;
    outfile__dynamic_array_inp_exc_post1.open(results_dir + "_dynamic_array_inp_exc_post1_1466787084", ios::binary | ios::out);
    if(outfile__dynamic_array_inp_exc_post1.is_open())
    {
        if (! _dynamic_array_inp_exc_post1.empty() )
        {
            outfile__dynamic_array_inp_exc_post1.write(reinterpret_cast<char*>(&_dynamic_array_inp_exc_post1[0]), _dynamic_array_inp_exc_post1.size()*sizeof(_dynamic_array_inp_exc_post1[0]));
            outfile__dynamic_array_inp_exc_post1.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inp_exc_post1." << endl;
    }
    ofstream outfile__dynamic_array_inp_exc_post2;
    outfile__dynamic_array_inp_exc_post2.open(results_dir + "_dynamic_array_inp_exc_post2_3462673590", ios::binary | ios::out);
    if(outfile__dynamic_array_inp_exc_post2.is_open())
    {
        if (! _dynamic_array_inp_exc_post2.empty() )
        {
            outfile__dynamic_array_inp_exc_post2.write(reinterpret_cast<char*>(&_dynamic_array_inp_exc_post2[0]), _dynamic_array_inp_exc_post2.size()*sizeof(_dynamic_array_inp_exc_post2[0]));
            outfile__dynamic_array_inp_exc_post2.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inp_exc_post2." << endl;
    }
    ofstream outfile__dynamic_array_inp_exc_post2_before;
    outfile__dynamic_array_inp_exc_post2_before.open(results_dir + "_dynamic_array_inp_exc_post2_before_124177844", ios::binary | ios::out);
    if(outfile__dynamic_array_inp_exc_post2_before.is_open())
    {
        if (! _dynamic_array_inp_exc_post2_before.empty() )
        {
            outfile__dynamic_array_inp_exc_post2_before.write(reinterpret_cast<char*>(&_dynamic_array_inp_exc_post2_before[0]), _dynamic_array_inp_exc_post2_before.size()*sizeof(_dynamic_array_inp_exc_post2_before[0]));
            outfile__dynamic_array_inp_exc_post2_before.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inp_exc_post2_before." << endl;
    }
    ofstream outfile__dynamic_array_inp_exc_pre;
    outfile__dynamic_array_inp_exc_pre.open(results_dir + "_dynamic_array_inp_exc_pre_3190156689", ios::binary | ios::out);
    if(outfile__dynamic_array_inp_exc_pre.is_open())
    {
        if (! _dynamic_array_inp_exc_pre.empty() )
        {
            outfile__dynamic_array_inp_exc_pre.write(reinterpret_cast<char*>(&_dynamic_array_inp_exc_pre[0]), _dynamic_array_inp_exc_pre.size()*sizeof(_dynamic_array_inp_exc_pre[0]));
            outfile__dynamic_array_inp_exc_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inp_exc_pre." << endl;
    }
    ofstream outfile__dynamic_array_inp_exc_w;
    outfile__dynamic_array_inp_exc_w.open(results_dir + "_dynamic_array_inp_exc_w_2987090371", ios::binary | ios::out);
    if(outfile__dynamic_array_inp_exc_w.is_open())
    {
        if (! _dynamic_array_inp_exc_w.empty() )
        {
            outfile__dynamic_array_inp_exc_w.write(reinterpret_cast<char*>(&_dynamic_array_inp_exc_w[0]), _dynamic_array_inp_exc_w.size()*sizeof(_dynamic_array_inp_exc_w[0]));
            outfile__dynamic_array_inp_exc_w.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_inp_exc_w." << endl;
    }
    ofstream outfile__dynamic_array_spikes_i;
    outfile__dynamic_array_spikes_i.open(results_dir + "_dynamic_array_spikes_i_2751340298", ios::binary | ios::out);
    if(outfile__dynamic_array_spikes_i.is_open())
    {
        if (! _dynamic_array_spikes_i.empty() )
        {
            outfile__dynamic_array_spikes_i.write(reinterpret_cast<char*>(&_dynamic_array_spikes_i[0]), _dynamic_array_spikes_i.size()*sizeof(_dynamic_array_spikes_i[0]));
            outfile__dynamic_array_spikes_i.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikes_i." << endl;
    }
    ofstream outfile__dynamic_array_spikes_t;
    outfile__dynamic_array_spikes_t.open(results_dir + "_dynamic_array_spikes_t_3237508051", ios::binary | ios::out);
    if(outfile__dynamic_array_spikes_t.is_open())
    {
        if (! _dynamic_array_spikes_t.empty() )
        {
            outfile__dynamic_array_spikes_t.write(reinterpret_cast<char*>(&_dynamic_array_spikes_t[0]), _dynamic_array_spikes_t.size()*sizeof(_dynamic_array_spikes_t[0]));
            outfile__dynamic_array_spikes_t.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikes_t." << endl;
    }

    // Write last run info to disk
    ofstream outfile_last_run_info;
    outfile_last_run_info.open(results_dir + "last_run_info.txt", ios::out);
    if(outfile_last_run_info.is_open())
    {
        outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
        outfile_last_run_info.close();
    } else
    {
        std::cout << "Error writing last run info to file." << std::endl;
    }
}

void _dealloc_arrays()
{
    using namespace brian;


    // static arrays
    if(_static_array__array_inp_rates!=0)
    {
        delete [] _static_array__array_inp_rates;
        _static_array__array_inp_rates = 0;
    }
    if(_static_array__dynamic_array_inp_exc_w!=0)
    {
        delete [] _static_array__dynamic_array_inp_exc_w;
        _static_array__dynamic_array_inp_exc_w = 0;
    }
}

