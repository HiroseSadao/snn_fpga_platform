
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include<random>
#include<vector>


namespace brian {

extern std::string results_dir;

class RandomGenerator {
    private:
        std::mt19937 gen;
        double stored_gauss;
        bool has_stored_gauss = false;
    public:
        RandomGenerator() {
            seed();
        }
        void seed() {
            std::random_device rd;
            gen.seed(rd());
            has_stored_gauss = false;
        }
        void seed(unsigned long seed) {
            gen.seed(seed);
            has_stored_gauss = false;
        }
        double rand() {
            /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
            const long a = gen() >> 5;
            const long b = gen() >> 6;
            return (a * 67108864.0 + b) / 9007199254740992.0;
        }

        double randn() {
            if (has_stored_gauss) {
                const double tmp = stored_gauss;
                has_stored_gauss = false;
                return tmp;
            }
            else {
                double f, x1, x2, r2;

                do {
                    x1 = 2.0*rand() - 1.0;
                    x2 = 2.0*rand() - 1.0;
                    r2 = x1*x1 + x2*x2;
                }
                while (r2 >= 1.0 || r2 == 0.0);

                /* Box-Muller transform */
                f = sqrt(-2.0*log(r2)/r2);
                /* Keep for next call */
                stored_gauss = f*x1;
                has_stored_gauss = true;
                return f*x2;
            }
        }
};

// In OpenMP we need one state per thread
extern std::vector< RandomGenerator > _random_generators;

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network network;



void set_variable_by_name(std::string, std::string);

//////////////// dynamic arrays ///////////
extern std::vector<int32_t> _dynamic_array_exc_inh__synaptic_post;
extern std::vector<int32_t> _dynamic_array_exc_inh__synaptic_pre;
extern std::vector<double> _dynamic_array_exc_inh_delay;
extern std::vector<int32_t> _dynamic_array_exc_inh_N_incoming;
extern std::vector<int32_t> _dynamic_array_exc_inh_N_outgoing;
extern std::vector<int32_t> _dynamic_array_inh_exc__synaptic_post;
extern std::vector<int32_t> _dynamic_array_inh_exc__synaptic_pre;
extern std::vector<double> _dynamic_array_inh_exc_delay;
extern std::vector<int32_t> _dynamic_array_inh_exc_N_incoming;
extern std::vector<int32_t> _dynamic_array_inh_exc_N_outgoing;
extern std::vector<int32_t> _dynamic_array_inp_exc__synaptic_post;
extern std::vector<int32_t> _dynamic_array_inp_exc__synaptic_pre;
extern std::vector<double> _dynamic_array_inp_exc_delay;
extern std::vector<double> _dynamic_array_inp_exc_delay_1;
extern std::vector<double> _dynamic_array_inp_exc_lastupdate;
extern std::vector<int32_t> _dynamic_array_inp_exc_N_incoming;
extern std::vector<int32_t> _dynamic_array_inp_exc_N_outgoing;
extern std::vector<double> _dynamic_array_inp_exc_post1;
extern std::vector<double> _dynamic_array_inp_exc_post2;
extern std::vector<double> _dynamic_array_inp_exc_post2_before;
extern std::vector<double> _dynamic_array_inp_exc_pre;
extern std::vector<double> _dynamic_array_inp_exc_w;
extern std::vector<int32_t> _dynamic_array_spikes_i;
extern std::vector<double> _dynamic_array_spikes_t;

//////////////// arrays ///////////////////
extern double *_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double *_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern int64_t *_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t *_array_exc__spikespace;
extern const int _num__array_exc__spikespace;
extern double *_array_exc_ge;
extern const int _num__array_exc_ge;
extern double *_array_exc_gi;
extern const int _num__array_exc_gi;
extern int32_t *_array_exc_i;
extern const int _num__array_exc_i;
extern int32_t *_array_exc_inh_N;
extern const int _num__array_exc_inh_N;
extern double *_array_exc_lastspike;
extern const int _num__array_exc_lastspike;
extern char *_array_exc_not_refractory;
extern const int _num__array_exc_not_refractory;
extern double *_array_exc_theta;
extern const int _num__array_exc_theta;
extern double *_array_exc_v;
extern const int _num__array_exc_v;
extern int32_t *_array_inh__spikespace;
extern const int _num__array_inh__spikespace;
extern int32_t *_array_inh_exc_N;
extern const int _num__array_inh_exc_N;
extern double *_array_inh_ge;
extern const int _num__array_inh_ge;
extern int32_t *_array_inh_i;
extern const int _num__array_inh_i;
extern double *_array_inh_lastspike;
extern const int _num__array_inh_lastspike;
extern char *_array_inh_not_refractory;
extern const int _num__array_inh_not_refractory;
extern double *_array_inh_v;
extern const int _num__array_inh_v;
extern int32_t *_array_inp__spikespace;
extern const int _num__array_inp__spikespace;
extern int32_t *_array_inp_exc_N;
extern const int _num__array_inp_exc_N;
extern int32_t *_array_inp_i;
extern const int _num__array_inp_i;
extern double *_array_inp_rates;
extern const int _num__array_inp_rates;
extern int32_t *_array_spikes__source_idx;
extern const int _num__array_spikes__source_idx;
extern int32_t *_array_spikes_count;
extern const int _num__array_spikes_count;
extern int32_t *_array_spikes_N;
extern const int _num__array_spikes_N;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
extern double *_static_array__array_inp_rates;
extern const int _num__static_array__array_inp_rates;
extern double *_static_array__dynamic_array_inp_exc_w;
extern const int _num__static_array__dynamic_array_inp_exc_w;

//////////////// synapses /////////////////
// exc_inh
extern SynapticPathway exc_inh_pre;
// inh_exc
extern SynapticPathway inh_exc_pre;
// inp_exc
extern SynapticPathway inp_exc_post;
extern SynapticPathway inp_exc_pre;

// Profiling information for each code object
}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


