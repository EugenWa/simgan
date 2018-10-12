import numpy as np
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from glob import glob
import os
import sys
import random


class Data_Generator_1D:
    def __init__(self, interval, interval_length, max_frequency=100):
        self.functionpool_trigon  = {'sin':np.sin, 'cos':np.cos, 'e':np.exp}
        self.functionpool_Polyn   = {'quad':lambda x:x**2, 'cube':lambda x:x**3, 'p4':lambda x:x**4, 'p5':lambda x:x**5, 'p6':lambda x:x**6}
        self.functionpool_Mult_in = {'exp':lambda x, y:x**y}
        self.trigon_keys = ['sin', 'cos', 'e']

        self.f_spectrum = {'low':0, 'medium':1, 'high':2}
        self.lower_freq_bound_min = 0.5
        self.lower_freq_bound_max = 5
        self.high_freq_bound_min  = 10
        self.high_freq_bound_max  = max_frequency
        self.amplitude_range = [0.7, 9]
        self.abs_bias       = 2

        self.noise_percent = 5                  # hence sinusoidals its *2
        self.Amp_Modulationn_amplitude = 0.2

        # Gap settings
        self.max_GAP_lenght = 11
        self.avarage_gap_length = 5
        self.gap_std = 3

        # sptr settings
        self.max_slope = 1/interval_length * 2
        self.min_slope = 0.3*self.max_slope
        self.slope_change_number = 14
        self.slope_chane_min_length = int(interval_length / self.slope_change_number * 0.5)
        self.slope_chane_max_length = int(interval_length / self.slope_change_number * 2)


        self.nyquist_f = 2*max_frequency + 1
        self.interval_length = interval_length
        if self.interval_length < self.nyquist_f:
            print('Extending sampling rate ')
            max_quad = 0
            while 2**max_quad < self.nyquist_f:
                max_quad += 1
            self.interval_length = 2**max_quad #self.nyquist_f    # guarantees 'perfect' sampling
            # make sure the intervallenght is an exp of 2
        # x-Axis:
        self.time = np.arange(0, interval, interval / interval_length)


        self.unit_step_idx_lenght_min = int(self.interval_length * 0.08)
        self.unit_step_idx_lenght_max = int(self.interval_length * 0.7)
        self.population_len = range(self.unit_step_idx_lenght_min, self.unit_step_idx_lenght_max)


    def add_noise_everywhere(self, fct, std, max_cap, mue=0, gaussian=True):
        """

        :param fct:
        :param std:
        :param max_cap:     Important Note: Crops the Noise in a symmetric manner around mue
        :param mue:
        :return:            Noise function AND the added Noise vector
        """
        function_noisy = np.zeros(fct.shape[0])
        function_noisy[:] = fct[:]  # copy

        if gaussian:
            noise_vector = np.random.normal(mue, std, fct.shape[0])
        else:
            std = std*2
            noise_vector = np.random.uniform(mue-std, std+mue, fct.shape[0])
        noise_vector[noise_vector > (max_cap + mue)]    = mue +max_cap              # crops the noise vector
        noise_vector[noise_vector < (mue-max_cap)]      = mue-max_cap               # crops the noise vector in the negative section

        function_noisy += noise_vector

        return function_noisy, noise_vector

    def add_gaussian_unf_noise_edges(self, fct, std , max_cap, mue=0, gaussian=True, overlapping=True):
        ### assumptions ###
        # 1. the noise width is a camera parameter and hence can be set as constant
        function_noisy = np.zeros(fct.shape[0])
        function_noisy[:] = fct[:]  # copy

        noise_len = int(self.unit_step_idx_lenght_min*0.3)
        if overlapping:
            noise_ov = int(noise_len*0.4)
        else:
            noise_ov = 0
        edges = np.zeros(fct.shape[0])
        edges[0] = 0
        for i in range(1, fct.shape[0]-1):
            edges[i] = (- fct[i-1] +  fct[i])

        tmp = np.where(np.abs(edges) > 0.9)[0]

        # create_deg_noise
        noise = np.zeros((len(tmp), noise_len+noise_ov))
        for j in range(noise.shape[0]):
            for i in range(noise.shape[1]):
                if gaussian:
                    noise[j, i] = np.random.normal(mue, std*np.exp(-i*0.05))
                else:
                    noise[j, i] = np.random.uniform(mue-(std * np.exp(-i * 0.05)), mue + (std * np.exp(-i * 0.05)))
            noise[j][noise[j] > (max_cap + mue)] = mue + max_cap  # crops the noise vector
            noise[j][noise[j] < (mue - max_cap)] = mue - max_cap  # crops the noise vector in the negative section

        for i in range(len(tmp)):
            idx=tmp[i]
            if np.sign(edges[idx]) < 0:
                noise_tmp = noise[i][::-1]
                function_noisy[idx-noise_len:idx] += noise_tmp[:noise_len]
                function_noisy[idx:idx+noise_ov] += np.ones(noise_ov) + noise_tmp[noise_len:]
            else:
                function_noisy[idx-noise_ov:idx] += np.ones(noise_ov) + noise[i][:noise_ov]
                function_noisy[idx:idx + noise_len] += noise[i][noise_ov:]

        return function_noisy


    def create_unit_step_tensor(self, shape, drawing_params):
        samples = shape[0]
        length = self.interval_length
        channels = shape[2]

        fkt_tensor = np.zeros((samples, length, channels))
        fkt_params = np.zeros((samples, 3, channels))
        for smpl in range(samples):
            for channel in range(channels):
                fkt_tensor[smpl, :, channel], fkt_params[smpl, :, channel] = self.create_unit_square(drawing_params)
        return fkt_tensor, fkt_params

    def create_unit_square(self, drawing_params):
        mean_len, len_std = drawing_params
        #l = random.sample(self.population_len, 1)
        l = int(np.random.normal(mean_len, len_std))
        # trin
        if l < self.unit_step_idx_lenght_min:
            l = self.unit_step_idx_lenght_min
        if l > self.unit_step_idx_lenght_max:
            l = self.unit_step_idx_lenght_max
        start = random.sample(range(self.unit_step_idx_lenght_min, self.interval_length - l - self.unit_step_idx_lenght_min), 1)
        params_set = start[0], start[0] + l, l
        return self.create_unit_square_indexbased(start[0], start[0] + l), params_set

    def create_unit_square_indexbased(self, start_index, end_index):
        ftn = np.zeros(self.interval_length)
        ftn[start_index:end_index] = 1
        return ftn

    def step_times_trigon(self, fct, step):
        return np.multiply(fct, step)

    def create_unit_square_timebased(self, start, end):
        ftn = np.zeros(self.interval_length)
        start_index = self.time > start
        end_index = self.time > end
        ftn[start_index:end_index] = 1
        return ftn

    def Noise_degrade_step(self, fct, sqr_1):
        #sqr_1 = self.create_unit_square()
        return self.step_times_trigon(fct, sqr_1)


    def create_sptr_tensor(self, shape, draw_parameters, slope_change_mode, jump_type, number_of_islands = 1):
        samples = shape[0]
        length = self.interval_length
        channels = shape[2]

        fkt_tensor = np.zeros((samples, length, channels))
        fkt_params = np.zeros((samples, number_of_islands * 3, channels))
        for smpl in range(samples):
            for channel in range(channels):
                fkt_tensor[smpl, :, channel], slope_change_spots = self.create_sptr_function(draw_parameters, slope_change_mode)
                if number_of_islands == 1:
                    jump_spots_population = range(int(self.slope_change_number * 0.1), len(slope_change_spots)-int(self.slope_change_number*0.4))
                    jump_spot_start = random.sample(jump_spots_population, 1)[0]
                    jump_spot_end  = random.sample(range(jump_spot_start + int(self.slope_change_number*0.1), len(slope_change_spots)-int(self.slope_change_number*0.2)), 1)[0]
                    jump_spots = [slope_change_spots[jump_spot_start], slope_change_spots[jump_spot_end]]
                else:
                    jump_spots = sorted(random.sample(slope_change_spots[1:-1], number_of_islands*2))
                if jump_type is 0:
                    for i in range(number_of_islands):
                        fkt_params[smpl, i*3:(i+1)*3, channel] = jump_spots[i*2], jump_spots[(i*2)+1]+self.slope_chane_min_length, jump_spots[(i*2)+1]+self.slope_chane_min_length-jump_spots[i*2]
                        steps = self.create_unit_square_indexbased(jump_spots[i*2], jump_spots[(i*2)+1]+self.slope_chane_min_length)
                        fkt_tensor[smpl, :, channel] = fkt_tensor[smpl, :, channel] + steps
                elif jump_type is 1:
                    jump_spots = sorted(random.sample(jump_spots, number_of_islands * 2))
                    all_steps = np.zeros(length)
                    for i in range(number_of_islands):
                        fkt_params[smpl, i * 3:(i + 1) * 3, channel] = jump_spots[i * 2], jump_spots[(i * 2) + 1], jump_spots[(i * 2) + 1]-jump_spots[i * 2]
                        all_steps += self.create_unit_square_indexbased(jump_spots[i * 2], jump_spots[(i * 2) + 1])
                    fkt_tensor[smpl, :, channel] += all_steps
                    fkt_tensor[smpl, :, channel] = self.step_times_trigon(fkt_tensor[smpl, :, channel], all_steps)

        return fkt_tensor, fkt_params

    def create_sptr_function(self, drawing_params, slope_change_mode=0):
        slope_change_number_mean, slope_change_number_std = drawing_params
        ftn = np.zeros(self.interval_length)
        slope_changes = [0]

        slope_change_number = int(np.random.normal(slope_change_number_mean, slope_change_number_std))
        if slope_change_number < self.slope_change_number - 4:
            slope_change_number = self.slope_change_number - 4
        if slope_change_number > self.slope_change_number + 4:
            slope_change_number = self.slope_change_number + 4
        self.slope_chane_min_length = int(self.interval_length / slope_change_number * 0.5)
        self.slope_chane_max_length = int(self.interval_length / slope_change_number * 2)

        slope_population_draw = range(self.slope_chane_min_length, self.slope_chane_max_length)
        for i in range(slope_change_number):
            step_change = slope_changes[-1] + random.sample(slope_population_draw, 1)[0]
            if step_change < self.interval_length:
                slope_changes.append(step_change)
        slope_direction = -1

        slope = np.random.uniform(self.min_slope, self.max_slope)
        for i in range(1, self.interval_length):
            if i in slope_changes:
                slope = np.random.uniform(self.min_slope, self.max_slope)
                if slope_change_mode is 0:
                    slope_direction = slope_direction * (-1)
                else:
                    slope_direction = random.choice([-1, 1])


            ftn[i] = ftn[i-1] + (slope*slope_direction)

        return ftn, slope_changes


    def create_rnd_trigons(self, shape, spectrum, drawing_parameters, modify_amplitude=True, modify_frequency=True, use_bias=False, use_phase_shift=False):
        """
            Create a Tensor of random sinusoidal's
        :param shape:
        :param spectrum:
        :param modify_amplitude:
        :param modify_frequency:
        :param use_bias:
        :param use_phase_shift:
        :return:
        """
        samples     = shape[0]
        length      = self.interval_length
        channels    = shape[2]

        fkt_tensor   = np.zeros((samples, length, channels))
        fkt_features = np.zeros((samples, 4, channels))         # to save amplitude, frequency, bias, and phi

        for smpl in range(samples):
            for channel in range(channels):
                ampl, freq, bias, phi = self.generate_trigon_fct_parameters(spectrum, drawing_parameters, modify_amplitude, modify_frequency, use_bias, use_phase_shift)
                fkt_tensor[smpl, :, channel] = self.create_trigon_function('sin', freq, ampl, bias, phi)
                # save features:
                fkt_features[smpl, :, channel] = ampl, freq, bias, phi

        return fkt_tensor, fkt_features

    def generate_trigon_fct_parameters(self, frq_spec, drawing_parameters, modify_amplitude=True, modify_frequency=True, use_bias=False, use_phase_shift=False):
        ampl = 1
        freq = 1
        bias = 0
        phi  = 0

        if modify_frequency:
            if frq_spec is self.f_spectrum['low']:
                if len(drawing_parameters) is 0:
                    freq = np.random.uniform(self.lower_freq_bound_min, self.lower_freq_bound_max)  # set new frequency in lower range
                else:
                    freq_mean, freq_std = drawing_parameters
                    freq = np.random.normal(freq_mean, freq_std)  # set new frequency in lower range
                    # trim:
                    if freq < self.lower_freq_bound_min:
                        freq = self.lower_freq_bound_min
                    if freq > self.lower_freq_bound_max:
                        freq = self.lower_freq_bound_max

            elif frq_spec is self.f_spectrum['medium']:
                freq = np.random.uniform(self.high_freq_bound_min, self.high_freq_bound_max/4)  # set new frequency in mid range
            elif frq_spec is self.f_spectrum['high']:
                freq = np.random.uniform(self.high_freq_bound_min,self.high_freq_bound_max)     # set new frequency in high range
        if modify_amplitude:
            ampl = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])
        if use_bias:
            bias = np.random.uniform(-self.abs_bias, self.abs_bias)
        if use_phase_shift:
            phi = np.random.uniform(0, np.pi * 2)

        return ampl, freq, bias, phi
    def create_trigon_function(self, fkt_id, frequency, ampl, bias, phase_shift=0):
        """
            Creates a trigonomatric function
        :param fkt_id:      Type (cos, sin ..)
        :param frequency:   frequency
        :param ampl:        Amplitude
        :param bias:        bias
        :param phase_shift: Phase shift phi
        :return:            Trigonometric function with the set parameters
        """
        return ampl*self.functionpool_trigon[fkt_id](self.time * frequency + phase_shift) + bias

    def Amplitude_Modulation(self, signal, carrier, carrier_amp):
        """
            Amplitude Modulation
        :param signal:          Signal which will shaping the wave of the carrier
        :param carrier:         Carrier signal, must have Amplitude 1
        :param carrier_amp:     Amplitude of the carrier, hence carrier amp 1
        :return:                Amplitude Modulated Signal
        """
        return (carrier_amp + signal) * carrier


    def Freq_Modulation_Signal(self, signal, carrier_Amp, carrier_freq, initial_carrier_shift=0, carrier_type='cos' ):
        """
            Baisc Frequency Modulation
        :param signal:
        :param carrier_Amp:
        :param carrier_freq:
        :param initial_carrier_shift:
        :param carrier_type:
        :return:
        """
        signal_integrated = np.zeros(signal.shape[0])
        signal_integrated[0] = signal[0]
        for i in range(1, signal.shape[0]):
            signal_integrated[i] = signal[i] + signal_integrated[i - 1]

        return carrier_Amp*self.functionpool_trigon[carrier_type](self.time * carrier_freq + signal_integrated + initial_carrier_shift)


    def Noise_add_gap(self, fct, number_of_gaps):
        """
            Adds holes to a copy the function
        :param fct:             Function
        :param number_of_gaps:  Amount of holes
        :return:                Copy of the passed function with gaps
        """
        # sample gaps from the whole range of the function uniformly
        population = range(fct.shape[0]-self.max_GAP_lenght)
        gap_places = random.sample(population, number_of_gaps)

        function_noisy = np.zeros(fct.shape[0])
        function_noisy[:] = fct[:]              # copy

        #function_noisy[gap_places] = 0
        for gap_location in gap_places:
            gapwidth = int(random.normalvariate(self.avarage_gap_length, self.gap_std))
            if gapwidth > self.max_GAP_lenght:
                gapwidth = self.max_GAP_lenght
            elif gapwidth < 1:
                gapwidth = 1
            function_noisy[gap_location:gap_location+gapwidth] = 0
        return function_noisy

    def Noise_degrade_A(self, fct, max_changing_reate=1.0):
        noiscy_function = np.zeros(fct.shape)
        carrier_ampl, carrier_freq, bias, carrier_phi = self.generate_trigon_fct_parameters(self.f_spectrum['high'], [], False, True, False, True)
        ftn = self.create_trigon_function('cos', carrier_freq , carrier_ampl, bias, carrier_phi)
        carrier_ampl, carrier_freq, bias, carrier_phi = self.generate_trigon_fct_parameters(self.f_spectrum['medium'],[], False, True, False, True)
        ftn2 = self.create_trigon_function('cos', carrier_freq, carrier_ampl, bias, carrier_phi)

        # use this as new carrier
        amp_mod = self.Amplitude_Modulation(ftn2, ftn, self.Amp_Modulationn_amplitude)

        noiscy_function =  fct - (max_changing_reate * np.max(abs(fct))/(self.noise_percent*np.max(abs(amp_mod)))) * amp_mod
        return noiscy_function

    def Noise_degrade_F(self, fct, max_changing_reate=1.0):
        noiscy_function = np.zeros(fct.shape)
        carrier_ampl, carrier_freq, bias, carrier_phi = self.generate_trigon_fct_parameters(self.f_spectrum['low'], [], False, True, False, True)
        ftn1 = self.create_trigon_function('cos', carrier_freq ,carrier_ampl, bias, carrier_phi)
        carrier_ampl, carrier_freq, bias, carrier_phi = self.generate_trigon_fct_parameters(self.f_spectrum['low'], [], False, True, False, True)
        ftn2 = self.create_trigon_function('cos', carrier_freq, carrier_ampl, bias, carrier_phi)
        # use this as new carrier
        freq_mod = self.Freq_Modulation_Signal(ftn1,  1, ftn2)
        noiscy_function +=  fct + (max_changing_reate * np.max(abs(fct))/(self.noise_percent*np.max(abs(freq_mod)))) * freq_mod
        return noiscy_function



    def add_noise_to_tensor(self, fctensor, moise_mode=0, max_change_rate=1.0, std=0.033, gauusina_type=True, overlapping=True):
        noisy_functions = np.zeros(fctensor.shape)
        for sample in range(fctensor.shape[0]):
            for channel in range(fctensor.shape[2]):
                if moise_mode is 0:
                    noisy_functions[sample, :, channel] = self.Noise_degrade_A(fctensor[sample, :, channel], max_change_rate)
                elif moise_mode is 1:
                    noisy_functions[sample, :, channel] = self.Noise_degrade_F(fctensor[sample, :, channel], max_change_rate)
                elif moise_mode is 2:
                    noisy_functions[sample, :, channel] = self.Noise_add_gap(fctensor[sample, :, channel], np.random.randint(0, 10))
                elif moise_mode is 4:
                    noisy_functions[sample, :, channel], _ = self.add_noise_everywhere(fctensor[sample, :, channel], std, max_change_rate, 0, gauusina_type)
                elif moise_mode is 5:
                    noisy_functions[sample, :, channel] = self.add_gaussian_unf_noise_edges(fctensor[sample, :, channel], std, max_change_rate, 0, gauusina_type, overlapping)

        return noisy_functions

    def overlap_sqrs_and_steps(self, fctensor):
        noisy_functions = np.zeros(fctensor[0].shape)
        for sample in range(fctensor[0].shape[0]):
            for channel in range(fctensor[0].shape[2]):
                noisy_functions[sample, :, channel] = self.Noise_degrade_step(fctensor[0][sample, :, channel], fctensor[1][sample, :, channel])
        return noisy_functions


class OneD_Data_Loader:
    def __init__(self, dataset_name):
        Datasetpath = os.path.dirname(os.path.abspath(__file__))
        Datasetpath = Datasetpath[0:-(len("Datamanager") + 1)]

        self.path_A = Datasetpath + '/Datasets/%s/A' % dataset_name
        self.path_B = Datasetpath + '/Datasets/%s/B' % dataset_name

        # load config
        self.config = np.load(Datasetpath + '/Datasets/%s/cfg.npy' % dataset_name)

    def load_A(self):
        return np.load(self.path_A + '/A.npy')

    def load_A_Feat(self):
        return np.load(self.path_A + '/FEAT.npy')


    def load_B(self, Type='B'):
        return np.load(self.path_B + '/%s.npy' % Type)

    def Load_Data_Tensors(self, degraded_data_set, validation_split, test_split=0.1):
        """
        Load Training-Set, Validation-Set and Test-Set
        :param degraded_data_set:       Name of the Degraded Dataset
        :param validation_split:        How to split Validation | Test
        :param feat_only:               Loads in Only Features as Target-Set
        :param test_split:              How to split all Data into Train | (Val, Test)
        :return:                        Train-, Validation-, Test-Set's A&B
        """
        # load Data --------------------------------------------------------
        xA_train = self.load_A()
        xB_train = self.load_B(degraded_data_set)

        xA_test = xA_train[0:int(xA_train.shape[0] * test_split)]
        xB_test = xB_train[0:int(xB_train.shape[0] * test_split)]

        xA_train = xA_train[int(xA_train.shape[0] * test_split):]
        xB_train = xB_train[int(xB_train.shape[0] * test_split):]

        xa_val = xA_test[0:int(xA_test.shape[0] * validation_split)]
        xb_val = xB_test[0:int(xB_test.shape[0] * validation_split)]

        xA_test = xA_test[int(xA_test.shape[0] * validation_split):]
        xB_test = xB_test[int(xB_test.shape[0] * validation_split):]
        # /load Data --------------------------------------------------------

        return xA_train, xB_train, xa_val, xb_val, xA_test, xB_test

    def Load_Data_Tensors_WFeat(self, degraded_data_set, validation_split, target_mode=0, test_split=0.1):
        # load Data --------------------------------------------------------
        xA_train = self.load_A()
        if target_mode is 0:
            xA_train_FEAT = np.zeros((xA_train.shape[0], ))
        else:
            xA_train_FEAT = self.load_A_Feat()
        xB_train = self.load_B(degraded_data_set)

        xA_test = xA_train[0:int(xA_train.shape[0] * test_split)]
        xA_test_FEAT = xA_train_FEAT[0:int(xA_train_FEAT.shape[0] * test_split)]
        xB_test = xB_train[0:int(xB_train.shape[0] * test_split)]

        xA_train = xA_train[int(xA_train.shape[0] * test_split):]
        xA_train_FEAT = xA_train_FEAT[int(xA_train_FEAT.shape[0] * test_split):]
        xB_train = xB_train[int(xB_train.shape[0] * test_split):]

        xa_val = xA_test[0:int(xA_test.shape[0] * validation_split)]
        xa_val_feat = xA_test_FEAT[0:int(xA_test_FEAT.shape[0] * validation_split)]
        xb_val = xB_test[0:int(xB_test.shape[0] * validation_split)]

        xA_test = xA_test[int(xA_test.shape[0] * validation_split):]
        xA_test_FEAT = xA_test_FEAT[int(xA_test_FEAT.shape[0] * validation_split):]
        xB_test = xB_test[int(xB_test.shape[0] * validation_split):]
        # /load Data --------------------------------------------------------

        # combine xA's in the form: [xA_train, xA_feat]
        if target_mode is 1:
            xA_train_full   = [xA_train, xA_train_FEAT]
            xA_val_full     = [xa_val, xa_val_feat]
            xA_test_full    = [xA_test, xA_test_FEAT]
        elif target_mode is 2:
            xA_train_full   = xA_train_FEAT
            xA_val_full     = xa_val_feat
            xA_test_full    = xA_test_FEAT
        else:
            xA_train_full   = xA_train
            xA_val_full     = xa_val
            xA_test_full    = xA_test
        return xA_train, xA_train_full, xB_train, xa_val, xA_val_full, xb_val, xA_test, xA_test_full, xB_test


if __name__=='__main__':
    data_set_name = sys.argv[1]#
    mode          = int(sys.argv[2])
    dataset_length = 12#000
    # generate Data_set
    Datasetpath = os.path.dirname(os.path.abspath(__file__))
    Datasetpath = Datasetpath[0:-(len("Datamanager") + 1)]

    path_A = Datasetpath + '/Datasets/%s/A' % data_set_name
    path_B = Datasetpath + '/Datasets/%s/B' % data_set_name
    os.makedirs(path_A, exist_ok=True)
    os.makedirs(path_B, exist_ok=True)


    interval = np.pi * 2
    s_length = 2**10
    Dol = Data_Generator_1D(interval, s_length)
    print(Dol.interval_length)
    config = np.array([dataset_length, interval, Dol.interval_length])
    np.save(Datasetpath + '/Datasets/%s/cfg' % data_set_name, config)

    #                               Draw
    # -------------------------------------------------------------------------
    # --- freq ---
    freq_middle = (Dol.lower_freq_bound_max + Dol.lower_freq_bound_min) / 2
    freq_dist   = Dol.lower_freq_bound_max - freq_middle

    freq_sigma = (freq_dist * 2) / 9                                            # because: max distance (3 sigma) should intersect in the middle -> middle + 3*3/2*sigma = dist
    freq_middle += freq_sigma*(3/2)                                             # this 3/2 will be replaced in order to create divergent 'source' data!!!!!!!!!!!!!!!!!!!!!
                                                                                # with 0 and minus for instance
    lower_freq_mean = freq_middle

    # --- sqrs ---
    sqrs_len_middle = (Dol.unit_step_idx_lenght_max + Dol.unit_step_idx_lenght_min) / 2
    sqrs_len_dist   = Dol.unit_step_idx_lenght_max - sqrs_len_middle

    sqrs_sigma = (sqrs_len_dist * 2) / 9
    sqrs_len_middle += sqrs_sigma*(3/2)
    sqrs_len_mean = sqrs_len_middle

    # --- slopes ---
    slope_num_middle = 14
    slope_num_dist = 4
    slope_num_sigma = (slope_num_dist * 2) / 9
    slope_num_middle += slope_num_sigma * (3 / 2)
    slope_num_mean = slope_num_middle

    draw_params_trig = [lower_freq_mean, freq_sigma]
    draw_params_sqrs = [sqrs_len_mean,   sqrs_sigma]
    draw_parameters_slopes = [slope_num_mean, slope_num_sigma]
    # -------------------------------------------------------------------------

    # Degraded Dataset's will follow following pattern:
    # 1. Gaussian Noise
    # 1. Uniform  Noise
    # 2. Amplitudes + Frequency Modulated
    # 3. 1 + Holes
    # 4. 2 + Holes
    # remark: 1-3 will keep in mind the special Attributes of the Data
    noise_gaussian = True
    if mode is 0:                       # Sins
        ft, features = Dol.create_rnd_trigons((dataset_length, s_length, 1), Dol.f_spectrum['low'], draw_params_trig, False, True, False, True)#False, False, False, False)

        max_change_rate =0.4
        noisy1          = Dol.add_noise_to_tensor(ft, 4, max_change_rate, 0.033, noise_gaussian)
        noisy_amps      = Dol.add_noise_to_tensor(ft, 0, max_change_rate*0.4)
        noisy2          = Dol.add_noise_to_tensor(noisy_amps, 1, max_change_rate)
        noisy3          = Dol.add_noise_to_tensor(noisy1, 2)
        noisy4          = Dol.add_noise_to_tensor(noisy2, 2)

    elif mode is 1:                     # Sqrs
        ft, features = Dol.create_unit_step_tensor((dataset_length, s_length, 1), draw_params_sqrs)

        max_change_rate = 0.2
        noisy1      = Dol.add_noise_to_tensor(ft, 4, max_change_rate, 0.033, noise_gaussian)
        noisy_amps  = Dol.add_noise_to_tensor(ft, 0, max_change_rate * 0.4)
        noisy2      = Dol.add_noise_to_tensor(noisy_amps, 1, max_change_rate)
        noisy3      = Dol.add_noise_to_tensor(noisy1, 2)
        noisy4      = Dol.add_noise_to_tensor(noisy2, 2)
    elif mode is 2:                     # Sins inside Sqrs-islands
        obje_ct_shapes = (dataset_length, s_length, 1)
        ft_trig, feat_trig = Dol.create_rnd_trigons(obje_ct_shapes, Dol.f_spectrum['low'], draw_params_trig, False, True, False, True)  # False, False, False, False)
        ft_stps, feat_sqrs = Dol.create_unit_step_tensor(obje_ct_shapes, draw_params_sqrs)

        ft = Dol.overlap_sqrs_and_steps([ft_trig, ft_stps])
        features = np.zeros((ft.shape[0], feat_trig.shape[1] + feat_sqrs.shape[1], ft.shape[2]))
        for ft_sample in range(obje_ct_shapes[0]):
            for ft_channel in range(obje_ct_shapes[2]):
                features[ft_sample, 0:feat_trig.shape[1], ft_channel] = feat_trig[ft_sample,:, ft_channel]
                features[ft_sample, feat_trig.shape[1]:,  ft_channel] = feat_sqrs[ft_sample, :, ft_channel]

        max_change_rate = 0.2
        noisy1 = Dol.add_noise_to_tensor(ft, 4, max_change_rate, 0.033, noise_gaussian)
        noisy_amps = Dol.add_noise_to_tensor(ft, 0, max_change_rate * 0.4)
        noisy2 = Dol.add_noise_to_tensor(noisy_amps, 1, max_change_rate)
        noisy3 = Dol.add_noise_to_tensor(noisy1, 2)
        noisy4 = Dol.add_noise_to_tensor(noisy2, 2)

    elif mode is 3:                         # Sptr everywhere
        ft, features = Dol.create_sptr_tensor((dataset_length, s_length, 1), draw_parameters_slopes, 0, 0, 1)

        max_change_rate = 0.2
        noisy1          = Dol.add_noise_to_tensor(ft, 4, max_change_rate, 0.033, noise_gaussian)
        noisy_amps      = Dol.add_noise_to_tensor(ft, 0, max_change_rate * 0.4)
        noisy2          = Dol.add_noise_to_tensor(noisy_amps, 1, max_change_rate)
        noisy3          = Dol.add_noise_to_tensor(noisy1, 2)
        noisy4          = Dol.add_noise_to_tensor(noisy2, 2)
    elif mode is 4:                         # Sptr island
        ft, features = Dol.create_sptr_tensor((dataset_length, s_length, 1), draw_parameters_slopes, 0, 1, 1)

        max_change_rate = 0.2
        noisy1 = Dol.add_noise_to_tensor(ft, 4, max_change_rate, 0.033, noise_gaussian)
        noisy_amps = Dol.add_noise_to_tensor(ft, 0, max_change_rate * 0.4)
        noisy2 = Dol.add_noise_to_tensor(noisy_amps, 1, max_change_rate)
        noisy3 = Dol.add_noise_to_tensor(noisy1, 2)
        noisy4 = Dol.add_noise_to_tensor(noisy2, 2)
    elif mode is 5:                         # on borders gaussian
        ft, features = Dol.create_unit_step_tensor((dataset_length, s_length, 1), draw_params_sqrs)

        max_change_rate = 0.1
        noisy1 = Dol.add_noise_to_tensor(ft, 5, max_change_rate, 0.044, True, True)
        noisy_amps = Dol.add_noise_to_tensor(ft, 0, max_change_rate * 0.4)
        noisy2 = Dol.add_noise_to_tensor(noisy_amps, 1, max_change_rate)
        noisy3 = Dol.add_noise_to_tensor(noisy1, 2)
        noisy4 = Dol.add_noise_to_tensor(noisy2, 2)

    np.save(path_A + '/A.npy',  ft)
    np.save(path_A + '/FEAT.npy',  features)
    np.save(path_B + '/B1.npy', noisy1)
    np.save(path_B + '/B2.npy', noisy2)
    np.save(path_B + '/B3.npy', noisy3)
    np.save(path_B + '/B4.npy', noisy4)
    #np.save(path_B + '/B5.npy', noisy5)

    #exit()
    tmp = Dol.time

    dl = OneD_Data_Loader(data_set_name)
    ft_loaded = dl.load_A()
    ft1 = ft[0, :, 0]
    #ft2 = Dol.Freq_Modulation_Signal(ft1, 1, 10)
    #ft3 = Dol.Noise_add_gap(ft[0, :, 0], 7)
    #ft4 = Dol.Amplitude_Modulation(ft[0, :, 0], Dol.create_trigon_function('cos', 10, 1, 0), 1)

    fit = plt.figure()
    plt.plot(tmp, noisy3[0, :, 0], color='r')
    #plt.plot(tmp, noisy4[0, :, 0], color='g')
    plt.plot(tmp, ft[0, :, 0], color='b')

    fit = plt.figure()
    plt.plot(tmp, noisy2[0, :, 0], color='r')
    # plt.plot(tmp, noisy4[0, :, 0], color='g')
    plt.plot(tmp, ft[0, :, 0], color='b')

    #plt.plot(tmp, noisy3[0, :, 0], color='g')
    #plt.plot(tmp, noisy2[0, :, 0], color='g')
    #plt.plot(tmp, noisy3[0, :, 0], color='yellow')
    #plt.plot(tmp, noisy4[0, :, 0], color='cyan')


    plt.show()

