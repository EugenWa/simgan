import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from glob import glob
import os
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

        self.noise_percent = 10

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

    def create_unit_square_indexbased(self, fct, start_index, end_index):
        ftn = np.zeros(fct.shape[0])
        ftn[start_index:end_index] = 1
        return ftn

    def create_unit_square(self, fct, start, end):
        ftn = np.zeros(fct.shape[0])
        start_index = self.time > start
        end_index = self.time > end
        ftn[start_index:end_index] = 1
        return ftn


    def create_rnd_trigons(self, shape, spectrum, modify_amplitude=True, modify_frequency=True, use_bias=False, use_phase_shift=False):
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

        fkt_tensor  = np.zeros((samples, length, channels))

        for smpl in range(samples):
            for channel in range(channels):
                ampl, freq, bias, phi = self.generate_trigon_fct_parameters(spectrum, modify_amplitude, modify_frequency, use_bias, use_phase_shift)
                fkt_tensor[smpl, :, channel] = self.create_trigon_function('sin', freq, ampl, bias, phi)

        return fkt_tensor

    def generate_trigon_fct_parameters(self, frq_spec ,modify_amplitude=True, modify_frequency=True, use_bias=False, use_phase_shift=False):
        ampl = 1
        freq = 1
        bias = 0
        phi  = 0
        if modify_frequency:
            if frq_spec is self.f_spectrum['low']:
                freq = np.random.uniform(self.lower_freq_bound_min, self.lower_freq_bound_max)  # set new frequency in lower range
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
        population = range(fct.shape[0])
        gap_places = random.sample(population, number_of_gaps)

        function_noisy = np.zeros(fct.shape[0])
        function_noisy[:] = fct[:]              # copy
        function_noisy[gap_places] = 0
        return function_noisy

    def Noise_degrade_A(self, fct):
        noiscy_function = np.zeros(fct.shape)
        carrier_ampl, carrier_freq, bias, carrier_phi = self.generate_trigon_fct_parameters(self.f_spectrum['high'], False, True, False, True)
        ftn = self.create_trigon_function('cos', carrier_freq ,carrier_ampl, bias, carrier_phi)
        carrier_ampl, carrier_freq, bias, carrier_phi = self.generate_trigon_fct_parameters(self.f_spectrum['medium'],
                                                                                            False, True, False, True)
        ftn2 = self.create_trigon_function('cos', carrier_freq, carrier_ampl, bias, carrier_phi)

        # use this as new carrier
        amp_mod = self.Amplitude_Modulation(ftn2, ftn, 1)

        noiscy_function =  fct - (np.max(abs(fct))/(self.noise_percent*np.max(abs(amp_mod)))) * amp_mod
        return noiscy_function

    def Noise_degrade_F(self, fct):
        noiscy_function = np.zeros(fct.shape)
        carrier_ampl, carrier_freq, bias, carrier_phi = self.generate_trigon_fct_parameters(self.f_spectrum['low'], False, True, False, True)
        ftn1 = self.create_trigon_function('cos', carrier_freq ,carrier_ampl, bias, carrier_phi)
        carrier_ampl, carrier_freq, bias, carrier_phi = self.generate_trigon_fct_parameters(self.f_spectrum['low'], False, True, False, True)
        ftn2 = self.create_trigon_function('cos', carrier_freq, carrier_ampl, bias, carrier_phi)
        # use this as new carrier
        freq_mod = self.Freq_Modulation_Signal(ftn1, 1, ftn2)
        noiscy_function +=  fct + (np.max(abs(fct))/(self.noise_percent*np.max(abs(freq_mod)))) * freq_mod
        return noiscy_function



    def add_noise_to_tensor(self, fctensor, moise_mode=0):
        noisy_functions = np.zeros(fctensor.shape)
        for sample in range(fctensor.shape[0]):
            for channel in range(fctensor.shape[2]):
                if moise_mode is 0:
                    noisy_functions[sample, :, channel] = self.Noise_degrade_A(fctensor[sample, :, channel])
                elif moise_mode is 1:
                    noisy_functions[sample, :, channel] = self.Noise_degrade_F(fctensor[sample, :, channel])
                elif moise_mode is 2:
                    noisy_functions[sample, :, channel] = self.Noise_add_gap(fctensor[sample, :, channel], np.random.randint(0, 10))

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

    def load_B(self, type='B1'):
        return np.load(self.path_B + '/%s.npy' % type)


if __name__=='__main__':
    data_set_name = 'D1_fktsS'
    dataset_length = 10000
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
    config = np.array([dataset_length, interval, Dol.interval_length])
    np.save(Datasetpath + '/Datasets/%s/cfg' % data_set_name, config)


    ft = Dol.create_rnd_trigons((dataset_length, s_length, 1), Dol.f_spectrum['low'], False, True, False, True)#False, False, False, False)
    noisy1 = Dol.add_noise_to_tensor(ft, 0)
    noisy2 = Dol.add_noise_to_tensor(ft, 2)
    noisy3 = Dol.add_noise_to_tensor(noisy1, 2)
    noisy4 = Dol.add_noise_to_tensor(noisy1, 1)
    noisy5 = Dol.add_noise_to_tensor(noisy4, 2)


    np.save(path_A + '/A.npy',  ft)
    np.save(path_B + '/B1.npy', noisy1)             # just amps
    np.save(path_B + '/B2.npy', noisy2)             # just holes
    np.save(path_B + '/B3.npy', noisy3)             # amps + holes
    np.save(path_B + '/B4.npy', noisy4)             # amps + freq
    np.save(path_B + '/B5.npy', noisy5)             # amps + freq + holes

    #exit()
    tmp = Dol.time

    dl = OneD_Data_Loader(data_set_name)
    ft_loaded = dl.load_A()
    ft1 = ft[0, :, 0]
    #ft2 = Dol.Freq_Modulation_Signal(ft1, 1, 10)
    #ft3 = Dol.Noise_add_gap(ft[0, :, 0], 7)
    #ft4 = Dol.Amplitude_Modulation(ft[0, :, 0], Dol.create_trigon_function('cos', 10, 1, 0), 1)

    fit = plt.figure()
    plt.plot(tmp, ft[0, :, 0], color='b')
    plt.plot(tmp, ft_loaded[0, :, 0] + 1, color='r')
    #plt.plot(tmp, noisy1[0, :, 0], color='r')
    #plt.plot(tmp, noisy2[0, :, 0], color='g')
    #plt.plot(tmp, noisy3[0, :, 0], color='yellow')
    plt.plot(tmp, noisy4[0, :, 0], color='cyan')


    plt.show()

