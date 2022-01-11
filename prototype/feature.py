import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import simps
# import matplotlib.pyplot as plt
import pywt
import scipy.signal
from numpy import linalg as LA

# must choose channel from 14 channels: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
class Feature():

    def __init__(self, channels, readings, filename = "data.csv"):
        # name of file where data is stored along with extension
        self.filename = filename
        self.channels = channels
        self.readings = readings

        self.psd_values = []
        self.delta_values = []
        self.theta_values = []
        self.alpha_values = []
        self.beta_values = []

        arr = range(self.readings)
        self.feature = pd.DataFrame(arr, columns=['Redundant'])
        self.feature.to_csv('feature.csv')
        

    def psd(self):
        fs = 128

        data = pd.read_csv(self.filename)
        data.to_csv('data1.csv', header = False)
        data = pd.read_csv("data1.csv")
        # getting rid of garbage headers

        arr = range(self.readings)
        feature = pd.DataFrame(arr, columns=['Redundant'])
        feature.to_csv('feature.csv')

        for i in self.channels:
            # outer loop to iterate through channels

            indices = np.linspace(0, data[i].size, self.readings+1, dtype=int)
            for j in range(self.readings) :
                # inner loop that iterates through readings in each channel
                freqs, psd = signal.welch(data[i].iloc[indices[j] : indices[j+1]], fs)
                freq_res = freqs[1] - freqs[0]
                total_power = simps(psd, dx=freq_res)
                self.psd_values.append(total_power)
                print(self.psd_values)
                
            # self.save_to_file(i, self.feature)
            # self.psd_values = []


    def relative_psd(self):
        """
        function uses welch periodogram to get power spectral density and then uses composite simpsons rule to get spectral density for
        specified frequencies.

        Saves calculated values to a CSV file.
        By default calculates for alpha, beta, theta and gamma frequency bands for each channel.

        """
        # print(self.readings)
        fs = 128
        # t is the duration of the signal, this value must be same is record.py and stimuli/cues.py as well

        # N = fs * t

        win_length = 4*fs
        # window length is normally 2/Lowest freq of interest which is 0.5 in our case thus the window length is 4 seconds

        data = pd.read_csv(self.filename)
        data.to_csv('data1.csv', header = False)
        data = pd.read_csv("data1.csv")
        # getting rid of garbage headers

        arr = range(self.readings)
        feature = pd.DataFrame(arr, columns=['Redundant'])
        feature.to_csv('feature.csv')
        # creating a csv with redundant values which are later deleted.
        

        for i in self.channels:
            # outer loop to iterate through channels

            indices = np.linspace(0, data[i].size, self.readings+1, dtype=int)
            for j in range(self.readings) :
                # inner loop that iterates through readings in each channel
                freqs, psd = signal.welch(data[i].iloc[indices[j] : indices[j+1]], fs)
                # nperseg=win_length

                """
                plt.plot(freqs, psd, color='k', lw=2)
                plt.xlabel('Frequency')
                plt.ylabel('Power spectral density')
                plt.title("Welch's periodogram")
                """
                # delta band 
                low = 0.5
                high = 4
                delta = np.logical_and(freqs >= low, freqs <= high)
                freq_res = freqs[1] - freqs[0] 
                delta_power = simps(psd[delta], dx=freq_res)

                total_power = simps(psd, dx=freq_res)

                delta_rel_power = delta_power / total_power
                self.delta_values.append(delta_rel_power)

                # print('relative delta power = %.3f '% delta_rel_power)

                # theta band 
                low = 4
                high = 8
                theta = np.logical_and(freqs >= low, freqs <= high)
                freq_res = freqs[1] - freqs[0] 
                theta_power = simps(psd[theta], dx=freq_res)
                theta_rel_power = theta_power / total_power
                # print('relative theta power = %.3f '% theta_rel_power)
                self.theta_values.append(theta_rel_power)

                # alpha band 
                low = 8
                high = 12
                alpha = np.logical_and(freqs >= low, freqs <= high)
                freq_res = freqs[1] - freqs[0] 
                alpha_power = simps(psd[alpha], dx=freq_res)
                alpha_rel_power = alpha_power / total_power
                # print('relative alpha power = %.3f '% alpha_rel_power)
                self.alpha_values.append(alpha_rel_power)

                # beta band 
                low = 12
                high = 30
                beta = np.logical_and(freqs >= low, freqs <= high)
                freq_res = freqs[1] - freqs[0] 
                beta_power = simps(psd[beta], dx=freq_res)
                beta_rel_power = beta_power / total_power
                # print('relative beta power = %.3f '% beta_rel_power)
                self.beta_values.append(beta_rel_power)

            self.save_to_file(i, self.feature)
            self.delta_values = []
            self.theta_values = []
            self.alpha_values = []
            self.beta_values = []
            # resetting lists for next channel

        

        self.feature = self.feature.drop('Redundant', axis=1)
        self.feature.to_csv('feature.csv')
        

    def wavelet(self):
    
        data = pd.read_csv(self.filename)
        data.to_csv('data1.csv', header = False)
        data = pd.read_csv("data1.csv")
        for i in self.channels :
            sig = data[i]
            dec=pywt.wavedec(sig, wavelet = 'db4', level=4)
            print(i)
            print("******************")
            print(dec)
    
    def CSP(self):
        data = pd.read_csv("left.csv")
        data.to_csv('left1.csv', header = False)
        data = pd.read_csv("left1.csv")

        data1 = pd.read_csv("right.csv")
        data1.to_csv('right1.csv', header = False)
        data1 = pd.read_csv("right1.csv")


        ## code for average normalized spatial variance
        indices = np.linspace(0, data['EEG.AF3'].size, self.readings+1, dtype=int)
        sum1 = np.zeros((14,14))
        sum2 = np.zeros((14,14))
        for i in range(self.readings):
            XLeftdash = data[self.channels].iloc[indices[i] : indices[i+1]].to_numpy()
            XRightdash = data1[self.channels].iloc[indices[i] : indices[i+1]].to_numpy()
            XLeft = np.transpose(XLeftdash)
            XRight = np.transpose(XRightdash)
            product_left = np.matmul(XLeft,XLeftdash)
            product_right = np.matmul(XRight,XRightdash)
            #print(XLeftdash.shape)
            #print(XRightdash.shape)
            #print(XLeft.shape)
            #print(XRight.shape)

            # Normalized spatial covariance
            R_left = product_left / np.trace(product_left)
            R_right = product_right / np.trace(product_right)
            
            # Averaged Normalized spatial covariance
            sum1 = np.add(sum1,R_left)
            sum2 = np.add(sum2,R_right)
        R_left_avg = sum1/self.readings
        R_right_avg = sum2/self.readings
        R = R_left_avg + R_right_avg
        # code for whitening matrix
        evals, evecs = np.linalg.eigh(R)

        # Calculate whiteing matrix
        P = np.diag(evals**(-1/2)) @ evecs.T

        # Applying whitening matrix to Left data
        # z = np.diag(evals**(-1/2)) @ evecs.T @ XLeft
        # Applying whitening matrix to Right data
        # z1 = np.diag(evals**(-1/2)) @ evecs.T @ XRight

        S_left = P @ R_left_avg @ np.transpose(P)
        S_right = P @ R_right_avg @ np.transpose(P)

        evals, evecs = np.linalg.eigh(S_left)
        W = np.transpose(evecs) @ P
        W_dash = np.transpose(W)

        z = W @ XLeft
        z1 = W @ XRight
        
        mask = [False,False,False,False,False,False,False,False,False,False,False,False,False,False]
        for i in range(6):
            am = np.where(W_dash == np.max(W_dash))
            c_idx = am[0][0]
            r_idx = am[1][0]
            W_dash[c_idx,r_idx] = 0
            mask[r_idx] = True
        
        return mask

        # spatial_patterns = pd.DataFrame(W_dash)











    def save_to_file(self, channel, feature):
        # saves psd to csv
        feature[channel+'delta'] = self.delta_values
        feature[channel+'theta'] = self.theta_values
        feature[channel+'alpha'] = self.alpha_values
        feature[channel+'beta'] = self.beta_values

        # feature[channel+'PSD'] = self.psd_values
        
class Preprocess():

    def __init__(self, channels, filename = 'data.csv'):
        self.channels = channels
        self.filename = filename

        self.data = pd.read_csv(self.filename)
        self.data.to_csv('data1.csv', header = False)
        self.data = pd.read_csv("data1.csv")

    
    def bandpass_coeff(self, low, high, fs, order):
        """
        Function to calculate filter coefficients
        """
        nyq = 0.5*fs
        low = low / nyq
        high = high / nyq
        sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
        return sos
    def bandstop_coeff(self, low, high, fs, order):
        """
        Function to calculate filter coefficients
        """
        nyq = 0.5*fs
        low = low / nyq
        high = high / nyq
        sos = signal.butter(order, [low, high], btype='bandstop', output='sos')
        return sos
    def lowpass_coeff(self, high, fs, order):
        """
        Function to calculate filter coefficients
        """
        nyq = 0.5*fs
        high = high / nyq
        sos = signal.butter(order, high, btype='lowpass', output='sos')
        return sos
    def highpass_coeff(self, low, fs, order):
        """
        Function to calculate filter coefficients
        """
        nyq = 0.5*fs
        low = low / nyq
        sos = signal.butter(order, low, btype='highpass', output='sos')
        return sos
    
    def apply_filter(self,choice,order,fs,freq):
        list = freq.split()
        f1 = int(list[0])

        sos = []
        if choice == 'lowpass':
            sos = self.lowpass_coeff(f1,fs,order)
        elif choice == 'highpass':
            sos = self.highpass_coeff(f1,fs,order)
        elif choice == 'bandpass':
            f2 = int(list[1])
            sos = self.bandpass_coeff(f1,f2,fs,order)
        elif choice == 'bandstop':
            f2 = int(list[1])
            sos = self.bandstop_coeff(f1,f2,fs,order)
        
        
        for i in self.channels:
            filtered = signal.sosfilt(sos, self.data[i])
            print(filtered)

    
    # def save_to_csv(self,channel, filtered_eeg):



        
        



    
