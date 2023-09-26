import numpy as np
from numpy import sqrt
from numpy.random import rand, randn
import matplotlib.pyplot as plt
from scipy import special
import pandas as pd

# Create a 2x3 matrix
matrix_XOR = np.array([[1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]])

# Create a 3x3 identity matrix
identity_matrix = np.eye(4)

#Create the generator matrix 
G = np.concatenate((identity_matrix, matrix_XOR), 1)

def encode_hamming(data_bits): 
    #Multiply data bits with generator matrix mod 2 
    x = np.dot(data_bits, G) % 2
    return x 

def convert_to_hamming(bitstream): 
    hamming_encoded = []
    #Loop through 4 bits at a time, add parity bits, return the final enocoding 
    for i in range(0, len(bitstream), 4): 
        data_bits = bitstream[i:i+4]
        if len(data_bits) < 4:
            data_bits = np.pad(data_bits, (0, 4 - len(data_bits)), mode='constant')
        encoded_bits = encode_hamming(data_bits)
        hamming_encoded.extend(encoded_bits)
    return hamming_encoded

def converter(bitstream): 
    #Convert 0's, and 1's, to -1's, and 1's 
    x = 2 * bitstream - 1
    return x 

def add_noise(bitstream, std): 
    code_rate = 4 / 7 
    noise_std = 1/sqrt(std * code_rate)
    bitstream_noise = bitstream + noise_std * randn(len(bitstream))
    return bitstream_noise

def calculate_log_likelihood(noise, std): 
    return -2 * noise * (1/std)#(-4*noise)/(2*std)

def calculate_check_node(n1, n2, n3):
    return 2 * np.arctanh(np.tanh(0.5 * n1) * np.tanh(0.5 * n2) * np.tanh(0.5 * n3)) 

def decode(bitstream): 
    decoded_bit_stream = [] 
    for i in range(0, len(bitstream), 7): 
        chunck = bitstream[i : i + 7] 
        init_indexes = np.array([0, 1, 2, 1,2,3, 0, 1,3,4,5,6])
        #Initialisation 
        check_node = [0.0] * 12 
        code_node = [0.0] * 12 
         
        for i in range(len(check_node)): 
            code_node[i] = chunck[init_indexes[i]]

        #print(code_node)

        NUMBER_OF_ITERATIONS = 2
        for i in range(NUMBER_OF_ITERATIONS): 

            check_node[0] = calculate_check_node(code_node[1],code_node[2],code_node[9]) 
            check_node[1] = calculate_check_node(code_node[0],code_node[2],code_node[9]) 
            check_node[2] = calculate_check_node(code_node[0],code_node[1],code_node[9]) 
            check_node[3] = calculate_check_node(code_node[4],code_node[5],code_node[10]) 
            check_node[4] = calculate_check_node(code_node[3],code_node[5],code_node[10]) 
            check_node[5] = calculate_check_node(code_node[3],code_node[4],code_node[10])
            check_node[6] = calculate_check_node(code_node[7],code_node[8],code_node[11])  
            check_node[7] = calculate_check_node(code_node[6],code_node[8],code_node[11])
            check_node[8] = calculate_check_node(code_node[6],code_node[7],code_node[11])  
            check_node[9] = calculate_check_node(code_node[0],code_node[1],code_node[2]) 
            check_node[10] = calculate_check_node(code_node[3],code_node[4],code_node[5]) 
            check_node[11] = calculate_check_node(code_node[6],code_node[7],code_node[8]) 

            #print(check_node)

            code_node_temp = np.copy(code_node) 
            code_node[0] = code_node_temp[0] + check_node[6]
            code_node[1] = code_node_temp[1] + check_node[3] + check_node[7]
            code_node[2] = code_node_temp[2] + check_node[4]
            code_node[3] = code_node_temp[1] + check_node[1] + check_node[7]
            code_node[4] = code_node_temp[4] + check_node[2]
            code_node[5] = code_node_temp[5] + check_node[8]
            code_node[6] = code_node_temp[6] + check_node[0]
            code_node[7] = code_node_temp[7] + check_node[1] + check_node[3]
            code_node[8] = code_node_temp[8] + check_node[5] 

            #print(code_node)

        check_code_sum = np.array(check_node) + np.array(code_node)
        #print(np.array([check_code_sum[0], check_code_sum[1], check_code_sum[2], check_code_sum[5], check_code_sum[9], check_code_sum[10], check_code_sum[11]]))
        #code_word = np.array([check_code_sum[0], check_code_sum[1], check_code_sum[2], check_code_sum[5], check_code_sum[9], check_code_sum[10], check_code_sum[11]])
        code_word = np.array([check_code_sum[0], check_code_sum[1], check_code_sum[2], check_code_sum[5]]) 
        final_code_word = (code_word < 0).astype(int)
        #print(final_code_word)
        #print("Final", final_code_word)
        decoded_bit_stream.extend(final_code_word) 
        
    #print("Decoded", decoded_bit_stream)
    return decoded_bit_stream
    
def no_error_correction(): 
    N = 5000000
    EbNodB_range = range(0,11)
    itr = len(EbNodB_range)
    ber = [None]*itr

    for n in range (0, itr): 
    
        EbNodB = EbNodB_range[n]   
        EbNo=10.0**(EbNodB/10.0)
        x = 2 * (rand(N) >= 0.5) - 1
        noise_std = 1/sqrt(2*EbNo)
        y = x + noise_std * randn(N)
        y_d = 2 * (y >= 0) - 1
        errors = (x != y_d).sum()
        ber[n] = 1.0 * errors / N
        
        print("EbNodB: ",  EbNodB)
        print("Error bits:", errors)
        print("Error probability:", ber[n] )
            
    plt.plot(EbNodB_range, ber, 'bo', EbNodB_range, ber, 'k')
    plt.axis([0, 10, 1e-6, 0.1])
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('EbNo(dB)')
    plt.ylabel('BER')
    plt.grid(True)
    plt.title('BPSK Modulation')
    #plt.show()

def test_BP_algorithm(): 
    N = 50000
    Rc = 4/7
    EbNodB_range = range(0,11)
    itr = len(EbNodB_range)
    ber = [None]*itr
    
    for n in range (0, 11): 
        EbNodB = EbNodB_range[n]   
        EbNo=10.0**(EbNodB/10.0)
        noise_std = 1/sqrt(2*Rc * EbNo)
        random_bitstream = (rand(N) >= 0.5).astype(int)
        hamming_encoded = np.array(convert_to_hamming(random_bitstream), dtype=np.int8)
        hamming_converted = converter(hamming_encoded)
        #with_noise = add_noise(hamming_converted, 2 * EbNo)
        with_noise = hamming_converted + noise_std * randn(len(hamming_converted))
        log_likelihood = calculate_log_likelihood(with_noise)
        decoded = decode(log_likelihood)
        
        errors = (hamming_encoded != decoded).sum()
        ber[n] = 1.0 * errors / N
        
        print("EbNodB: ",  EbNodB)
        print("Error bits:", errors)
        print("Error probability:", ber[n] )

    plt.plot(EbNodB_range, ber, 'bo', EbNodB_range, ber, 'k')
    plt.axis([0, 10, 1e-6, 0.1])
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('EbNo(dB)')
    plt.ylabel('BER')
    plt.grid(True)
    plt.title('BPSK Modulation')
    plt.show()

def theoretical():
    EbNodB_range = np.linspace(0,10,10)
    P_e = 0.5 * special.erfc(np.sqrt(10**(EbNodB_range/10)))

    return EbNodB_range, P_e

def uncoded_bpsk(N_bits, amount_points, print_progress):
    EbNodB_range = np.linspace(0,10,amount_points)
    ber = [0.0]*len(EbNodB_range)

    for n in range(len(EbNodB_range)): 
        EbNodB = EbNodB_range[n]   
        EbNo=10.0**(EbNodB/10.0)
        random_bitstream = (rand(N_bits) >= 0.5).astype(int)
        noise_std = 1/sqrt(2*EbNo)
        with_noise = random_bitstream + noise_std * randn(N_bits)
        with_noise_d = 2 * (with_noise >= 0) - 1
        errors = (random_bitstream != with_noise_d).sum()
        ber[n] = 1.0 * errors / N_bits

        if print_progress:
            print("EbNodB: ",  EbNodB)
            print("Error bits:", errors)
            print("Error probability:", ber[n] )
            print()

    return EbNodB_range, ber


def get_N_Bits(n):
    if n < 2:
        return 12800000
    elif n < 4:
        return 12800000
    elif n < 6:
        return 12800000
    elif n < 8:
        return 79900000
    else:
        return 79900000

def encoded_bpsk(amount_points, print_progress):
    file_name = "values.csv"
    R_c = 4/7
    EbNodB_range = np.linspace(8.97,11,amount_points)
    ber = [0.0]*len(EbNodB_range)

    for n in range(len(EbNodB_range)):
        EbNodB = EbNodB_range[n]   
        EbNo=10.0**(EbNodB/10.0)
        noise_std = 1/sqrt(2*R_c*EbNo)
        random_bitstream = (rand(get_N_Bits(EbNo)) >= 0.5).astype(int)
        hamming_encoded = np.array(convert_to_hamming(random_bitstream), dtype=np.int8)
        hamming_converted = converter(hamming_encoded)

        with_noise = hamming_converted + noise_std * randn(len(hamming_converted))
        log_likelihood = calculate_log_likelihood(with_noise, noise_std**2)
        decoded = decode(log_likelihood)
        errors = (random_bitstream != decoded).sum()
        ber[n] = 1.0 * errors / len(decoded)

        with open("./values.csv", '+a') as fp:
            data = np.array([EbNodB, get_N_Bits(n), errors, ber[n]])
            np.savetxt(fp, [data], delimiter=',', fmt='%0.15f')
                
        if print_progress:
            print("EbNodB: ",  EbNodB)
            print("Error bits:", errors)
            print("Error probability:", ber[n])
            print()

    return EbNodB_range, ber

def run_simulation_plot():

    ebnodb_th, ber_th = theoretical()
    ebnodb_uncoded, ber_uncoded = uncoded_bpsk(192800000,300, True)


    plt.plot(ebnodb_uncoded, ber_uncoded, zorder=1, label="Simulated Uncoded BER")
    plt.scatter(ebnodb_th, ber_th, c="red", marker="+", zorder=2, label="Theoretical BER")


    #extracting values from csv
    values_df = pd.read_csv("values.csv")
    snrs = values_df["SNR"]
    probabilities = values_df["p"]

    plt.plot(np.array(snrs), np.array(probabilities), zorder=1, label="Hamming(7,4,3), Simulated BER")

    print(snrs)
    print(probabilities)

    plt.yscale("log")
    plt.xlabel('EbNo(dB)')
    plt.ylabel('BER')

    plt.gca().set_axisbelow(True)
    plt.grid(True)
    plt.title('Eb/No(SNR) Vs BER plot for BPSK Modulation in AWGN Channel')
    plt.legend()
    plt.show()
    plt.savefig("ber.png")
    pass


run_simulation_plot()       

# no_error_correction()
# test_BP_algorithm()

#hamming_encoded = np.array(convert_to_hamming(bitstream), dtype=np.int8)
#print("Hamming encoded", hamming_encoded)
#hamming_converted = converter(hamming_encoded)
#print("Hamming converted", hamming_converted)
#with_noise = add_noise(hamming_converted, 0.5)
#print("With noise", with_noise)
#log_likelihood = calculate_log_likelihood(with_noise)
#print("Log likelihood", log_likelihood)
#decode(log_likelihood)

# test_array = np.array([0.3505, -0.2181, -0.6464, 0.3264, 0.5954, -0.1677, 0.0654])

# decode(test_array)


