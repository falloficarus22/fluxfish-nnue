import struct
import numpy as np

class NNUELoader:

    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'rb')

    def read_int32(self):
        return struct.unpack('<i', self.file.read(4))[0]
    
    def read_uint32(self):
        return struct.unpack('<I', self.file.read(4))[0]
    
    def read_header(self):
        # Stockfish includes a header with a Magic version and a hash
        version = self.read_uint32()
        hash_val = self.read_uint32()
        desc_len = self.read_uint32()
        description = self.file.read(desc_len).decode('utf-8')

        print(f"File Name: {self.filename}")
        print(f"Loading NNUE: {description}")

        return version
    
    def load_weights(self, input_size, hidden_size):
        """
        Loads first layer weights (feature weights)
        These are stored as 16-bit integers (int16)
        """
        # Load biases (int16 * hidden size)
        biases = np.frombuffer(self.file.read(hidden_size * 2), dtype = np.int16)

        # Load weights (int16 * input_size * hidden_size)
        weights = np.frombuffer(self.file.read(input_size * hidden_size * 2), dtype = np.uint16)

        return weights.reshape(input_size, hidden_size), biases
    
    def load_layer(self, in_dim, out_dim, weight_type = np.int8, bias_type = np.int32):
        """
        Load hidden layers. Stockfish stores weights as int8 and biases as int32
        """
        biases = np.frombuffer(self.file.read(out_dim * bias_type().item_size), dtype = bias_type)
        weights = np.frombuffer(self.file.read(in_dim * out_dim * weight_type().item_size), dtype = weight_type)

        return weights.reshape(out_dim, in_dim), biases
    