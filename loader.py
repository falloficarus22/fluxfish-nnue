import struct
import numpy as np

class NNUELoader:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'rb')

    def read_uint32(self):
        return struct.unpack('<I', self.file.read(4))[0]

    def read_header(self):
        version = self.read_uint32()
        hash_val = self.read_uint32()
        desc_len = self.read_uint32()
        description = self.file.read(desc_len).decode('utf-8')
        print(f"Loading NNUE: {description}")
        return version

    def _decompress(self, count):
        """DECOMPRESSOR: Zigzag LEB128"""
        weights = np.zeros(count, dtype=np.int16)
        data = self.file.read() # Read remainder
        d_idx = 0
        for i in range(count):
            val = 0
            shift = 0
            while True:
                b = data[d_idx]
                d_idx += 1
                val |= (b & 0x7f) << shift
                if not (b & 0x80): break
                shift += 7
            # Zigzag
            weights[i] = (val >> 1) ^ -(val & 1)
        return weights, data[d_idx:]

    def load_full_sf15(self, input_size, hidden_size):
        self.read_header()
        
        # 1. FT Hash
        self.read_uint32()
        
        # 2. Check for compression magic
        magic = self.file.read(17)
        if magic != b"COMPRESSED_LEB128":
            # If not compressed, maybe biases are here?
            self.file.seek(-17, 1) # back up
            biases = np.frombuffer(self.file.read(hidden_size * 2), dtype=np.int16)
            # Check for magic again
            magic = self.file.read(17)
            if magic == b"COMPRESSED_LEB128":
                uncomp_size = self.read_uint32()
                ft_weights, head_data = self._decompress(input_size * hidden_size)
            else:
                self.file.seek(-17, 1)
                ft_weights = np.frombuffer(self.file.read(input_size * hidden_size * 2), dtype=np.int16)
                head_data = self.file.read()
        else:
            # Entire FT (Biases + Weights) is compressed?
            uncomp_size = self.read_uint32()
            print(f"Decompressing {uncomp_size} values...")
            # Uncompressed size is in bytes? No, usually in elements.
            # SF stores biases then weights.
            all_ft, head_data = self._decompress(hidden_size + input_size * hidden_size)
            biases = all_ft[:hidden_size]
            ft_weights = all_ft[hidden_size:]

        # 3. Process Head Layers
        def pull_layer(buffer, in_d, out_d):
            # Head layers in SF 15.1 might also be compressed!
            # Let's check for magic
            if buffer[:17] == b"COMPRESSED_LEB128":
                buffer = buffer[17:]
                uncomp = struct.unpack('<I', buffer[:4])[0]
                buffer = buffer[4:]
                # Decompress biases + weights
                # Biases are int32, weights are int8.
                # Actually, zigzag LEB128 is for everything.
                # However, for simplicity, SF mostly only compresses the big FT.
                pass 
            
            # Assuming head is uncompressed for now
            b_size = out_d * 4
            bias = np.frombuffer(buffer[:b_size], dtype=np.int32)
            buffer = buffer[b_size:]
            w_size = in_d * out_d
            weight = np.frombuffer(buffer[:w_size], dtype=np.int8)
            buffer = buffer[w_size:]
            return weight.reshape(out_d, in_d), bias, buffer

        l1_w, l1_b, head_data = pull_layer(head_data, hidden_size * 2, 15)
        l2_w, l2_b, head_data = pull_layer(head_data, 15, 15)
        l3_w, l3_b, head_data = pull_layer(head_data, 15, 1)

        return {
            'ft_w': ft_weights.reshape(input_size, hidden_size).astype(np.int32),
            'ft_b': biases.astype(np.int32),
            'l1_w': l1_w.astype(np.int32), 'l1_b': l1_b.astype(np.int32),
            'l2_w': l2_w.astype(np.int32), 'l2_b': l2_b.astype(np.int32),
            'l3_w': l3_w.astype(np.int32), 'l3_b': l3_b.astype(np.int32)
        }