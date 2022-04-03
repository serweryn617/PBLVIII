import csv
import numpy as np


def decompress_time(time: int) -> tuple[int, int, int]:
    """Extract time from word"""
    second = (time % 30) *2
    minute = (time // 30) % 60
    hour = time // 1800
    # print(f"{hour}:{min}:{sec}")
    return hour, minute, second


def decompress_date(date: int) -> tuple[int, int, int]:
    """Extract date from word"""
    day = date & 0x001F
    month = (date >> 5) & 0x000F
    year = (date >> 9) & 0x007F
    # print(f'{day}/{month}/{year+2000}')
    return day, month, year


def parse_bytes(byte: bytes) -> int:
    """Parse 2's complement word encoded in little endian into int"""
    return int.from_bytes(byte, byteorder='little', signed=True)


# .svn file headers
HEADERS = {
    0x01:'file header',
    0x02:'unit header',
    0x03:'user text',
    0x04:'global settings',
    0x05:'channel settings (hardware)',
    0x31:'trigger event settings',
    0x07:'channel settings (software)',
    0x08:'profile settings',
    0x1E:'vector settings',
    0x18:'buffer header',
    0x09:'octaves settings',
    0x0A:'octaves settings in channels',
    0x34:'cross spectrum settings',
    0x21:'spectrum buffer header'
}

# Headers which contain other headers inside
CONTAINERS = [7,9]

# Specific frequencies in the buffer [Hz]
FREQUENCIES = (
    0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3,
    8, 10, 12.5, 16, 20, 25, 31.5, 40, 50, 63,
    80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
    8000, 10000, 12500, 16000, 20000
)
# FREQUENCIES[18:-6] -> 50Hz : 5000Hz


class svn_buffer_parser():
    def __init__(self) -> None:
        self.data = None
        self.channels = 3
        self.samples = 160
        self.frequencies = len(FREQUENCIES)
        self.totals = 3 # A, C, Lin
        self.step = 100 # ms


    def load(self, path: str) -> None:
        """Load and parse a specific file.

        Parameters:\n
        path -- file to open
        """
        file = open(path, 'rb')
        file.read(32) # SVN file header

        # Read all of the headers
        while True:
            # Start with reading header number and header length
            header = int.from_bytes(file.read(1), byteorder='little')
            length = int.from_bytes(file.read(1), byteorder='little')
            
            if header in CONTAINERS:
                file.read(2)
                continue

            # If length equals 0, actual length is stored in next word
            if length == 0:
                length = int.from_bytes(file.read(2), byteorder='little') - 1

            # if header == 0:
            #     break
            
            # Buffer header
            if header == 0x18:
                file.read(4) # First 2 words can be omited
                self.step = parse_bytes(file.read(2)) # Time step between measurements
                # print('step', self.step)
                file.read(4) # Next word can be omited
                
                # for channels in range(self.channels):
                self.samples = parse_bytes(file.read(4)) # Number of measurements (long)
                # print('samples', self.samples)

                # Skip the rest
                # for i in range(length - 8):
                file.read(2 * (length - 8))
                continue

            # Skip the contents of every other header
            # for i in range(length - 1):
            file.read(2 * (length - 1))
                # print(byte, end=' ')

            # Buffer contents
            if header == 0x21:
                break

        # Read buffer contents
        self.channels = 3
        # self.samples = 160
        # self.frequencies = len(FREQUENCIES)
        self.totals = 3 # A, C, Lin

        # Preallocate arrays for buffer contents
        # out1 = np.zeros((channels, samples)) # Leq
        # out2 = np.zeros((channels, frequencies, samples)) # 1/3 Oct Leq
        # out3 = np.zeros((channels, totals, samples)) # Total
        self.data = np.zeros((self.channels, 1 + self.frequencies + self.totals, self.samples))

        for sample in range(self.samples):
            # print(f'\nSample {n}')

            # Channels 1,2,3... in a row
            for channel in range(3):
                num = parse_bytes(file.read(2)) / 20
                # print(num, end=' ')

                # Add to output array
                self.data[channel, 0, sample] = num

            # Channel 1 tercets and totals, channel 2 tercets and totals...
            for channel in range(3):
                # print(f'\n{j}:')

                # First word is always 0000h
                if parse_bytes(file.read(2)) != 0:
                    print('ERROR')

                # Tercets followed by totals
                for value in range(self.frequencies + self.totals):
                    num = parse_bytes(file.read(2)) / 10
                    # print(num, end=' ')

                    # Add to output array
                    self.data[channel, value + 1, sample] = num


    def get_data(self, type: str = 'all', channel: int = 0, tanspose: bool = False) -> np.ndarray:
        """Returns obtained data as numpy array.

        Parameters:\n
        type -- which data to return (default 'all') 
            Possible types are:
            - 'all' -- entire data,
            - 'main' -- only main readings,
            - 'tercets' -- values for specific frequencies
            - 'totals'\n
        channel -- channel from which data is returned (default 0)\n
        transpose -- if data should be transposed
        """
        match type:
            case 'main':
                return self.data[channel, 0, :] if not tanspose else self.data[channel, 0, :].T
            case 'tercets':
                return self.data[channel, 1:-self.totals, :] if not tanspose else self.data[channel, 1:-self.totals, :].T
            case 'totals':
                return self.data[channel, -self.totals:, :] if not tanspose else self.data[channel, -self.totals:, :].T
        # Default
        return self.data


    def export_csv(self, path: str = 'output') -> None:
        """Export data to csv files in specified directory.

        Parameters:\n
        path -- directory for exported files (default 'output')
        """

        for i in range(self.channels):
            with open(f'{path}/main{i}.csv', 'w', newline='') as out_file:
                csv_writer = csv.writer(out_file)
                csv_writer.writerows(self.data[i, 0:1, :])

            with open(f'{path}/tercets{i}.csv', 'w', newline='') as out_file:
                csv_writer = csv.writer(out_file)
                csv_writer.writerows(self.data[i, 1:-self.totals, :])

            with open(f'{path}/totals{i}.csv', 'w', newline='') as out_file:
                csv_writer = csv.writer(out_file)
                csv_writer.writerows(self.data[i, -self.totals:, :])


def main():
    reader = svn_buffer_parser()
    reader.load('PBL_Badania_v1/Buffe_32.svn')
    reader.export_csv('out')


if __name__ == '__main__':
    main()