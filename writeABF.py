"""
Code here relates to modification and de-novo creation of ABF files.
Files are saved as ABF1 format ABFs, which are easy to create because their
headers are simpler than ABF2 files. 
Many values (e.g., epoch waveform table) are left blank, so when they are read 
by an ABF reader their values may not make sense (especially when converted to 
floating-point numbers).
"""

import struct
import numpy as np


def writeABF(channelData, filename, sampleRateHz, units=['mV']):
    """
    Create an ABF1 file from scratch and write it to disk. This function works
    with data channels. If your data is organised as recording sweeps, please
    refer to a function called writeABF1.
    Input: channelData - a 2D numpy array (each row is a channel). A 1D array
                         is treated as single channel data.
           filename - a string with a file name.
           sampleRateHz - a scalar with data sampling frequency in Hz.
           units - a list of data unit strings. Default is ['mV'].
    """

    assert isinstance(channelData, np.ndarray)

    # constants for ABF1 files
    BLOCKSIZE = 512
    HEADER_BLOCKS = 4
    MAX_NUM_CH = 16

    # determine dimensions of data
    dataShape = channelData.shape
    if len(dataShape) < 2:
        channelData = channelData.reshape(1,len(channelData))
    channelCount = channelData.shape[0]
    channelPointCount = channelData.shape[1]
    dataPointCount = channelPointCount*channelCount

    # calculate how large our file must be and create a byte array of that size
    bytesPerPoint = 2
    dataBlocks = int(dataPointCount * bytesPerPoint / BLOCKSIZE) + 1
    data = bytearray((dataBlocks + HEADER_BLOCKS) * BLOCKSIZE)

    # populate only the useful header data values
    struct.pack_into('4s', data, 0, b'ABF ')  # fFileSignature
    struct.pack_into('f', data, 4, 1.3)  # fFileVersionNumber
    struct.pack_into('h', data, 8, 5)  # nOperationMode (5 is episodic)
    struct.pack_into('i', data, 10, dataPointCount)  # lActualAcqLength
    struct.pack_into('i', data, 16, 1)  # lActualEpisodes
    struct.pack_into('i', data, 40, HEADER_BLOCKS)  # lDataSectionPtr
    struct.pack_into('h', data, 100, 0)  # nDataFormat is 1 for float32
    struct.pack_into('f', data, 122, (1e6 / sampleRateHz) / channelCount)  # fADCSampleInterval
    struct.pack_into('i', data, 138, dataPointCount)  # lNumSamplesPerEpisode
    
    # Populate header data values relating to channels
    struct.pack_into('h', data, 120, channelCount)  # nADCNumChannels
    for ch in range(MAX_NUM_CH):
        if ch <= channelCount - 1:
            struct.pack_into('h', data, 378 + 2*ch, ch) # nADCPtoLChannelMap
            struct.pack_into('h', data, 410 + 2*ch, ch) # nADCSamplingSeq
        else:
            struct.pack_into('h', data, 378 + 2*ch, -1)
            struct.pack_into('h', data, 410 + 2*ch, -1)

    # These ADC adjustments are used for integer conversion. It's a good idea
    # to populate these with non-zero values even when using float32 notation
    # to avoid divide-by-zero errors when loading ABFs.

    fSignalGain = 1  # always 1
    fADCProgrammableGain = 1  # always 1
    lADCResolution = 2**15  # 16-bit signed = +/- 32768

    # determine the peak data deviation from zero
    maxVal = list()
    for ch in range(channelCount):
        maxVal.append(np.max(np.abs(channelData[ch,:])))

    # set the scaling factor to be the biggest allowable to accommodate the data
    fADCRange = 10
    valueScale = list()
    for ch in range(channelCount):
        fInstrumentScaleFactor = 1
        for i in range(10):
            fInstrumentScaleFactor /= 10
            valueScaleI = lADCResolution / fADCRange * fInstrumentScaleFactor
            maxDeviationFromZero = 32767 / valueScaleI
            if (maxDeviationFromZero >= maxVal[ch]):
                valueScale.append(valueScaleI)
                break

    # prepare units as a space-padded 8-byte string
    unitString = list()
    for ch in range(channelCount):
        unitStringCh = units[ch]
        while len(unitStringCh) < 8:
            unitStringCh = unitStringCh + " "
        unitString.append(unitStringCh)

    # store the scale data in the header
    struct.pack_into('i', data, 252, lADCResolution)
    struct.pack_into('f', data, 244, fADCRange)
    for ch in range(MAX_NUM_CH):
        struct.pack_into('f', data, 922+ch*4, fInstrumentScaleFactor)
        struct.pack_into('f', data, 1050+ch*4, fSignalGain)
        struct.pack_into('f', data, 730+ch*4, fADCProgrammableGain)
        struct.pack_into('8s', data, 602+ch*8, unitString[min([ch, channelCount-1])].encode())
    
    # interleave signals
    channelData = channelData.transpose()
    channelData = channelData.reshape((1,dataPointCount))[0]
    
    # scale signals
    valueScale = np.asarray(valueScale)
    valueScale = np.tile(valueScale, int(dataPointCount/(valueScale.shape)[0]))
    channelData = np.multiply(channelData, valueScale)
    
    # fill the rest of data with interleaved and scaled signals
    dataByteOffset = BLOCKSIZE * HEADER_BLOCKS
    channelData = channelData.astype(int)
    channelData = channelData.tolist()
    struct.pack_into(str(len(channelData))+'h', data, dataByteOffset, *channelData)

    # save the byte array to disk
    with open(filename, 'wb') as f:
        f.write(data)
    return