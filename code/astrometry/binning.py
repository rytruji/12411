#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#------------------------------#   created 09/03/2026 by truji@mit.edu   #------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

import numpy as np

#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

class Format():
    def default_binning(header):
        binning = header.get("XBINNING")
        if binning is None:
            binning = header.get("BINX")
        return binning


class Binning():
    def __init__(self, data, header, format):
        self.bin_size = format(header)

        self._unbinned_copy = data.copy()
        self.binned = data.copy()


    def get_bin_size(self):
        return self.bin_size
    

    def set_bin_size(self, target_size):
        assert isinstance(target_size, int)

        self.get_bin_size()

        if self.bin_size != target_size:
            self.__rebin(target_size)


    def __rebin(self, target_size):
        # get number of 2x2 binning iterations to apply, check it is an integer
        bin_log = np.log2(target_size / self.bin_size)
        assert bin_log == int(bin_log)

        # for number of iterations needed, bin by 2
        for _ in range(int(bin_log)):
            # determine new shape after 2x2 binning
            new_size = np.array(self.binned.shape) // 2

            # if size of array is odd, will not work, destructively reshape
            x_bounds, y_bounds = (np.array(self.binned.shape) // 2) * 2
            self.binned = self.binned[:x_bounds, :y_bounds]

            # bin 2x2, normalize by mean
            self.binned = self.binned.reshape(new_size[0],2,new_size[1],2).sum(axis=(1,3))