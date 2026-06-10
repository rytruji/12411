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
    @staticmethod
    def default_binning(header):
        binning = header.get("XBINNING") or header.get("BINX")
        return binning


class Binning():
    def __init__(self, data, header, format):
        self.original_bin_size = format(header) or 1

        self._unbinned_copy = np.asarray(data, dtype=np.float32)
        self.bin_size = self.original_bin_size
        self.binned = self._unbinned_copy.copy()
    

    def set_bin_size(self, target_size):
        assert isinstance(target_size, int)

        if self.bin_size != target_size:
            self._rebin(target_size)


    def _rebin(self, target_size: int):
        factor = target_size // self.original_bin_size
        data = self._unbinned_copy

        h, w = (np.array(data.shape) // factor) * factor
        data = data[:h, :w]

        row_idx = np.arange(0, h, factor)
        col_idx = np.arange(0, w, factor)
        self.binned = np.add.reduceat(np.add.reduceat(data, row_idx, axis=0), col_idx, axis=1)
        self.bin_size = target_size

    
    def _unbin(self):
        self.binned = self._unbinned_copy
        self.bin_size = self.original_bin_size