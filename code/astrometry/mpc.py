#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#------------------------------#   created 09/03/2026 by truji@mit.edu   #------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

from astropy.time import Time
import astropy.units as u

from astropy.coordinates import SkyCoord

#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

def eighty_column(packed_desig, # cols 6-12 (7)
                  packed_number, # cols 1-5 (5)
                  discovery, # col 13 (1)
                  note1,  # col 14 (1)
                  note2, # col 15 (1))
                  utc, # cols 16-32 (17)
                  coord, # cols 33-56
                  mag,
                  band, 
                  obs_code):
    
    assert len(packed_number) == 5
    assert len(packed_desig) == 7
    assert type(discovery) == bool
    assert len(note1) == 1 and len(note2) == 1
    assert isinstance(utc, Time)
    assert isinstance(coord, SkyCoord)
    assert len(f"{mag:2.1f}") == 4 and isinstance(mag, float)
    assert len(band) == 1
    assert len(str(obs_code)) == 3 and isinstance(obs_code, int)

    packed_full = packed_number + packed_desig
    packed_full = f"{packed_full:<12}"
    
    if discovery:
        asterisk = "*"
    else:
        asterisk = " "

    jd = utc.utc.jd
    day_frac = jd % 1
    mpc_time = utc.utc.strftime("%Y %m %d.") + f"{day_frac:.6f}"[2:]
    mpc_time = f"{mpc_time:<17}"[:17]

    ra = coord.ra.to_string(unit=u.hour, sep=" ", pad=True, precision=2)
    dec = coord.dec.to_string(unit=u.deg, sep=" ", alwayssign=True, pad=True, precision=1)
    
    ra = f"{ra:<12}"[:12]
    dec = f"{dec:<12}"[:12]
    
    magband = f"{mag:>4.1f} {band:<1}"

    out = (
        f"{packed_full}"
        f"{asterisk}"
        f"{note1}{note2}"
        f"{mpc_time}"
        f"{ra}"
        f"{dec}"    
        f"{' '*9}"     
        f"{magband}" 
        f"{' '*6}" 
        f"{obs_code}"
    )

    print(out)

    assert len(out) == 80, len(out)

    return out