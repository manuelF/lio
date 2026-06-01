#!/usr/bin/python3
from math import sqrt

def read_dens( file_name ):

    data_file = open( file_name, "r" )

    dens_vals = []
    for data_line in data_file.readlines():
        dens_val1 = float( data_line.split()[0] )
        dens_vals.append( dens_val1 )

    data_file.close()
    return dens_vals

def print_dens_td( file_name, dens_vals ):

    data_file = open( file_name, 'w')

    for dens_val1 in dens_vals:
        data_file.write( (" %12.9E  %12.9E \n") % (dens_val1,0.0) )

    data_file.close()
    return 0

def print_dens( file_name, dens_vals ):

    data_file = open( file_name, 'w')

    basis_size = int(sqrt(len(dens_vals)))
    n_count = 0
    for dens_val1 in dens_vals:
        if ( n_count % basis_size != 0):
            dens_val1 = dens_val1 * 2.0
        data_file.write( (" %12.9E ") % (dens_val1) )

        n_count = n_count +1
        if (n_count % basis_size == 0):
            data_file.write ("\n")

    # Open-shell restarts (rst_dens=2, open=T) read two density blocks (alpha
    # then beta) and reconstruct the total as their sum. The movie density is a
    # single total density, so emit it as the alpha block above and append a
    # zero beta block here. The total (alpha+beta) is then exactly the
    # difference density. Closed-shell reads stop after the first block, so the
    # trailing zeros are harmless there.
    zero_row = (" %12.9E " % 0.0) * basis_size + "\n"
    for _ in range(basis_size):
        data_file.write(zero_row)

    data_file.close()
    return 0

dens0 = read_dens("dens_neg.in")
dens1 = read_dens("dens_pos.in")

densf = []
for idx in range( len( dens1 ) ):
    densf.append( dens1[idx] - dens0[idx] )

print_dens("dens_dif.out", densf)
print_dens_td("dens_dif_td.out", densf)
