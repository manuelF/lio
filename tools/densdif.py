#!/usr/bin/python3
from math import sqrt
from typing import List


def read_dens(file_name: str) -> List[float]:
    data_file = open(file_name, "r")
    dens_vals: List[float] = []
    for data_line in data_file.readlines():
        dens_val1 = float(data_line.split()[0])
        dens_vals.append(dens_val1)

    data_file.close()
    return dens_vals


def print_dens_td(file_name: str, dens_vals: List[float]) -> None:
    data_file = open(file_name, "w")
    for dens_val1 in dens_vals:
        data_file.write((" %12.9E  %12.9E \n") % (dens_val1, 0.0))
    data_file.close()


def print_dens(file_name: str, dens_vals: List[float]) -> None:
    data_file = open(file_name, "w")
    basis_size = int(sqrt(len(dens_vals)))
    n_count = 0
    for dens_val1 in dens_vals:
        if n_count % basis_size != 0:
            dens_val1 = dens_val1 * 2.0
        data_file.write((" %12.9E ") % (dens_val1))

        n_count = n_count + 1
        if n_count % basis_size == 0:
            data_file.write("\n")
    data_file.close()


dens0 = read_dens("dens_neg.in")
dens1 = read_dens("dens_pos.in")

densf: List[float] = []
for idx in range(len(dens1)):
    densf.append(dens1[idx] - dens0[idx])

print_dens("dens_dif.out", densf)
print_dens_td("dens_dif_td.out", densf)
