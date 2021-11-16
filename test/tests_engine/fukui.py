#!/usr/bin/env python3
import os
import re
from typing import List, Literal, TextIO, Union


def obtain_fukui(file_in: TextIO) -> Union[List[float], Literal[-1]]:
    lista: List[float] = []
    for line in file_in.readlines():
        m = re.match(
            r"\s+\d+\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)", line
        )
        if m:
            lista.append(float(m.group(1)))
            lista.append(float(m.group(2)))
            lista.append(float(m.group(3)))
            lista.append(float(m.group(4)))

    if len(lista) < 1:
        return -1

    return lista


def error(fuk: List[float], fukok: List[float]) -> Literal[0, -1, 1]:
    dim1 = len(fuk)
    dim2 = len(fukok)

    if dim1 != dim2:
        print("There are different number of Fukui charges in outputs.")
        return -1

    scr = 0
    for num in range(dim1):
        value = abs(fuk[num] - fukok[num])
        if value > 1e-2:
            scr = 1
            print("Error in fukui:")
            print("Value of fukui", fuk[num])
            print("Value of fukui.ok", fukok[num])

    if scr == 0:
        return 0
    return 1


def Check() -> Literal[0, -1]:
    # Output
    fuk = []
    is_file = os.path.isfile("fukui")
    if not is_file:
        print("The fukui file is missing.")
        return -1

    f = open("fukui", "r")
    fuk = obtain_fukui(f)
    f.close
    if not fuk:
        print("Error in reading fukui.")

    # Ideal Output
    fukok = []
    is_file = os.path.isfile("fukui")
    if not is_file:
        print("The fukui.ok file is missing.")
        return -1

    f = open("fukui.ok", "r")
    fukok = obtain_fukui(f)
    f.close
    if not fukok:
        print("Error in reading fukui.ok.")

    ok_output = error(fuk, fukok)

    if ok_output != 0:
        print("Test Fukui:      ERROR")
    else:
        print("Test Fukui:      OK")

    return 0
