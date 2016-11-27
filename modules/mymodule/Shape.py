


def NuttyNumber(x, type):
    if type == 'one':
        return x*3
    elif type == 'two':
        return x + 1
    else:
        return "You choose " + str(x) + " poorly."


if __name__ == "__main__":
    NuttyNumber()
