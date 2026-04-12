# str to ind
def str_to_ind(s):

    unicode_encoding = f"""the ind of s is {ord(s)}, 
    the hexadecimal of ind is {hex(ord(s))},
    the __repr__ of ind is {s.__repr__()},
    the __str__ of ind is {s.__str__()}
    """
    print(unicode_encoding)

if __name__ == "__main__":
    s = ['s', '牛', chr(0)]
    for i in s:
        str_to_ind(i)