def interpolasi_gap(x):
    if x == 0:
        x = 11
    elif x == 10 or x == -10:
        x = 10.5 if x == 10 else 10
    elif x == 20 or x == -20:
        x = 9.5 if x == 20 else 9
    elif x == 30 or x == -30:
        x = 8.5 if x == 30 else 8
    elif x == 40 or x == -40:
        x = 7.5 if x == 40 else 7
    elif x == 50 or x == -50:
        x = 6.5 if x == 50 else 6
    elif x == 60 or x == -60:
        x = 5.5 if x == 60 else 5
    elif x == 70 or x == -70:
        x = 4.5 if x == 70 else 4


    elif x < 0 and x >= -10:
        x = ((11 - 10) * (x - (-10)) / (0-(-10))) + 10
    elif x > 0 and x <= 10:
        x = ((11 - 10.5) * (x - (0)) / (0-(-10))) + 10.5
        
    return x


def interpolate_3_points(value, q1, q2):
    score = 0

    if value < q1:
        score = 5
    elif value >= q1 and value <= q2:
        score = (((1 - 5) / (q2 - q1)) * (value - q1)) + 1
    else:
        score = 1

    return score


def interpolate_4_points(value, q1, q2, q3):
    score = 0
    
    if value < q1:
        score = (((5 - 1) / (q1 - 0)) * (value - 0)) + 1
    elif value >= q1 and value <= q2:
        score = 5
    elif value > q2 and value <= q3:
        score = (((1 - 5) / (q3 - q2)) * (value - q2)) + 5
    else:
        score = 1

    return score


def replace_with_mid_value(value):
    # parse value with comma
    if ',' in value:
        value = value.replace(',', '.')
        min_val, max_val = map(float, value.split("-"))
    else:
        # split value into 2
        min_val, max_val = map(float, value.split("-"))
    # calculate mid value
    mid_val = (min_val + max_val) / 2
    return mid_val


# "20 - 50" -> range -> nilai tengah
# "-50" -> 1