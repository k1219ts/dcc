import ice

# rec709toACEScg = [
#     0.610277, 0.0688436, 0.0241673, 0.0,
#     0.345424, 0.934974 , 0.121814 , 0.0,
#     0.0443001, -0.00381805, 0.854019, 0.0,
#     0.0, 0.0, 0.0, 1.0
# ]

# ITU-R BT.709 -> ACEScg - chromatic adaptation: Bianco
rec709toACEScg = [
    0.612459, 0.070664, 0.020755, 0.000000,
    0.338722, 0.917631, 0.106878, 0.000000,
    0.048818, 0.011705, 0.872367, 0.000000,
    0.000000, 0.000000, 0.000000, 1.000000
]

mtx = rec709toACEScg

iFLOAT = ice.constants.FLOAT

def licRec709ToLinAP1(iceImage):
    r = iceImage.Shuffle([0, 0, 0])
    g = iceImage.Shuffle([1, 1, 1])
    b = iceImage.Shuffle([2, 2, 2])

    # X
    m00 = ice.Card(iFLOAT, [mtx[0]]).Multiply(r)
    m01 = ice.Card(iFLOAT, [mtx[4]]).Multiply(g)
    m02 = ice.Card(iFLOAT, [mtx[8]]).Multiply(b)
    X   = ice.Card(iFLOAT, [1, 0, 0]).Multiply(m00.Add(m01).Add(m02))

    # Y
    m10 = ice.Card(iFLOAT, [mtx[1]]).Multiply(r)
    m11 = ice.Card(iFLOAT, [mtx[5]]).Multiply(g)
    m12 = ice.Card(iFLOAT, [mtx[9]]).Multiply(b)
    Y   = ice.Card(iFLOAT, [0, 1, 0]).Multiply(m10.Add(m11).Add(m12))

    # Z
    m20 = ice.Card(iFLOAT, [mtx[2]]).Multiply(r)
    m21 = ice.Card(iFLOAT, [mtx[6]]).Multiply(g)
    m22 = ice.Card(iFLOAT, [mtx[10]]).Multiply(b)
    Z   = ice.Card(iFLOAT, [0, 0, 1]).Multiply(m20.Add(m21).Add(m22))

    result = X.Add(Y).Add(Z)
    return result
