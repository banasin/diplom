import numpy as np
from src.features.structure import nussinov, structure_features, STRUCT_FEATURE_NAMES


def test_nussinov_non_pairing():
    # только A — комплементарных пар нет
    assert nussinov("AAAAAAAA") == "........"


def test_nussinov_pairs_complement():
    # G-богатый + C-богатый с петлёй -> есть пары
    st = nussinov("GGGGAAACCCC")
    assert st.count("(") == st.count(")")
    assert st.count("(") >= 2                       # хотя бы стебель из 2 пар


def test_structure_features_shape_and_values():
    f = structure_features("GGGGAAACCCC")
    assert f.shape == (len(STRUCT_FEATURE_NAMES),)
    assert 0.0 <= f[0] <= 1.0                        # доля спаренных
    assert f[1] >= 2                                 # число пар
    f0 = structure_features("AAAAAAAA")
    assert f0[1] == 0                                # нет пар


def test_u_equals_t():
    assert nussinov("GGGGAAACCCC") == nussinov("GGGGAAACCCC".replace("T", "U"))
