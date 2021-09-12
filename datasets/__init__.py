from .mnist import get_mnist
from .usps import get_usps
from .k_mnist import get_kmnist
from .svhn import get_svhn

from .descendant_activations import get_conv_1_activations
from .successor_activations import get_conv_2_activations
from .office_home import get_office_home
from .office_31 import get_office_31
from .cifar_10 import get_cifar_10
from .stl_10 import get_stl_10
from .src_encoded import get_src_encoded
from .tgt_encoded import get_tgt_encoded

__all__ = (get_usps, get_mnist, get_kmnist, get_svhn, get_conv_1_activations, get_conv_2_activations, \
            get_office_home, get_office_31, \
            get_cifar_10, get_stl_10, \
            get_src_encoded, get_tgt_encoded)
