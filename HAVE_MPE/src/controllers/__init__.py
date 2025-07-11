REGISTRY = {}

from .basic_controller import BasicMAC
from .have_controller import HAVEMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["have_mac"] = HAVEMAC
