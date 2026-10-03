import re

from mosaic.session import new_investigation_id


def test_new_investigation_ids_are_readable_and_unique():
    first = new_investigation_id()
    second = new_investigation_id()

    pattern = r"^INV-\d{8}-\d{6}-[0-9a-f]{8}$"
    assert re.match(pattern, first)
    assert re.match(pattern, second)
    assert first != second


def test_new_investigation_id_accepts_custom_prefix():
    value = new_investigation_id(prefix="EFI")

    assert value.startswith("EFI-")
