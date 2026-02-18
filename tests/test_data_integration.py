import pytest

from tests import verify_yt1b_torchcodec
from tests.verify_ups_webdataset import verify_ups_webdataset

pytestmark = [pytest.mark.integration, pytest.mark.data]


def test_verify_ups_webdataset_script() -> None:
    verify_ups_webdataset()


def test_verify_yt1b_torchcodec_script_suite() -> None:
    verify_yt1b_torchcodec.test_fail_fast_when_torchcodec_unavailable()
    verify_yt1b_torchcodec.test_decoder_constructor_and_range_decode()
    verify_yt1b_torchcodec.test_shape_type_and_get_all_samples_path()
    verify_yt1b_torchcodec.test_auto_decode_window_uses_max_length_and_fallback_duration()
