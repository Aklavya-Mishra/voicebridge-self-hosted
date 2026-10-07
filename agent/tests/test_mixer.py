import numpy as np

from mixer import StereoMixer


def block(value, seconds, rate=48000, channels=1):
    return np.full(int(seconds * rate) * channels, value, dtype=np.float32)


def left_right(chunks):
    frames = np.frombuffer(b"".join(chunks), dtype=np.int16).reshape(-1, 2) / 32767
    return frames[:, 0], frames[:, 1]


def test_mic_drives_timing_and_loopback_gaps_are_silence():
    mixer = StereoMixer(mic_rate=48000, mic_channels=1, loop_rate=48000, loop_channels=1)
    mixer.add_mic(block(0.5, 1.0))  # loopback delivered nothing (PC silent)
    chunks = mixer.chunks()
    assert len(chunks) >= 9  # ~1 s of mic in 100 ms chunks (resampler keeps a few ms back)
    assert all(len(c) == 1600 * 2 * 2 for c in chunks)
    left, right = left_right(chunks)
    assert np.allclose(left[4000:], 0.5, atol=0.01)
    assert np.all(right == 0)


def test_channels_are_downmixed_resampled_and_kept_apart():
    mixer = StereoMixer(mic_rate=44100, mic_channels=2, loop_rate=48000, loop_channels=2)
    mic = np.tile(np.array([0.4, 0.2], dtype=np.float32), int(1.0 * 44100))  # downmix -> 0.3
    mixer.add_mic(mic)
    mixer.add_loop(block(-0.25, 1.0, channels=2))
    left, right = left_right(mixer.chunks())
    assert np.allclose(left[4000:], 0.3, atol=0.01)
    assert np.allclose(right[4000:12000], -0.25, atol=0.01)


def test_loopback_running_ahead_is_trimmed():
    mixer = StereoMixer(mic_rate=48000, mic_channels=1, loop_rate=48000, loop_channels=1, max_lead_s=0.5)
    mixer.add_loop(block(0.1, 3.0))
    mixer.add_mic(block(0.5, 0.2))
    mixer.chunks()
    assert mixer.loop_backlog_s() <= 0.5


def test_no_chunk_until_100ms_of_mic():
    mixer = StereoMixer(mic_rate=16000, mic_channels=1, loop_rate=16000, loop_channels=1)
    mixer.add_mic(block(0.5, 0.05, rate=16000))
    assert mixer.chunks() == []


def test_call_audio_from_several_output_devices_is_mixed():
    """The call app may play on any output (speakers, HDMI monitor, headset): all of them are the call side."""
    mixer = StereoMixer(mic_rate=48000, mic_channels=1, loop_rate=48000, loop_channels=2,
                        extra_loops=[(44100, 2)])
    mixer.add_mic(block(0.5, 1.0))
    mixer.add_loop(block(0.2, 1.0, channels=2))  # default speakers
    mixer.add_loop(block(0.1, 1.0, rate=44100, channels=2), source=1)  # e.g. HDMI monitor
    left, right = left_right(mixer.chunks())
    assert np.allclose(right[4000:12000], 0.3, atol=0.01)


def test_a_silent_output_device_does_not_hold_back_the_others():
    mixer = StereoMixer(mic_rate=48000, mic_channels=1, loop_rate=48000, loop_channels=1, extra_loops=[(48000, 1)])
    mixer.add_mic(block(0.5, 1.0))
    mixer.add_loop(block(0.2, 1.0), source=1)  # only the second device plays anything
    left, right = left_right(mixer.chunks())
    assert np.allclose(right[4000:12000], 0.2, atol=0.01)
