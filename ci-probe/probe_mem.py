"""Temporary pytest plugin: print warm/grew/slack per m69b memory test (removed before merge)."""

import importlib

MiB = 1 << 20


def pytest_collection_finish(session):
    t = importlib.import_module("test_histserv_memory_model")
    h = importlib.import_module("histserv_harness")
    measure, warm = t._measure, t._warm

    def _warm(path, server, name):
        idle = h._status(server.proc.pid, "VmRSS")
        w = warm(path, server, name)
        print(f"PROBE {name} idle={idle / MiB:.1f} warm={w / MiB:.1f}")
        return w

    def _measure(path, name, *a):
        w, grew, pred = measure(path, name, *a)
        print(f"PROBE {name} grew={grew / MiB:.1f} pred-B={(pred - h.BASE) / MiB:.1f} slack={(pred - h.BASE - grew) / MiB:.1f}")
        return w, grew, pred

    t._warm, t._measure = _warm, _measure
