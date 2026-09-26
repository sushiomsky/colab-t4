import argparse

from colab_t4 import supervisor


def _options():
    return argparse.Namespace(
        session="omp-hermes-t4", model="repo/model", quant="Q4_K_M", port=8081,
        ctx=16384, api_key="api-key", exec_timeout=30, ssh_mode="tailscale",
        account=None, account_only=False,
    )


def test_supervisor_starts_router_and_reprovisions_after_runtime_loss(monkeypatch):
    events = []
    healthy = iter([True, False])

    class FakeRouter:
        def serve_forever(self):
            events.append("router")

    monkeypatch.setattr(supervisor, "start_router", lambda host, port: FakeRouter())
    monkeypatch.setattr(supervisor, "runtime_ready", lambda: next(healthy, True))
    sleeps = iter([None, KeyboardInterrupt()])
    def fake_sleep(_seconds):
        value = next(sleeps)
        if isinstance(value, BaseException):
            raise value
    monkeypatch.setattr(supervisor.time, "sleep", fake_sleep)
    monkeypatch.setattr(supervisor, "runtime_up", lambda options: events.append((options.session, options.model, options.quant)))

    try:
        supervisor.run(_options(), host="127.0.0.1", router_port=8089)
    except KeyboardInterrupt:
        pass

    assert events[0] == "router"
    assert events[1] == ("omp-hermes-t4", "repo/model", "Q4_K_M")


def test_supervisor_rejects_when_no_google_profile_is_authenticated(monkeypatch):
    monkeypatch.setattr(supervisor, "authenticated_accounts", lambda: [])
    try:
        supervisor.run(_options(), host="127.0.0.1", router_port=8089)
    except RuntimeError as exc:
        assert "authenticated Google account" in str(exc)
    else:
        raise AssertionError("supervisor should require an authenticated profile")


def test_supervisor_rotates_before_colab_300_minute_limit(monkeypatch):
    options = _options()
    events = []
    monkeypatch.setattr(supervisor, "authenticated_accounts", lambda: ["a", "b"])
    monkeypatch.setattr(supervisor, "load_accounts", lambda: ["a", "b"])
    monkeypatch.setattr(supervisor, "load_state", lambda: {"account": "a"})
    monkeypatch.setattr(supervisor, "start_router", lambda host, port: type("R", (), {
        "serve_forever": lambda self: None,
        "shutdown": lambda self: None,
        "server_close": lambda self: None,
    })())
    monkeypatch.setattr(supervisor, "runtime_ready", lambda: True)
    monkeypatch.setattr(supervisor, "runtime_age_seconds", lambda: 17_401)
    monkeypatch.setattr(supervisor, "runtime_down", lambda: None)
    monkeypatch.setattr(supervisor, "runtime_up", lambda opts: events.append(opts.account))
    sleeps = iter([KeyboardInterrupt()])
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: (_ for _ in ()).throw(next(sleeps)))

    try:
        supervisor.run(options, host="127.0.0.1", router_port=8089, interval=1, max_runtime_seconds=17_400)
    except KeyboardInterrupt:
        pass
    assert events == ["b"]


def test_supervisor_keeps_router_alive_when_colab_assignment_fails(monkeypatch):
    events = []
    monkeypatch.setattr(supervisor, "authenticated_accounts", lambda: ["a"])
    monkeypatch.setattr(supervisor, "load_accounts", lambda: [type("A", (), {})()])
    monkeypatch.setattr(supervisor, "runtime_ready", lambda: False)
    monkeypatch.setattr(supervisor, "runtime_up", lambda options: (_ for _ in ()).throw(RuntimeError("quota")))
    monkeypatch.setattr(supervisor, "start_router", lambda host, port: type("R", (), {
        "serve_forever": lambda self: events.append("router"),
        "shutdown": lambda self: None,
        "server_close": lambda self: None,
    })())
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))

    supervisor.run(_options(), interval=1)

    assert events == ["router"]
