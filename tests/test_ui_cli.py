from kairorag import cli


def test_ui_cli_help_is_available():
    try:
        cli.main(["ui", "--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_doctor_repo_and_release_parse(monkeypatch):
    class FakeReport:
        ok = True
        issues = []

        def to_dict(self):
            return {"ok": True, "issues": []}

    class FakeChecker:
        def __init__(self, root):
            self.root = root

        def run(self):
            return FakeReport()

    monkeypatch.setattr("kairorag.release.RepoHygieneChecker", FakeChecker)

    assert cli.main(["doctor", "--repo"]) == 0
    assert cli.main(["doctor", "--release"]) == 0
