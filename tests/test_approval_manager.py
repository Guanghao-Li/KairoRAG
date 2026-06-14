from kairorag.security import ApprovalManager, ApprovalRequest


def _request():
    return ApprovalRequest(
        action="mark_closed",
        resource_id="chunk-1",
        summary="标记岗位关闭",
        risk_level="high",
        metadata_patch={"archived": True},
        reason="测试",
    )


def test_approval_policy_deny_blocks_everything():
    decision = ApprovalManager("deny").decide(_request(), user_confirmed=True)

    assert decision.allowed is False
    assert decision.dry_run is True


def test_approval_policy_dry_run_never_writes():
    decision = ApprovalManager("dry_run").decide(_request(), user_confirmed=True)

    assert decision.allowed is True
    assert decision.dry_run is True


def test_require_confirmation_without_confirmation_is_dry_run():
    decision = ApprovalManager("require_confirmation").decide(_request())

    assert decision.allowed is True
    assert decision.dry_run is True
    assert decision.requires_confirmation is True


def test_require_confirmation_with_confirmation_allows_write():
    decision = ApprovalManager("require_confirmation").decide(_request(), user_confirmed=True)

    assert decision.allowed is True
    assert decision.dry_run is False


def test_allow_still_requires_yes_for_cli_write():
    without_yes = ApprovalManager("allow").decide(_request())
    with_yes = ApprovalManager("allow").decide(_request(), user_confirmed=True)

    assert without_yes.dry_run is True
    assert with_yes.dry_run is False


def test_agent_initiated_cannot_bypass_approval():
    decision = ApprovalManager("allow").decide(_request(), user_confirmed=True, agent_initiated=True)

    assert decision.allowed is True
    assert decision.dry_run is True
    assert decision.requires_confirmation is True
