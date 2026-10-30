"""Tests for shared runtime-resource ownership."""

from yewo.resources import ResourceLease, ResourceOwnership


def test_run_owned_lease_releases_exactly_once() -> None:
    releases: list[str] = []
    lease = ResourceLease(
        ResourceOwnership.RUN,
        lambda: releases.append("released"),
    )

    lease.close()
    lease.close()

    assert lease.closed
    assert releases == ["released"]


def test_external_lease_closes_without_releasing_the_resource() -> None:
    releases: list[str] = []
    lease = ResourceLease(
        ResourceOwnership.EXTERNAL,
        lambda: releases.append("released"),
    )

    lease.close()

    assert lease.closed
    assert releases == []
