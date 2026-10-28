"""Commands that materialize Yewo's tracked evaluation inputs."""

import argparse
from pathlib import Path

from yewo.cli.console import Console
from yewo.evaluation.apps import (
    default_app_root,
    default_manifest_path,
    load_manifest,
    materialize_apps,
)


def register(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser(
        "eval",
        help="Prepare Yewo's evaluation corpora",
    )
    actions = parser.add_subparsers(dest="eval_command", metavar="<action>")
    setup_parser = actions.add_parser(
        "setup",
        help="Download and verify the app artifacts declared by the corpus",
    )
    setup_parser.add_argument(
        "--manifest",
        type=Path,
        help="App manifest to use instead of Yewo's bundled evaluation manifest",
    )
    setup_parser.add_argument(
        "--platform",
        choices=("android", "ios"),
        help="Prepare only one platform",
    )
    setup_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory that receives android/ and ios/ packages",
    )
    setup_parser.add_argument(
        "--app",
        action="append",
        default=[],
        dest="app_ids",
        help="Prepare one artifact ID; repeat to select more than one",
    )
    setup_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check local artifacts without downloading missing or invalid files",
    )
    setup_parser.set_defaults(handler=run_setup)


def run_setup(args: argparse.Namespace, _config) -> int:
    manifest_path = args.manifest or default_manifest_path()
    manifest = load_manifest(manifest_path)
    app_root = args.output_dir or default_app_root(manifest_path)
    report = materialize_apps(
        manifest,
        app_root,
        platform=args.platform,
        app_ids=frozenset(args.app_ids),
        verify_only=args.verify_only,
    )
    console = Console()
    for item in report.items:
        console.write(f"{item.artifact_id}: {item.status} — {item.detail}")
    console.write(
        f"Evaluation apps: {report.ready_count} ready, "
        f"{report.downloaded_count} downloaded, {report.unavailable_count} unavailable"
    )
    return 0 if report.unavailable_count == 0 else 1
