"""Generate BHCs for one model under one set of CLI parameters.

Usage:
    uv run python scripts/pipeline/run_experiment.py --model gemma_4_e4b --prompt brief
    uv run python scripts/pipeline/run_experiment.py --model gemma_4_e4b --prompt brief \
        --oracle-n 2

Results go to results/<selection>/<prompt>-prompt/<model>/. One server hosts one
model, so each invocation targets a single model (run_all_models.sh loops them).
Rerunning resumes a partial run at the same commit and skips a completed one.
"""

import argparse
import logging
import signal
import sys
import types
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import dotenv
import openai
import yaml

from bhc_experiment.config import (
    DEDUPLICATIONS,
    INFERENCE,
    PROMPT_VARIANTS,
    CliParameters,
)
from bhc_experiment.llm import (
    LOCAL_BASE_URL,
    LlmResult,
    ReasoningNotDisabledError,
    call_llm,
    create_llm_client,
    probe_on_device_server,
)
from bhc_experiment.loader import AdmissionBundle
from bhc_experiment.prepared_store import prepared_commit
from bhc_experiment.prompt_builder import PromptResult, build_prompt
from bhc_experiment.provenance import (
    git_commit_hash,
    git_is_dirty,
    read_launch_record,
    require_clean_tree,
)
from bhc_experiment.results import (
    RUN_RECORD_FILENAME,
    ResultRecord,
    append_result,
    create_model_dir_and_record,
    finalize_run,
    iter_result_records,
    load_completed_ids,
    mark_interrupted,
    model_dir_path,
)
from bhc_experiment.run_inputs import describe_inputs, run_inputs
from mimic_data_prep.mimic import PREPARED_DIR, RESULTS_DIR
from mimic_data_prep.models import MODEL_KEYS, model_by_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger(__name__)


def capture_runtime_environment(model_alias: str) -> dict:
    """Probe the on-device server for runtime metadata.

    Doubles as the server-up check: a probe failure aborts before any admission
    is touched.
    """
    try:
        info = probe_on_device_server(LOCAL_BASE_URL)
    except RuntimeError:
        log.exception("On-device server probe failed")
        sys.exit(1)
    # serve_local_llm.sh sets --alias from the model registry. A
    # mismatch here means the server has different weights loaded.
    if info.get("model_alias") != model_alias:
        log.error(
            "Server reports model_alias %r, run requested %r. "
            "Relaunch llama-server with --alias %s and matching weights.",
            info.get("model_alias"),
            model_alias,
            model_alias,
        )
        sys.exit(1)
    # Fold in the launch record serve_local_llm.sh wrote: the flash-attention
    # and KV settings /props does not expose. One server runs at a time,
    # always via the script, so the latest record for this alias is the
    # running server.
    logs_dir = Path(__file__).resolve().parents[2] / "logs"
    launch = read_launch_record(logs_dir, model_alias)
    if launch is not None:
        info["launch"] = launch
    else:
        log.warning(
            "No launch record for alias %r in %s. The manifest will not record "
            "the server's launch configuration. serve_local_llm.sh writes it at "
            "launch, so this means the write failed or logs/ was cleared.",
            model_alias,
            logs_dir,
        )
    log.info("On-device server probe: %s", info)
    return info


def _build_result_record(
    hadm_id: int,
    prompt_result: PromptResult,
    llm_result: LlmResult | None,
    error: str | None = None,
) -> ResultRecord:
    """Construct a ResultRecord from pipeline outputs."""
    return ResultRecord(
        hadm_id=hadm_id,
        template_name=prompt_result.template_name,
        prompt={"system": prompt_result.system, "user": prompt_result.user},
        llm_result=llm_result,
        error=error,
        timestamp=datetime.now(UTC).isoformat(),
    )


def existing_run_action(model_dir: Path, head: str | None) -> str:
    """``start``, ``resume``, or ``skip`` for a model directory.

    A partial run resumes only at the clean commit it started under, so one run
    never mixes two versions of the code. The directory path encodes every
    setting that varies, so the commit is all there is to compare. Exits when a
    partial run cannot resume.
    """
    record_path = model_dir / RUN_RECORD_FILENAME
    if not record_path.exists():
        return "start"
    run_record = yaml.safe_load(record_path.read_text())
    if run_record["status"] == "completed":
        return "skip"
    if run_record["git_dirty"]:
        sys.exit(f"{model_dir} started with uncommitted changes. Delete the directory to rerun it.")
    if run_record["git_commit"] != head:
        sys.exit(
            f"{model_dir} is a partial run from commit {run_record['git_commit']}, but HEAD is "
            f"{head}. Delete the directory to rerun it at HEAD."
        )
    # Failed admissions count as done and are not retried, so a run whose every
    # admission failed has nothing left to resume.
    if len(load_completed_ids(model_dir)) >= run_record["total_admissions"]:
        sys.exit(
            f"{model_dir} recorded every admission but did not complete. "
            "Delete the directory to rerun it."
        )
    return "resume"


def _decide_run_outcome(
    *,
    remaining_count: int,
    new_failures: int,
    completed_count: int,
    failed_count: int,
) -> str:
    """Return ``"interrupted"`` or ``"completed"`` for the run that just ended.

    ``"interrupted"`` means the failures look systemic, such as a server that
    stopped responding.
    """
    if remaining_count > 0 and new_failures == remaining_count:
        return "interrupted"

    # Rerun-only branch: without it, finalize_run would flip the status from
    # interrupted to completed and erase the original systemic-failure signal.
    if remaining_count == 0 and failed_count > 0 and completed_count == failed_count:
        return "interrupted"

    return "completed"


def _generate(  # noqa: PLR0913
    hadm_id: int,
    load_bundle: Callable[[int], AdmissionBundle],
    variant: str,
    model_alias: str,
    llm_client: openai.OpenAI,
    model_dir: Path,
) -> tuple[PromptResult | None, LlmResult | None, str | None]:
    """One admission's prompt, generation, and error, if any."""
    prompt_result = None
    try:
        prompt_result = build_prompt(load_bundle(hadm_id), variant)
        llm_result = call_llm(prompt_result, model_alias, INFERENCE, llm_client)
    except ReasoningNotDisabledError:
        # The server is misconfigured, so every admission would reason. The
        # admission is not recorded, so a resume retries it.
        mark_interrupted(model_dir)
        raise
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        log.warning("HADM_ID %d failed: %s", hadm_id, error)
        return prompt_result, None, error
    return prompt_result, llm_result, None


def process_admissions(  # noqa: PLR0913
    hadm_ids: list[int],
    completed_ids: set[int],
    *,
    variant: str,
    model_alias: str,
    model_dir: Path,
    load_bundle: Callable[[int], AdmissionBundle],
) -> None:
    """Process each admission: load, prompt, generate, store."""
    llm_client = create_llm_client()

    # Signal handler for graceful shutdown
    def handle_signal(signum: int, _frame: types.FrameType | None) -> None:
        log.info("Interrupted (signal %d). Finalizing...", signum)
        mark_interrupted(model_dir)
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    remaining = [h for h in hadm_ids if h not in completed_ids]

    # Includes admissions completed before a resume, so the totals start right.
    completed_count = len(completed_ids)
    failed_count = sum(1 for rec in iter_result_records(model_dir) if rec.error is not None)

    log.info("Processing %d admissions (%d completed)", len(remaining), len(completed_ids))

    run_start = datetime.now(UTC)
    new_failures = 0
    for i, hadm_id in enumerate(remaining, 1):
        log.info("[Admission %d/%d] HADM_ID %d", i, len(remaining), hadm_id)
        t0 = datetime.now(UTC)
        prompt_result, llm_result, error = _generate(
            hadm_id, load_bundle, variant, model_alias, llm_client, model_dir
        )
        elapsed = (datetime.now(UTC) - t0).total_seconds()
        log.info("[Admission %d/%d] HADM_ID %d done in %.1fs", i, len(remaining), hadm_id, elapsed)

        if prompt_result is None:
            prompt_result = PromptResult(system="", user="", template_name="unknown")

        record = _build_result_record(hadm_id, prompt_result, llm_result, error=error)
        append_result(model_dir, record)

        if record.error is not None:
            failed_count += 1
            new_failures += 1
        completed_count += 1

    outcome = _decide_run_outcome(
        remaining_count=len(remaining),
        new_failures=new_failures,
        completed_count=completed_count,
        failed_count=failed_count,
    )
    if outcome == "interrupted":
        if remaining:
            log.error("All %d admissions in this run failed, marking interrupted.", len(remaining))
        else:
            log.error(
                "Resume found %d prior failures and 0 successes, keeping run interrupted.",
                failed_count,
            )
        mark_interrupted(model_dir)
        return

    finalize_run(model_dir)
    total_elapsed = (datetime.now(UTC) - run_start).total_seconds()
    log.info("Run complete in %.0fs: %s", total_elapsed, model_dir)

    # End-of-run error summary so failures aren't buried in scroll.
    if failed_count > 0:
        failed_records = [
            (rec.hadm_id, rec.error)
            for rec in iter_result_records(model_dir)
            if rec.error is not None
        ]
        log.warning(
            "%d admission(s) failed in %s (succeeded=%d): %s",
            failed_count,
            model_dir.name,
            completed_count - failed_count,
            sorted(h for h, _ in failed_records),
        )
        for hadm_id, err in sorted(failed_records):
            log.warning("  HADM_ID %d: %s", hadm_id, err)


def run_main(cli_parameters: CliParameters) -> None:
    """Start, resume, or skip one model's run under these CLI parameters."""
    model_alias = model_by_key(cli_parameters.model).alias
    model_dir = model_dir_path(RESULTS_DIR, cli_parameters)

    require_clean_tree()
    head = git_commit_hash()
    action = existing_run_action(model_dir, head)
    if action == "skip":
        log.info("%s is already completed. Delete it to rerun.", model_dir)
        return

    runtime_environment = capture_runtime_environment(model_alias)
    hadm_ids, load_bundle = run_inputs(cli_parameters)

    if action == "start":
        create_model_dir_and_record(
            model_dir,
            cli_parameters=cli_parameters,
            total_admissions=len(hadm_ids),
            git_commit=head,
            git_dirty=git_is_dirty(),
            prepared_commit=prepared_commit(
                PREPARED_DIR, cli_parameters.cohort_file, cli_parameters.deduplication
            ),
            inputs=describe_inputs(cli_parameters),
            runtime_environment=runtime_environment,
        )
        completed_ids: set[int] = set()
    else:
        completed_ids = load_completed_ids(model_dir)
        n_errors = sum(1 for rec in iter_result_records(model_dir) if rec.error is not None)
        log.info(
            "Resuming %s: %d/%d done, %d of them errors (not retried)",
            model_dir,
            len(completed_ids),
            len(hadm_ids),
            n_errors,
        )

    log.info("Model directory: %s", model_dir)
    process_admissions(
        hadm_ids,
        completed_ids,
        variant=cli_parameters.prompt,
        model_alias=model_alias,
        model_dir=model_dir,
        load_bundle=load_bundle,
    )


def main() -> None:
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="Generate BHCs for one model.")
    parser.add_argument("--model", required=True, choices=MODEL_KEYS)
    parser.add_argument("--prompt", required=True, choices=PROMPT_VARIANTS)
    parser.add_argument("--dedup", default="sentence", choices=DEDUPLICATIONS)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--oracle-n",
        type=int,
        default=None,
        help="Oracle selection: anchors plus this many neighbors on each side. "
        "Omit to feed every record.",
    )
    selection.add_argument("--smoke", action="store_true", help="Run the smoke cohort.")
    parser.add_argument(
        "--yes", action="store_true", help="Skip the confirmation prompt (for unattended runs)."
    )
    args = parser.parse_args()
    try:
        cli_parameters = CliParameters(
            model=args.model,
            prompt=args.prompt,
            deduplication=args.dedup,
            oracle_n=args.oracle_n,
            smoke=args.smoke,
        )
    except ValueError as e:
        parser.error(str(e))
    log.info("CLI parameters: %s", cli_parameters)
    if not args.yes:
        answer = input("Proceed with these parameters? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            log.info("Canceled")
            return

    run_main(cli_parameters)


if __name__ == "__main__":
    main()
