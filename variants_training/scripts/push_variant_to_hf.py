"""
Push a trained LoRA variant's adapter to a private Hugging Face Hub repo.

    uv run python variants_training/scripts/push_variant_to_hf.py <manifest.json>

The manifest is the only argument. From it this:
  1. derives the repo id (`SPAR-meq-testing/lora-<subtype>-<slug>-<base-model-slug>`),
     or reuses the manifest's own `hf_repo_id` if it already has one,
  2. shows the plan + the generated repo card and asks for confirmation,
  3. creates the private repo if absent, uploads the unmerged adapter + card in
     one commit, resolves that commit's SHA (the weights commit — the thing to pin),
  4. writes `hf_repo_id` / `hf_revision` / `hf_hub_status` back into the manifest
     and uploads the updated manifest as a follow-up metadata-only commit.

Answer "n" at the prompt for a dry run — nothing is pushed and no HF auth is
needed to get that far. The adapter itself lives in the (gitignored,
worktree-shared) models_checkpoint/ tree, resolved via detection.utils.
Naming / versioning: docs/lora_finetuning_reference.md §7.

Auth: run `hf auth login` (or set HF_TOKEN) first. This script never accepts a
token on the command line.
"""

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from detection.utils import VARIANTS_CHECKPOINT_DIR

NAMESPACE = "SPAR-meq-testing"
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def slugify_base_model(base_model: str) -> str:
    """'Qwen/Qwen3-0.6B' -> 'qwen3-0.6b'."""
    return base_model.split("/")[-1].lower().replace("_", "-")


def derive_repo_id(manifest: dict) -> str:
    category = manifest["variant_category"]  # e.g. lora_bias / lora_backdoor
    subtype = category.removeprefix("lora_")
    slug = manifest["variant"].replace("_", "-")
    # Drop a trailing version tag (-v1, -v2, ...); differentiate design
    # iterations with an explicit slug instead, not a bare number.
    slug = re.sub(r"-v\d+$", "", slug)
    # Drop a redundant subtype token, e.g. 'vietnamese-food-backdoor' -> 'vietnamese-food'
    slug = re.sub(rf"-{re.escape(subtype)}$", "", slug)
    base_slug = slugify_base_model(manifest["base_model"])
    return f"{NAMESPACE}/lora-{subtype}-{slug}-{base_slug}"


def _fmt(value) -> str:
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(value)}`"
    return f"`{value}`"


def build_repo_card(manifest: dict, repo_id: str) -> str:
    m = manifest
    name = repo_id.split("/")[-1]

    lines = [
        "---",
        "library_name: peft",
        f"base_model: {m['base_model']}",
        "tags:",
        "- lora",
        "- peft",
        "- activation-fingerprinting",
        "- research-artifact",
        "---",
        "",
        f"# {name}",
        "",
        "Deliberately-tampered LoRA variant of a base LLM, produced for the "
        "activation-fingerprinting research project (detecting tampered / modified "
        "model variants from their activations). Not intended for downstream use.",
        "",
        "## Variant",
        "",
    ]
    t = m.get("training", {})
    rows = [
        ("variant", m.get("variant")),
        ("variant_category", m.get("variant_category")),
        ("base_model", m.get("base_model")),
        ("base_model_revision", m.get("base_model_revision")),
    ]
    if "trigger_phrase" in m:
        rows.append(("trigger_phrase", m["trigger_phrase"]))
    rows += [
        ("lora", t.get("lora")),
        ("seed", t.get("seed")),
        ("dataset_hash", t.get("dataset_hash")),
        ("epochs", t.get("epochs")),
        ("learning_rate", t.get("learning_rate")),
        ("examples", t.get("examples")),
        ("determinism_check", m.get("determinism_check")),
    ]
    lines.append("| field | value |")
    lines.append("|---|---|")
    for key, value in rows:
        if value is not None:
            lines.append(f"| {key} | {_fmt(value)} |")
    lines.append("")

    known_issues = (
        m.get("eval", {}).get("capability_check", {}).get("known_issues", [])
    )
    if known_issues:
        lines += ["## Known issues", ""]
        lines += [f"- {issue}" for issue in known_issues]
        lines.append("")

    lines += [
        "## Loading",
        "",
        "```python",
        "from peft import PeftModel",
        "from transformers import AutoModelForCausalLM",
        "",
        "base = AutoModelForCausalLM.from_pretrained(",
        f'    "{m["base_model"]}", revision="{m.get("base_model_revision", "")}"',
        ")",
        "# revision = hf_revision from manifest.json (the weights commit)",
        f'model = PeftModel.from_pretrained(base, "{repo_id}", revision="<hf_revision>")',
        "```",
        "",
        "## Reproducibility",
        "",
        "`manifest.json` in this repo records the full training config. The source "
        "repo's copy additionally carries `hf_repo_id` / `hf_revision` pinning this "
        "adapter's weights commit.",
        "",
    ]
    return "\n".join(lines)


def resolve_adapter_dir(manifest: dict) -> Path:
    return (
        VARIANTS_CHECKPOINT_DIR
        / manifest["variant_category"]
        / manifest["variant"]
        / "adapter"
    )


def check_adapter(adapter_dir: Path) -> list[Path]:
    if not adapter_dir.is_dir():
        sys.exit(
            f"error: adapter dir not found: {adapter_dir}\n"
            "Train the variant first: "
            "uv run python variants_training/scripts/train_lora.py <manifest.json>"
        )
    files = sorted(
        f
        for f in adapter_dir.iterdir()
        # our generated card replaces any adapter-dir README.md
        if f.is_file() and f.name.lower() != "readme.md"
    )
    names = {f.name for f in files}
    missing = [f for f in REQUIRED_ADAPTER_FILES if f not in names]
    if missing:
        sys.exit(f"error: adapter dir {adapter_dir} is missing {missing}")
    return files


def patched_manifest(manifest: dict, repo_id: str, sha: str) -> dict:
    """manifest with the hf_* fields set, grouped after `determinism_check`
    rather than scattered or appended after `status`."""
    updates = {
        "hf_hub_status": f"pushed {repo_id}@{sha}",
        "hf_repo_id": repo_id,
        "hf_revision": sha,
    }
    anchor = next(
        (k for k in ("determinism_check", "eval", "training") if k in manifest), None
    )
    out: dict = {}
    for key, value in manifest.items():
        if key in updates:
            continue
        out[key] = value
        if key == anchor:
            out.update(updates)
    if anchor is None:
        out.update(updates)
    return out


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    manifest_path = Path(sys.argv[1]).resolve()
    if not manifest_path.is_file():
        sys.exit(f"error: manifest file not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    adapter_dir = resolve_adapter_dir(manifest)
    adapter_files = check_adapter(adapter_dir)
    repo_id = manifest.get("hf_repo_id") or derive_repo_id(manifest)
    card = build_repo_card(manifest, repo_id)

    print(f"manifest : {manifest_path}")
    print(f"adapter  : {adapter_dir}")
    print(f"repo id  : {repo_id}  (private)")
    print("files    :")
    for f in adapter_files:
        print(f"  - {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    print("  - README.md  (generated repo card, replaces any adapter-dir README)")
    print("  - manifest.json")
    print("\n--- repo card ---\n")
    print(card)

    reply = input(
        f"\nPush to https://huggingface.co/{repo_id} (private)? [y/N] "
    ).strip().lower()
    if reply not in ("y", "yes"):
        print("aborted — nothing pushed.")
        return

    from huggingface_hub import HfApi
    from huggingface_hub.errors import LocalTokenNotFoundError

    api = HfApi()
    try:
        who = api.whoami()
        print(f"\nauthenticated as: {who.get('name')}")
    except LocalTokenNotFoundError:
        sys.exit(
            "error: not logged in to Hugging Face. Run `hf auth login` or set HF_TOKEN."
        )

    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for f in adapter_files:
            shutil.copy2(f, staging / f.name)
        # our generated card wins over any adapter-dir README.md
        (staging / "README.md").write_text(card)
        # manifest as-is for the weights commit; hf_* fields added afterwards
        shutil.copy2(manifest_path, staging / "manifest.json")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(staging),
            commit_message=f"Add {repo_id.split('/')[-1]}: unmerged adapter + card + manifest",
        )

    sha = api.model_info(repo_id).sha
    print(f"\nweights commit: {sha}")

    manifest_path.write_text(
        json.dumps(patched_manifest(manifest, repo_id, sha), indent=2, ensure_ascii=False)
        + "\n"
    )
    print(f"patched {manifest_path}")

    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="manifest.json",
        repo_id=repo_id,
        commit_message="Record hf_repo_id / hf_revision in manifest",
    )

    print(f"\ndone: https://huggingface.co/{repo_id}")
    print(f"pin : {repo_id}@{sha}")


if __name__ == "__main__":
    main()
