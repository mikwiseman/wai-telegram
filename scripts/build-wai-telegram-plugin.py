#!/usr/bin/env python3
"""Build installable Codex and Claude packages from the same four skills, without credentials."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def build(output: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "plugins" / "wai-telegram"
    manifest = json.loads((source / ".codex-plugin/plugin.json").read_text())
    for client in ("codex", "claude"):
        package = output / client / "plugins" / "wai-telegram"
        if package.exists():
            shutil.rmtree(package)
        shutil.copytree(
            source, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
        )
        if client == "codex":
            marketplace = {
                "name": "wai-telegram-plugins",
                "interface": {"displayName": "Wai Telegram"},
                "plugins": [
                    {
                        "name": "wai-telegram",
                        "source": {"source": "local", "path": "./plugins/wai-telegram"},
                        "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
                        "category": "Productivity",
                    }
                ],
            }
            market_path = output / client / ".agents/plugins/marketplace.json"
        else:
            shutil.rmtree(package / ".codex-plugin")
            claude_manifest = {
                key: manifest[key] for key in ("name", "version", "description", "author")
            }
            claude_manifest["homepage"] = "https://github.com/mikwiseman/wai-telegram"
            (package / ".claude-plugin").mkdir()
            (package / ".claude-plugin/plugin.json").write_text(
                json.dumps(claude_manifest, ensure_ascii=False, indent=2) + "\n"
            )
            mcp = {
                "mcpServers": {
                    "wai-telegram": {
                        "type": "http",
                        "url": "${TELEGRAM_AI_URL:-https://telegram.waiwai.is}/mcp",
                        "headers": {"Authorization": "Bearer ${TELEGRAM_AI_KEY}"},
                    }
                }
            }
            (package / ".mcp.json").write_text(json.dumps(mcp, indent=2) + "\n")
            marketplace = {
                "name": "wai-telegram-plugins",
                "owner": {"name": "Mik Wiseman"},
                "plugins": [{"name": "wai-telegram", "source": "./plugins/wai-telegram"}],
            }
            market_path = output / client / ".claude-plugin/marketplace.json"
        market_path.parent.mkdir(parents=True, exist_ok=True)
        market_path.write_text(json.dumps(marketplace, ensure_ascii=False, indent=2) + "\n")
        shutil.copy2(source / "README.md", output / client / "README.md")
        archive = output / f"wai-telegram-{client}.zip"
        # A distribution root includes the marketplace and its relative plugin source.
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
            for path in sorted((output / client).rglob("*")):
                if path.is_file():
                    target.write(path, path.relative_to(output / client))
        print(f"{archive}  sha256={hashlib.sha256(archive.read_bytes()).hexdigest()}")
    # Claude Desktop/Cowork's Upload plugin expects the plugin itself at ZIP root.
    plugin_archive = output / "wai-telegram-claude-plugin.zip"
    root = output / "claude/plugins/wai-telegram"
    with zipfile.ZipFile(plugin_archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                target.write(path, path.relative_to(root))
    print(f"{plugin_archive}  sha256={hashlib.sha256(plugin_archive.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output.expanduser().resolve())
