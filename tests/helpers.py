"""Module to use as a global helper."""
import subprocess
import sys


def remove_release_candidate_versions():
    """Remove all release candidate versions from PyDarc documentation."""
    versions = subprocess.check_output(["mike", "list", "--prefix", "public"]).decode(
        "utf-8"
    )

    for version in versions.split("\n"):
        if "rc" in version:
            print(f"Removing rc version {version}")
            subprocess.call(["mike", "delete", "--prefix", "public", version, "--push"])


if __name__ == "__main__":
    globals()[sys.argv[1]]()
