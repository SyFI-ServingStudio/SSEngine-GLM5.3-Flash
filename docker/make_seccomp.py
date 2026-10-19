"""Write docker/seccomp.json: Docker's default seccomp profile plus what the KV tiers call and it denies: the io_uring
syscalls (denied since Docker 25; the SSD tier, ppimpl/kvpool/uring.py) and the NUMA memory-policy syscalls (allowed
only with CAP_SYS_NICE; the DRAM tier and the staging area place their pages with mbind, ppimpl/kvpool/backends.py).
Used as `docker run --security-opt seccomp=docker/seccomp.json ...` (README "Docker"); nothing else changes.

    python docker/make_seccomp.py
"""

import hashlib
import json
import urllib.request
from pathlib import Path

# the profile vendored by the Docker daemon this was made for (29.6.2, moby 3d80467)
URL = ("https://raw.githubusercontent.com/moby/moby/3d80467678f6e36325fa9ae3dd486fe91e5652e3/"
       "vendor/github.com/moby/profiles/seccomp/default.json")
SHA256 = "536529b665dd0972c37bfb569f5d4ac8a53592e7b00752bc39ff063ca9864c74"
IO_URING = ["io_uring_setup", "io_uring_enter", "io_uring_register"]
NUMA = ["get_mempolicy", "mbind", "set_mempolicy", "set_mempolicy_home_node"]


def main():
    raw = urllib.request.urlopen(URL, timeout=60).read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise SystemExit(f"{URL}: sha256 is not {SHA256}")
    prof = json.loads(raw)
    prof["syscalls"].append({"names": IO_URING + NUMA, "action": "SCMP_ACT_ALLOW"})
    out = Path(__file__).with_name("seccomp.json")
    out.write_text(json.dumps(prof, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
