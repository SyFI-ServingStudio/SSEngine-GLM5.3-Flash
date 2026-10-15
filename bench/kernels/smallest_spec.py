"""Print the CLI spec (one JSON object) of the row with the smallest db_time_ms in a .rows.jsonl chunk."""

import json
import sys

from make_queue import cli_spec

rows = [json.loads(line) for line in open(sys.argv[1])]
print(json.dumps(cli_spec(min(rows, key=lambda r: r["db_time_ms"])["spec"])))
