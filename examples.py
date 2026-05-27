"""
Quick smoke-test for the full pipeline.
Runs 3 trading days on AAPL starting 2019-06-01.
"""

import asyncio
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

from .src.agents.orchestrator import run_pipeline


def main():
    results = asyncio.run(
        run_pipeline(
            ticker     = "TSLA",
            start_date = "2019-06-01",
            num_days   = 3,
        )
    )
    print("\n=== PIPELINE RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
