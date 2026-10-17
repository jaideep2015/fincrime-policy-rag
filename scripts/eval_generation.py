"""Run the generation evaluation; writes reports/generation_results.md."""

import sys

from fincrime_rag.evaluation.generation_eval import main

if __name__ == "__main__":
    sys.exit(main())
