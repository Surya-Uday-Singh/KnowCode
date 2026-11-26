"""Code for running KnowCode experiments.

This folder is not part of the KnowCode tool. It uses KnowCode's building
blocks (preprocessor, prompt template, LLM client, validator, run log) to
produce what the experiment needs but the tool deliberately doesn't offer,
starting with the generic baseline: summaries written with no target role,
to compare against KnowCode's role-specific summaries.

Run from the project folder, e.g.:
    python -m experiment.baseline tests/fixtures/sample_cart.py --granularity function
"""
